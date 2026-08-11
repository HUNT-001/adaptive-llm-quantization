"""Mobile AI backend — Anvil retargeted to on-device LLM inference.

This is the Mobile-track counterpart to `profiler.py`. The planner, baselines,
objective, and search harness do not change; only the profiler backend and the
*meaning* of the metric fields do (see docs/BUILD_PLAN_MOBILE.md):

    throughput  -> generation tokens/sec        (higher is better)
    latency_ms  -> time-to-first-token in ms     (lower is better)
    memory_mb   -> GGUF size on disk in MB        (lower is better)
    accuracy    -> quality = 1 - dPerplexity/base (>= floor to ship)

Because minimizing per-token latency is equivalent to maximizing tokens/sec,
the existing constrained objective works unchanged; the only addition is an
optional size budget (Objective.size_budget_mb).

Two backends, mirroring the cloud module:
  * MobileMockProfiler - analytic tokens/sec model. No phone needed. Lets us
                         develop and unit-test the whole loop offline. Its TRUE
                         cost constants deliberately differ from the planner's
                         surrogate priors, so online calibration does real work.
  * MobileProfiler     - the real backend: builds a per-layer-quantized GGUF
                         with `llama-quantize --tensor-type`, scores quality
                         with `llama-perplexity` on the host, and measures speed
                         with `llama-bench` on the phone over adb. This is the
                         only class that touches the device; the planner does
                         not change.
"""
from __future__ import annotations

import json
import os
import random
import re
import subprocess
from typing import Callable, Dict, List, Optional, Tuple

from .model import Block, Metrics, ModelSpec, OptState
from .profiler import Profiler


# ---------------------------------------------------------------------------
# Model spec for a small on-device LLM (development against the mock).
# ---------------------------------------------------------------------------
# On a real device you build this from one profiling sweep of your GGUF:
# per-layer tensor bytes -> base_memory_mb, per-token time -> base_latency_ms,
# a quick per-layer quant probe -> quant_sensitivity. Blocks are marked
# fusible=False because llama.cpp's graph is fixed, so the action space is
# per-layer precision + thread count only (no fusion toggle) -- still 3^N,
# combinatorial, which is what gives the planner something to prove.

def mobile_llm_spec(n_layers: int = 16, seed: int = 7) -> ModelSpec:
    """A transformer-shaped spec sized like a ~1B on-device LLM (e.g.
    Llama-3.2-1B / Qwen2.5-1.5B). base_latency_ms is per-token decode time in
    ms at F16; base_memory_mb is that block's F16 weight size."""
    rng = random.Random(seed)
    blocks: List[Block] = []
    # token embedding: large weight, very quant-sensitive, cheap compute
    blocks.append(Block("embed", base_latency_ms=0.9, base_memory_mb=252.0,
                        quant_sensitivity=0.9, fusible=False))
    for i in range(n_layers):
        # attention: quant-sensitive; MLP: robust and the bulk of the weight
        blocks.append(Block(
            f"attn_{i}",
            base_latency_ms=rng.uniform(0.8, 1.4),
            base_memory_mb=rng.uniform(24, 40),
            quant_sensitivity=rng.uniform(0.55, 0.85),
            fusible=False,
        ))
        blocks.append(Block(
            f"mlp_{i}",
            base_latency_ms=rng.uniform(1.3, 2.1),
            base_memory_mb=rng.uniform(60, 96),
            quant_sensitivity=rng.uniform(0.05, 0.25),
            fusible=False,
        ))
    # lm head: ties to embedding size, moderately sensitive
    blocks.append(Block("lm_head", base_latency_ms=1.1, base_memory_mb=252.0,
                        quant_sensitivity=0.5, fusible=False))
    return ModelSpec(name=f"mobile_llm_{n_layers}L", blocks=blocks,
                     base_accuracy=1.0)


# ---------------------------------------------------------------------------
# Analytic mock of the phone. TRUE constants differ from the surrogate priors
# in planner.py (_SPEEDUP/_MEM/_ACC_PENALTY, serial 0.10) on purpose.
# ---------------------------------------------------------------------------
# Decode on a phone is largely memory-bandwidth bound, so quantization buys a
# lot of speed and threads help little (high serial fraction). None of these
# true values are visible to the planner; it starts from its generic priors and
# calibrates toward these from measurements.
_TRUE_SPEEDUP = {"fp16": 1.00, "int8": 0.66, "int4": 0.44}   # per-token latency
_TRUE_MEM = {"fp16": 1.00, "int8": 0.50, "int4": 0.28}       # Q4_K ~ 4.5 bpw
# Per-block quality penalty (before per-block sensitivity weighting). Q8_0 is
# near-lossless, so int8 is cheap; Q4_K costs more, and costs *most* on the
# quant-sensitive blocks (attention/embeddings) -- which is exactly the trap
# greedy falls into and the planner navigates around.
_TRUE_ACC_PENALTY = {"fp16": 0.000, "int8": 0.0015, "int4": 0.012}
_TRUE_SERIAL = 0.35   # bandwidth-bound decode: extra threads barely help


class MobileMockProfiler(Profiler):
    """Analytic surrogate of the Z7s. Deterministic given a seed. Returns
    Metrics whose fields carry the Mobile meanings documented above."""

    def __init__(self, spec: ModelSpec, noise: float = 0.03, seed: int = 0):
        super().__init__()
        self.spec = spec
        self.noise = noise
        self._rng = random.Random(seed)

    def _thread_scaling(self, threads: int) -> float:
        return _TRUE_SERIAL + (1 - _TRUE_SERIAL) / threads

    def _measure(self, state: OptState) -> Metrics:
        spec = self.spec
        per_token_ms = 0.0
        size_mb = 0.0
        quality_loss = 0.0
        for i, blk in enumerate(spec.blocks):
            prec = state.precision[i]
            per_token_ms += blk.base_latency_ms * _TRUE_SPEEDUP[prec]
            size_mb += blk.base_memory_mb * _TRUE_MEM[prec]
            quality_loss += _TRUE_ACC_PENALTY[prec] * (0.3 + 1.4 * blk.quant_sensitivity)
        per_token_ms *= self._thread_scaling(state.threads)
        per_token_ms *= 1.0 + self._rng.uniform(-self.noise, self.noise)
        quality = max(0.0, spec.base_accuracy - quality_loss)
        tokens_per_sec = 1000.0 / per_token_ms
        # latency_ms = TTFT proxy (one token of prefill), memory_mb = size,
        # accuracy = quality, throughput = tokens/sec.
        return Metrics(latency_ms=per_token_ms, memory_mb=size_mb,
                       accuracy=quality, throughput=tokens_per_sec)


# ---------------------------------------------------------------------------
# Real backend: the llama.cpp binaries you built in Termux on the Z7s.
# ---------------------------------------------------------------------------
# GGUF quant type each precision label maps to. CRITICAL Arm finding (measured
# on the Z7s): KleidiAI only provides accelerated kernels for Q4_0 and Q8_0 --
# NOT the K-quants (Q4_K/Q6_K). llama-perplexity prints, verbatim:
#   "kleidiai: no kernel for tensor type q4_K, not accelerated by KleidiAI
#    (kernels available for Q4_0 and Q8_0)"
# That is why Q8_0 prefills at ~67 t/s while the popular Q4_K_M default manages
# only ~29 t/s on the same phone. So Anvil's precision levels are the two
# KleidiAI-accelerated types: int8 -> Q8_0, int4 -> Q4_0. Both also avoid the
# K-quant "columns divisible by 256" constraint, so per-layer overrides apply
# cleanly on any model.
_GGUF_TYPE = {"fp16": "F16", "int8": "Q8_0", "int4": "Q4_0"}


def _parse_bench_json(stdout: str) -> Tuple[float, float, int]:
    """llama-bench -o json -> (generation t/s, prefill t/s, model_size bytes).

    The JSON is an array of runs; the prefill run has n_gen == 0 (n_prompt > 0)
    and the generation run has n_prompt == 0 (n_gen > 0). `avg_ts` is the
    tokens/sec for each. Verified against real Z7s output."""
    data = json.loads(stdout)
    gen_ts = prefill_ts = 0.0
    size_bytes = 0
    for e in data:
        size_bytes = e.get("model_size", size_bytes)
        if e.get("n_gen", 0) > 0:
            gen_ts = float(e["avg_ts"])
        elif e.get("n_prompt", 0) > 0:
            prefill_ts = float(e["avg_ts"])
    return gen_ts, prefill_ts, size_bytes


def _parse_perplexity(text: str) -> float:
    """Pull the final PPL out of llama-perplexity output
    ('Final estimate: PPL = 12.34 +/- 0.56')."""
    m = re.search(r"PPL\s*=\s*([0-9]+\.?[0-9]*)", text)
    if not m:
        raise ValueError("no 'PPL =' found in llama-perplexity output")
    return float(m.group(1))


class MobileProfiler(Profiler):
    """Real Mobile backend, runnable entirely in Termux on the Z7s (no adb when
    Anvil runs on the phone). Per config:

      1. _apply_config -> `llama-quantize` with per-block `--tensor-type`
         overrides -> a mixed-precision GGUF, cached by state key.
      2. _bench        -> `llama-bench -o json` -> generation t/s + prefill t/s
         + on-disk size.
      3. _quality      -> `llama-perplexity` vs the F16 base perplexity
         (optional; quality is config-only so you may also run it on a laptop
         build and inject base_perplexity/ppl_text). Without it, runs speed-only
         (quality fixed at 1.0) -- set ppl_text + base_perplexity to enforce the
         quality floor.

    Metric fields carry the Mobile meanings: throughput = generation t/s,
    latency_ms = 1000/prefill_t_s (a TTFT proxy, lower better), memory_mb = GGUF
    size, accuracy = quality.
    """

    def __init__(
        self,
        spec: ModelSpec,
        base_gguf: str,                        # F16/BF16 GGUF to quantize from
        bin_dir: str = "~/llama.cpp/build/bin",
        base_type: str = "Q8_0",               # majority type for non-blk tensors
        ppl_text: Optional[str] = None,        # text file for llama-perplexity
        base_perplexity: Optional[float] = None,
        work_dir: str = ".",
        keep_gguf: bool = False,               # keep each config's GGUF on disk?
    ):
        super().__init__()
        self.spec = spec
        self.base_gguf = os.path.expanduser(base_gguf)
        self.bin_dir = os.path.expanduser(bin_dir)
        self.base_type = base_type
        self.ppl_text = ppl_text
        self.base_perplexity = base_perplexity
        self.work_dir = os.path.expanduser(work_dir)
        self.keep_gguf = keep_gguf

    def _bin(self, name: str) -> str:
        return os.path.join(self.bin_dir, name)

    def _tensor_type_args(self, state: OptState) -> List[str]:
        """OptState.precision -> repeated `--tensor-type blk\\.i\\..*=TYPE`."""
        args: List[str] = []
        for i, prec in enumerate(state.precision):
            args += ["--tensor-type", f"blk\\.{i}\\..*={_GGUF_TYPE[prec]}"]
        return args

    def _apply_config(self, state: OptState) -> str:
        """Build (and cache) the per-layer-quantized GGUF; return its path."""
        out = os.path.join(self.work_dir, f"anvil_{state.key()}.gguf")
        if not os.path.exists(out):
            subprocess.run(
                [self._bin("llama-quantize"), *self._tensor_type_args(state),
                 self.base_gguf, out, self.base_type],
                check=True, capture_output=True, text=True)
        return out

    def _bench(self, gguf: str, state: OptState) -> Tuple[float, float, int]:
        out = subprocess.run(
            [self._bin("llama-bench"), "-m", gguf,
             "-t", str(state.threads), "-o", "json"],
            check=True, capture_output=True, text=True)
        return _parse_bench_json(out.stdout)

    def _quality(self, gguf: str) -> float:
        if self.ppl_text is None or self.base_perplexity is None:
            return 1.0  # speed-only mode
        out = subprocess.run(
            [self._bin("llama-perplexity"), "-m", gguf, "-f", self.ppl_text],
            check=True, capture_output=True, text=True)
        ppl = _parse_perplexity(out.stdout + out.stderr)
        return max(0.0, 1.0 - (ppl - self.base_perplexity) / self.base_perplexity)

    def _measure(self, state: OptState) -> Metrics:  # pragma: no cover
        gguf = self._apply_config(state)
        try:
            gen_ts, prefill_ts, size_bytes = self._bench(gguf, state)
            quality = self._quality(gguf)
        finally:
            # Metrics are cached by state.key() in the base class, so we never
            # rebuild the same config -- the GGUF is disposable. Deleting it
            # keeps dozens of ~1 GB configs from filling the phone.
            if not self.keep_gguf and gguf != self.base_gguf and os.path.exists(gguf):
                os.remove(gguf)
        return Metrics(
            latency_ms=(1000.0 / prefill_ts) if prefill_ts else float("inf"),
            memory_mb=size_bytes / (1024.0 * 1024.0),
            accuracy=quality,
            throughput=gen_ts,
        )
