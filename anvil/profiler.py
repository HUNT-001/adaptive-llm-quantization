"""Profiler backends.

This module defines the single boundary between Anvil's planner and the real
world. The planner only ever calls `Profiler.evaluate(state) -> Metrics` and
reads `Profiler.evaluations` (the budget spent). Everything above this line is
hardware-agnostic.

Two backends:
  * MockProfiler   - an analytic cost model. No hardware needed. Lets us
                     develop, unit-test, and prove the planner-beats-baseline
                     result on a laptop. Adds noise so it behaves like a real
                     measurement (repeatable configs are cached like real ones).
  * PerformixProfiler - drives Arm Performix's `apx` CLI over SSH against the
                     Oracle Ampere target and parses real latency/memory. This
                     is the ONLY class that changes when moving to hardware; the
                     planner does not.

Because the interface is identical, the exact experiment we validate on the
mock re-runs unchanged on the Arm box.
"""
from __future__ import annotations

import random
import subprocess
from typing import Dict, Optional

from .model import Metrics, ModelSpec, OptState


# The mock's *true* hardware constants. These deliberately DIFFER from the
# surrogate's initial priors in planner.py (_SPEEDUP/_MEM/_ACC_PENALTY, serial
# 0.10, fusion 0.93). That gap is the point: it forces the planner's online
# calibration to actually do work — exactly as it must on a real Arm target,
# where the analytic prior is only a starting guess and Performix measurements
# correct it. If the mock and surrogate shared constants, the surrogate would
# be a perfect oracle and "update the world model" would be theater. Here the
# mock is a bit slower to quantize (int gains smaller), scales worse across
# threads, fuses slightly better, and loses a bit more accuracy to quant than
# the prior assumes; calibration recovers the difference.
_TRUE_SPEEDUP = {"fp16": 1.00, "int8": 0.68, "int4": 0.52}  # latency multiplier
_TRUE_MEM = {"fp16": 1.00, "int8": 0.50, "int4": 0.25}      # memory (bit-exact)
_TRUE_ACC_PENALTY = {"fp16": 0.000, "int8": 0.014, "int4": 0.075}
_TRUE_SERIAL = 0.14     # fraction of work that stays serial (prior assumes 0.10)
_TRUE_FUSION = 0.90     # latency multiplier when fused (prior assumes 0.93)


class Profiler:
    """Interface. Counts every real evaluation and caches by state key so no
    configuration is ever paid for twice (mirrors how you'd treat a costly
    on-device benchmark)."""

    def __init__(self) -> None:
        self.evaluations = 0
        self._cache: Dict[str, Metrics] = {}

    def evaluate(self, state: OptState) -> Metrics:
        key = state.key()
        if key in self._cache:
            return self._cache[key]
        self.evaluations += 1
        m = self._measure(state)
        self._cache[key] = m
        return m

    def _measure(self, state: OptState) -> Metrics:  # pragma: no cover
        raise NotImplementedError


class MockProfiler(Profiler):
    """Analytic surrogate of a real Arm target. Deterministic given a seed."""

    def __init__(self, spec: ModelSpec, noise: float = 0.02, seed: int = 0):
        super().__init__()
        self.spec = spec
        self.noise = noise
        self._rng = random.Random(seed)

    def _thread_scaling(self, threads: int) -> float:
        # Amdahl-ish: diminishing returns. The true serial fraction is larger
        # than the surrogate's prior assumes, so threads help less than hoped.
        return _TRUE_SERIAL + (1 - _TRUE_SERIAL) / threads

    def _measure(self, state: OptState) -> Metrics:
        spec = self.spec
        latency = 0.0
        memory = 0.0
        acc_loss = 0.0
        for i, blk in enumerate(spec.blocks):
            prec = state.precision[i]
            l = blk.base_latency_ms * _TRUE_SPEEDUP[prec]
            if state.fused[i]:
                l *= _TRUE_FUSION  # fusion removes launch/round-trip overhead
            latency += l
            memory += blk.base_memory_mb * _TRUE_MEM[prec]
            # Sensitive blocks lose more accuracy when quantized. This is what
            # traps greedy: quantizing a sensitive block looks locally bad, but
            # doing it while compensating elsewhere can be globally optimal.
            acc_loss += _TRUE_ACC_PENALTY[prec] * (0.3 + 1.4 * blk.quant_sensitivity)
        latency *= self._thread_scaling(state.threads)
        # multiplicative measurement noise, like a real benchmark
        latency *= 1.0 + self._rng.uniform(-self.noise, self.noise)
        accuracy = max(0.0, spec.base_accuracy - acc_loss)
        throughput = 1000.0 / latency  # inferences/sec from ms
        return Metrics(latency, memory, accuracy, throughput)


class PerformixProfiler(Profiler):
    """Real backend: runs the model on the Arm target and reads metrics from
    Arm Performix's CLI (`apx`).

    Setup (see docs/BUILD_PLAN.md):
      * Oracle Ampere A1 target on Ubuntu 22.04/24.04 (Neoverse N1).
      * SSH key auth + passwordless sudo (Performix needs elevated perf access).
      * `apx` installed on this host; target reachable as `apx` connection.

    This is a scaffold: fill `_run_workload` with your model's inference command
    and `_parse` with the apx report fields once you see real output on day 1.
    """

    def __init__(
        self,
        spec: ModelSpec,
        target: str,
        workload_cmd: str,
        apx_bin: str = "apx",
        accuracy_fn=None,
    ):
        super().__init__()
        self.spec = spec
        self.target = target            # e.g. "ubuntu@<oracle-ip>"
        self.workload_cmd = workload_cmd
        self.apx_bin = apx_bin
        # accuracy is model-specific (perplexity / task eval); injected so the
        # profiler stays generic. Must return a float given an OptState.
        self.accuracy_fn = accuracy_fn

    def _apply_config_remote(self, state: OptState) -> None:
        """Materialize `state` on the target: write the quant/fusion/thread
        config the workload will pick up. Implement for your model export.
        TODO(day 2): translate OptState -> your runtime's config file/flags."""
        raise NotImplementedError("wire OptState to your model export")

    def _run_and_profile(self, state: OptState) -> str:
        """Run the workload under Performix and return the raw apx report.
        TODO(day 1): confirm the exact `apx` subcommand + flags against your
        target, then capture stdout here."""
        cmd = [
            self.apx_bin, "record", "--target", self.target,
            "--", *self.workload_cmd.split(),
        ]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return out.stdout

    def _parse(self, report: str, state: OptState) -> Metrics:
        """Parse latency/memory/throughput out of the apx report.
        TODO(day 1): map the real field names once you see one report."""
        raise NotImplementedError("parse apx report fields")

    def _measure(self, state: OptState) -> Metrics:  # pragma: no cover
        self._apply_config_remote(state)
        report = self._run_and_profile(state)
        return self._parse(report, state)
