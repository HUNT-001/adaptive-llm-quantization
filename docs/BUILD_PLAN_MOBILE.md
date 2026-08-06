# Anvil (Mobile) — Build Plan

**Target:** Arm Create: AI Optimization Challenge 2026, **Mobile AI** track.
**Deadline:** 14 Aug 2026. **Budget:** $0. **Device:** iQOO Z7s (Snapdragon 695,
2×Cortex-A78 + 6×Cortex-A55, 8 GB RAM, Android 13, ARM64).
**Judging axes:** Tech · WOW · Impact · DevX.

This plan repositions Anvil to Mobile AI. The planner does not
change — only the profiler backend and the metrics it optimizes. Everything in
`docs/BUILD_PLAN.md` about the offline engine still holds; this document
supersedes the *hardware* half.

## 1. The pitch (why this wins in Mobile AI)

Everyone in the Mobile track ships "we quantized a model to Q4 and it runs on a
phone." Anvil ships an **agent that decides, per layer, how to quantize a model
so it runs best on *your specific phone*** — and explains each decision. It
treats on-device optimization as sequential decision-making: predict which
per-layer precision change is worth measuring, apply it with `llama-quantize`,
benchmark it on the Z7s, calibrate, repeat — converging on a better
size/speed/quality trade-off in far fewer on-device benchmarks than the
quantize-and-pray loop.

Why it scores:

- **Tech:** real per-layer mixed-precision quantization realized through
  llama.cpp's `--tensor-type`, KleidiAI-accelerated INT8/INT4 kernels, and a
  surrogate-guided beam search with online calibration against *real device
  measurements*.
- **WOW:** the agent narrates its plan, then you watch tokens/sec climb and the
  model shrink on a real mid-range phone — not a flagship. "It tuned this model
  to my ₹18k phone" is a stronger demo than one more Q4 benchmark.
- **Impact:** ships as a reusable CLI any Arm-mobile dev points at their own
  model + device. The optimizer, not a single artifact, is the deliverable.
- **DevX:** one-command reproduction; works with zero hardware via the mock
  backend; the on-device path is documented end to end.

The defensible novelty claim is unchanged: *strategic search over the
per-layer optimization space reaches a better feasible configuration in fewer
benchmark evaluations than random or greedy search* — now proven with real
tokens/sec and time-to-first-token on an Arm phone.

## 2. What changes from the Cloud plan (and what does not)

**Unchanged:** `model.py`, `actions.py`, `planner.py`, `baselines.py`,
`search.py`, `objective.py` (form), the beam search, the surrogate + online
calibration, the whole comparison harness and its tests.

**Changed:** one class — the profiler backend — plus what the metrics *mean*.

```
OptState (per-layer precision + threads)
      │
      ▼  MobileProfiler.evaluate()
  llama-quantize --tensor-type ...   →  build a per-layer-quantized GGUF (host)
      │
      ├─ quality: llama-perplexity on host  (hardware-independent → cheap)
      └─ speed:   adb push GGUF → llama-bench on the Z7s  (the only on-device cost)
      │
      ▼
  Metrics(tokens_per_sec, ttft_ms, size_mb, quality)
```

Key insight that halves the on-device cost: **model quality is
device-independent.** Perplexity of a given GGUF is the same everywhere, so
compute it once on the laptop; only *speed* (tokens/sec, TTFT) needs the phone.
The expensive on-device loop shrinks to a single `llama-bench` run per config.

## 3. Metrics and objective (the Mobile reframing)

The Mobile track explicitly rewards: model size on disk/in memory, model
quality for a given size, and model speed (tokens/sec, TTFT). Anvil optimizes
exactly these. Reinterpret `Metrics`:

| Field (code) | Mobile meaning | Source |
|---|---|---|
| `throughput` | generation tokens/sec | `llama-bench` tg on the Z7s |
| `latency_ms` | time-to-first-token (ms) | `llama-bench` pp / prefill on the Z7s |
| `memory_mb`  | GGUF size on disk (MB) | `stat` the quantized file |
| `accuracy`   | quality = `1 − (Δperplexity / base_perplexity)` | `llama-perplexity` on host |

**Objective:** maximize generation tokens/sec, subject to (a) a **quality
floor** (perplexity increase under a set %, e.g. quality ≥ 0.95) and (b) a
**size budget** (fits comfortably in 8 GB and under a target on-disk size). This
is the current constrained objective with the latency term replaced by a
throughput term and a second constraint (size) added — a small edit to
`objective.py`, not a rewrite.

## 4. The realized action space on llama.cpp

Actions stay **per-block**, which keeps the space combinatorial and gives the
planner something to prove. On llama.cpp they map to concrete quantize flags:

- **Per-layer precision** — `PRECISIONS` map to GGUF quant types, e.g.
  `fp16 → F16`, `int8 → Q8_0`, `int4 → Q4_K_M`. `OptState.precision[i]` for
  block *i* becomes a `--tensor-type "blk\.i\..*=<TYPE>"` override passed to
  `llama-quantize` (the flag takes regex and may be repeated per tensor group).
  A model with N blocks over 3 precisions is 3^N configurations — still far too
  large to brute force.
- **Threads** — `THREAD_OPTIONS` becomes `(2, 4, 6, 8)` for the Z7s (2 big
  A78 + 6 little A55); passed as llama.cpp `-t`. The planner finds the sweet
  spot instead of assuming "more is better" (little cores often hurt).
- **Fusion → repurpose.** llama.cpp's graph is fixed, so the per-block "fuse"
  toggle from the cloud version isn't meaningful. Replace it with a global
  runtime flag the planner can toggle (e.g. flash-attention on/off, or mmap),
  or drop it and rely on per-layer quant + threads — which alone is a rich
  combinatorial space. Recommended: drop fuse for v1, list attention/kernel
  flags as future work.

## 5. Day 1–2 — de-risk on the phone (do this before anything else)

Exit criterion: **one real tokens/sec + TTFT number, measured on the Z7s, for a
per-layer-quantized model you built.** Everything else builds on that.

1. **Build llama.cpp on the phone (Termux path, simplest).** Install Termux,
   then `pkg install clang make cmake git`, clone llama.cpp, and build with the
   Arm CPU backend (KleidiAI kernels are in-tree for INT4/INT8). Confirm
   `llama-cli` and `llama-bench` run.
2. **Pick a small model.** Something that fits 8 GB with headroom and still
   shows quant deltas — e.g. Llama-3.2-1B or Qwen2.5-1.5B. Get an F16 GGUF as
   the un-quantized baseline.
3. **Prove per-layer quant works.** On the laptop, run `llama-quantize` with a
   single `--tensor-type` override on one block and confirm the output GGUF
   differs in size and still loads. This validates `_apply_config_remote`'s core
   mechanism.
4. **Capture one on-device benchmark.** `adb push` the GGUF to the phone (or
   build/run in Termux), run `llama-bench`, and record prefill + generation
   tokens/sec and the file size. Save the raw output — those field names are
   what `_parse` reads.
5. **Capture one perplexity number on the host** for the same GGUF, so the
   quality half of the loop is proven too.

If Termux fights you, fallback: cross-compile llama.cpp with the Android NDK on
the Windows laptop and push the binary + model via `adb`. Same interface.

## 6. Implementation deltas (the concrete code work)

1. **`profiler.py` → add `MobileProfiler(Profiler)`.**
   - `_apply_config_remote(state)`: translate `OptState.precision` into
     `--tensor-type` regex overrides, call `llama-quantize` on the host to
     produce a per-layer GGUF, cache by state key (reuse the base F16).
   - `quality_fn(gguf)`: run `llama-perplexity` on the host; convert to the
     `quality` field. (Host-side, cheap, cache it.)
   - `_run_and_profile(state)`: `adb push` the GGUF, run `llama-bench` on the
     Z7s with the state's thread count, return raw stdout.
   - `_parse(report)`: pull tokens/sec + TTFT into `Metrics`.
2. **`model.py`** — set `THREAD_OPTIONS = (2, 4, 6, 8)`; `PRECISIONS` already
   fine (labels map to GGUF types in the profiler).
3. **`objective.py`** — swap the latency-reduction term for a throughput-gain
   term; add the size-budget constraint alongside the quality floor.
4. **`models.py`** — build the `ModelSpec` from one profiling sweep of the real
   model (per-layer tensor sizes → `base_memory_mb`; a quick per-layer quant
   probe → `quant_sensitivity`).
5. **New `MobileMockProfiler`** — analytic tokens/sec model for offline dev and
   CI, keeping the deliberate mock-vs-surrogate discrepancy so calibration still
   does real work with no phone attached.
6. **Tests** — mirror the existing suite against the mobile mock (budget,
   feasibility, beats-baselines across seeds, termination, calibration).

## 7. The headline experiment (unchanged shape, real device data)

`experiments/run_experiment.py` runs **Anvil vs greedy vs random on the same
on-device benchmark budget**, same model, same quality floor + size budget, and
produces the convergence chart (best feasible speed vs evaluations), the Pareto
chart (tokens/sec vs quality, with the floor drawn), and `results.json` with
Anvil's decision trace. On-device benchmarks are slow (quantize + push + bench
is minutes each), so keep the budget modest — **20–40 evals** — which is exactly
the regime where the surrogate earns its keep.

## 8. Timeline (~19 days, compressed — start today)

- **Days 1–2:** Section 5 phone de-risk. One real tokens/sec number + one
  perplexity number.
- **Days 3–5:** implement `MobileProfiler` + the objective/metric edits + the
  mobile mock; get the baseline + a couple of hand configs measured on the Z7s.
- **Days 6–12:** run the headline experiment on-device; land "Anvil beats
  greedy/random on the same benchmark budget" with real phone numbers; tune beam
  width / candidates per round for the small budget.
- **Days 13–17:** package CLI + README, fill the dashboard with real results,
  record the <3-min video (narrate the plan, then show tokens/sec climbing and
  the model shrinking on the phone).
- **Days 18–19:** buffer; verify one-command reproduction; submit.

## 9. Risk register (mobile-specific)

| Risk | Mitigation |
|---|---|
| Thermal throttling skews tokens/sec on the SD695 | Median of N runs, cool-down between configs, report steady-state; note throttling explicitly. |
| On-device eval is slow (quantize+push+bench) | Small model + budget 20–40; surrogate measures few candidates; quality computed on host, not phone. |
| SD695 has no SME2/i8mm | Gains come from dotprod INT8 + KleidiAI, not SME2 — real and honest; set expectations in the writeup. |
| Per-layer quant granularity limited by GGUF tensor grouping | Use `--tensor-type` per attention/ffn tensor group; if a block can't be split cleanly, treat the block as the unit (still combinatorial). |
| 8 GB RAM ceiling | Keep quantized model well under ~2 GB; the size-budget constraint enforces it. |
| Repro fails for judges | `--out` writes everything; mock backend needs no phone; README one-command run. |

## 10. Scope discipline

**In scope:** the agent/CLI; the beats-baselines Pareto result with real Z7s
tokens/sec + TTFT; a reproducible repo + <3-min video.

**Future-work slides:** attention/kernel-flag actions, SME2 devices, ExecuTorch
backend alongside llama.cpp, a learned surrogate/RL policy, multi-device tuning.

## 11. References

- Mobile track learning paths (Arm): "Build an Android chat app with Llama,
  KleidiAI, ExecuTorch, and XNNPACK"; "Run LLM inference on Android with
  KleidiAI, MediaPipe, and XNNPACK"; "Measure LLM inference performance with
  KleidiAI and SME2 on Android".
- llama.cpp quantize (`--tensor-type`, mixed precision): `tools/quantize/README.md`.
- llama.cpp on Android: `docs/android.md`.
- Arm Kleidi / KleidiAI: https://www.arm.com/markets/artificial-intelligence/software/kleidi
