# Anvil — Build Plan

**Target:** Arm Create: AI Optimization Challenge 2026, **Cloud AI** track.
**Deadline:** 14 Aug 2026, 16:00 PDT. **Budget:** $0 (Oracle Ampere A1 free tier).
**Judging:** Tech 40 · WOW 25 · Impact 20 · DevX 15.

## 1. The pitch (what we're building and why it wins)

Anvil is an agent that treats model optimization as **sequential decision-making**
instead of brute-force search. It profiles a model on Arm with Arm Performix,
reasons about which optimization to apply next, applies it, re-benchmarks, and
converges toward a better latency/accuracy tradeoff in far fewer trials than the
quantize-and-pray loop everyone else submits — and it explains each decision.

Why this scores on every axis:

- **Tech (40):** real optimizations, real Performix measurements on Arm64, and a
  non-trivial surrogate-guided beam search with online calibration.
- **WOW (25):** it drives Arm's brand-new flagship (Performix) in exactly the
  agentic loop Arm built it for, and the agent *narrates its reasoning* before
  each measured gain. Demo'd to Arm's own developer evangelists.
- **Impact (20):** ships as a reusable CLI + Python package any Arm dev points at
  their own model. Reusable artifact = what this axis rewards.
- **DevX (15):** one-command reproducible run, clean README, works with zero
  hardware via the mock backend.

The novelty claim, stated defensibly: *strategic search over the optimization
space provably reaches a better feasible configuration in fewer benchmark
evaluations than random or greedy search.* We do **not** claim a trained RL
world-model — that's future work (Section 8). The claim we make is the claim we
prove, with a chart.

## 2. Architecture

```
ModelSpec ──▶ Anvil planner ──▶ OptState (config)
                  │  ▲
        predict   │  │ calibrate (online)
                  ▼  │
             Surrogate cost model
                  │
                  ▼  measure (spends budget)
        Profiler backend
          ├─ MockProfiler      (analytic; laptop dev + tests)
          └─ PerformixProfiler (apx CLI/MCP over SSH → Oracle Ampere A1)
                  │
                  ▼
        convergence.png · pareto.png · results.json · dashboard
```

The **only** class that differs between laptop and Arm hardware is the profiler
backend. That boundary is the whole engineering story: we validate the algorithm
offline, then swap in real measurements with no planner changes.

## 3. Day 1 — de-risk the one thing that can kill the project

Do this *before writing any optimization code.* If Performix won't run on the
free tier, we need to know immediately.

1. **Spin up the target.** Oracle Cloud Always Free → Ampere A1 Compute (up to
   4 OCPU / 24 GB RAM free), **Ubuntu 22.04 or 24.04** (Performix supports Amazon
   Linux 2023 / Ubuntu 22.04 / 24.04 targets). Ampere A1 = Ampere Altra =
   Neoverse N1, which Performix supports.
2. **Prep the target for Performix:** SSH key-based auth, and **passwordless
   sudo** for your user (Performix runs privileged perf counters and cannot type
   a sudo password). Confirm you can `ssh` in and `sudo -n true` returns cleanly.
3. **Install Performix on your laptop (the "host").** Use the DEB package on
   Linux hosts (avoids the FUSE/libfuse2 AppImage issue on Ubuntu 22.04+). The
   install includes the GUI and the **`apx` CLI** — we script against `apx`.
   Install guide: https://learn.arm.com/install-guides/performix/
4. **Connect host → target** and capture ONE real report from any workload
   (even `python -c "..."`). Save the raw `apx` output. This tells us the exact
   field names to parse in `PerformixProfiler._parse`.
5. **Note the MCP option.** Performix ships an MCP server; if you'd rather have
   the agent call Performix as an MCP tool (nice for the WOW narrative), capture
   one MCP call/response too. Either path fits the same `Profiler` interface.

Exit criterion for Day 1: one real latency/memory number, measured on the Oracle
box, in your terminal. Everything else builds on that.

## 4. The action space (why the planner has something to prove)

Actions are **per-block**, not global. For an N-block model each block chooses a
precision (fp16/int8/int4) and a fuse flag, plus a global thread count, giving a
space of size `(3·2)^N · |threads|` — ~10¹¹ for a 14-block model. Brute force is
impossible, so strategic search matters. If you instead offered ~8 global presets,
exhaustive search would win and the planner would be pointless. Keep the space
combinatorial. Details and rationale in `docs/ACTION_SPACE.md`.

Three real optimizations for the hardware version (pick these, wire end-to-end,
don't add more):

1. **Per-block quantization** (INT8 / INT4). On Arm, use a runtime that supports
   mixed precision per layer — e.g. llama.cpp GGUF with per-tensor quant, or
   ONNX Runtime with the Arm (KleidiAI/ACL) backend. `OptState.precision[i]` maps
   to which blocks get which quant.
2. **Layer/operator fusion.** Toggle available fusions (e.g. ORT graph
   optimization levels, or fused LayerNorm/attention kernels).
3. **Thread / kernel tuning.** Thread count and, where exposed, kernel choice.

`PerformixProfiler._apply_config_remote` is the one function that translates an
`OptState` into your runtime's config; implement it for whichever runtime you
choose on Day 2.

## 5. The headline experiment (this IS the submission)

`experiments/run_experiment.py` runs **Anvil vs greedy vs random on the same
evaluation budget**, same model, same accuracy floor, fresh profiler each, and
produces `convergence.png` (best feasible latency vs evaluations) + `pareto.png`
(latency vs accuracy scatter) + `results.json` (incl. Anvil's decision trace).

The winning visual is the convergence chart: Anvil's curve drops faster and
lower than greedy (which plateaus at a local optimum) and random (which barely
moves). Reproduce it on the mock today; reproduce it on Arm in Week 2–3 with real
Performix numbers — same script, same chart, real data.

Accuracy is non-optional: a planner that only cuts latency while wrecking quality
is trivial. The accuracy floor + the Pareto plot make "we retained quality" a
visible, defensible claim. On hardware, wire `accuracy_fn` (perplexity on a small
eval set, or a task-accuracy check) into `PerformixProfiler`.

## 6. Three-week timeline

**Week 1 — de-risk + baseline.**
Day 1: Section 3 (Performix on Oracle, one real number). Days 2–3: pick the
runtime, implement `_apply_config_remote` + `_run_and_profile` + `_parse` for one
real model (start small: TinyLlama / a distilBERT / a small GGUF). Days 4–5:
accuracy harness; get baseline fp16 + a couple of hand configs measured on Arm.
The offline engine (this repo) is already done, so Week 1 is purely the hardware
bridge.

**Week 2 — the planner on real data.**
Run `anvil_plan` with `PerformixProfiler`. Land the "Anvil beats greedy/random on
the same budget" result with **real** Arm measurements. Tune beam width /
candidates-per-round if evaluations are slow (each on-device benchmark costs real
time, so keep the budget modest — 30–60 evals). This is the deliverable; protect
time for it.

**Week 3 — package + film.**
Polish the CLI + README, generate final charts, fill the dashboard with real
`results.json`, record the <3-min video (see Section 7), verify one-command
reproduction on a fresh Oracle instance, submit. Keep 2 days of buffer.

## 7. The 3-minute video (where WOW is won)

Structure: (0:00–0:20) the problem — everyone brute-forces optimization.
(0:20–1:10) show `anvil optimize` running and **read the decision trace aloud** —
the agent naming each move and its predicted vs measured effect. (1:10–2:10) the
convergence chart: Anvil vs greedy vs random, real Arm numbers, same budget.
(2:10–2:45) `anvil compare` reproducing it live + the dashboard. (2:45–3:00) one
line of future work (learned surrogate / RL). No third-party trademarks or
copyrighted music (rules requirement). Show it running on the Arm box.

## 8. Scope discipline & future work

**In scope (three deliverables only):** the agent/CLI; the beats-baseline Pareto
result with real Performix numbers; a reproducible repo + video.

**Explicitly cut to "future work" slides:** trained RL policy / learned world
model, LangGraph multi-agent split, SDK + GitHub Action, pruning/sparsity actions.
They're great narrative closers; building them at 40% each loses the demo. The
math framing (S=(L,M,A,T,G), R=αΔT+βΔL+γΔM+δΔA, π*=argmax E[ΣR]) belongs on one
slide as the principled formulation Anvil approximates.

## 9. Risk register

| Risk | Mitigation |
|------|------------|
| Performix won't run on Oracle free tier | Day-1 gate. Fallback profiler: Linux `perf` + wall-clock timers behind the same `Profiler` interface, still on Arm. |
| On-device benchmarks are slow | Small model + modest budget (30–60 evals); the surrogate means we measure few candidates. |
| Planner ties baselines on tiny spaces | Keep the space combinatorial (per-block actions, ≥6 layers). Verified: planner wins on the 14-block demo. |
| Quant destroys accuracy | Accuracy floor in the objective + Pareto plot; report retained-accuracy explicitly. |
| Repro fails for judges | `--out` writes everything; README one-command run; mock backend needs no hardware. |

## 10. References

- Challenge: https://arm-ai-optimization-challenge.devpost.com/ · rules: https://arm-ai-optimization-challenge.devpost.com/rules
- Arm Performix: https://developer.arm.com/servers-and-cloud-computing/arm-performix
- Performix install: https://learn.arm.com/install-guides/performix/
