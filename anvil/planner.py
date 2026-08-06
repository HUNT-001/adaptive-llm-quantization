"""The Anvil planner — strategic search over the optimization space.

This is the novel core and the reason the project is not "another benchmark
script". Instead of brute-forcing configurations, the planner:

  1. Keeps a *surrogate* cost model that predicts (latency, memory, accuracy)
     for any configuration for free. On the mock it mirrors the analytic model;
     on the Arm box it is seeded from one profiling sweep and then *calibrated
     online* from real Performix measurements (the "update world model" step).
  2. Runs a beam search over per-block decisions. At each round it generates
     the neighbors of every beam member, predicts their score with the
     surrogate, and spends its scarce profiler budget ONLY on the few most
     promising candidates (predict improvement -> choose next -> measure).
  3. Calibrates the surrogate on the residual between predicted and measured
     latency/accuracy, so its predictions get sharper as it goes.
  4. Emits a human-readable decision trace ("why this move") — this is the demo
     moment where the agent explains itself before the number moves.

Because it explores non-greedy sequences guided by a model, it escapes the
local optima that trap greedy hill-climbing, and it wastes far fewer real
evaluations than random sampling.
"""
from __future__ import annotations

from typing import List, Optional

from .actions import neighbors
from .model import Metrics, ModelSpec, OptState
from .objective import Objective
from .profiler import Profiler
from .search import Recorder, SearchResult

# Same cost-model *form* the surrogate assumes. On real hardware these come
# from an initial profiling sweep; the online calibration corrects the rest.
_SPEEDUP = {"fp16": 1.00, "int8": 0.62, "int4": 0.45}
_MEM = {"fp16": 1.00, "int8": 0.50, "int4": 0.25}
_ACC_PENALTY = {"fp16": 0.000, "int8": 0.010, "int4": 0.060}


class Surrogate:
    """Cheap predictor of metrics for a configuration. Calibrated online."""

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.k_lat = 1.0    # multiplicative latency correction (EMA-updated)
        self.k_acc = 1.0    # accuracy-loss correction

    def predict(self, s: OptState) -> Metrics:
        latency = 0.0
        memory = 0.0
        acc_loss = 0.0
        for i, blk in enumerate(self.spec.blocks):
            prec = s.precision[i]
            l = blk.base_latency_ms * _SPEEDUP[prec]
            if s.fused[i]:
                l *= 0.93
            latency += l
            memory += blk.base_memory_mb * _MEM[prec]
            acc_loss += _ACC_PENALTY[prec] * (0.3 + 1.4 * blk.quant_sensitivity)
        serial = 0.10
        latency *= serial + (1 - serial) / s.threads
        latency *= self.k_lat
        accuracy = max(0.0, self.spec.base_accuracy - self.k_acc * acc_loss)
        return Metrics(latency, memory, accuracy, 1000.0 / latency)

    def calibrate(self, s: OptState, measured: Metrics, alpha: float = 0.3) -> None:
        """Nudge the correction factors toward the observed residual (EMA)."""
        pred = self.predict(s)
        if pred.latency_ms > 0:
            ratio = measured.latency_ms / pred.latency_ms
            self.k_lat *= (1 - alpha) + alpha * ratio
        pred_loss = self.spec.base_accuracy - pred.accuracy
        meas_loss = self.spec.base_accuracy - measured.accuracy
        if pred_loss > 1e-6:
            ratio_a = max(0.0, meas_loss / pred_loss)
            self.k_acc *= (1 - alpha) + alpha * ratio_a


def anvil_plan(
    spec: ModelSpec,
    profiler: Profiler,
    objective: Objective,
    budget: int,
    beam_width: int = 4,
    candidates_per_round: int = 6,
) -> SearchResult:
    base = profiler.evaluate(OptState.baseline(spec))
    rec = Recorder(objective, base)
    surrogate = Surrogate(spec)
    surrogate.calibrate(OptState.baseline(spec), base)

    trace: List[str] = [
        f"baseline: {base.latency_ms:.1f} ms, acc {base.accuracy:.3f}, "
        f"mem {base.memory_mb:.0f} MB (floor acc>={objective.accuracy_floor})"
    ]

    beam: List[OptState] = [OptState.baseline(spec)]
    round_no = 0

    while profiler.evaluations < budget:
        round_no += 1
        # 1. Propose: all neighbors of every beam member (deduped).
        proposals = {}
        for state in beam:
            for act in neighbors(spec, state):
                cand = act.apply(state)
                proposals[cand.key()] = (cand, act.name)
        if not proposals:
            break

        # 2. Predict each proposal's score for free with the surrogate.
        scored = []
        for cand, act_name in proposals.values():
            pm = surrogate.predict(cand)
            ps = objective.score(pm, base)
            scored.append((ps, cand, act_name, pm))
        scored.sort(key=lambda x: x[0], reverse=True)

        # 3. Spend real budget only on the top few predicted candidates.
        evals_before_round = profiler.evaluations
        measured_this_round: List[tuple] = []
        for ps, cand, act_name, pm in scored[:candidates_per_round]:
            if profiler.evaluations >= budget:
                break
            m = profiler.evaluate(cand)
            note = (
                f"{act_name}: surrogate predicted {pm.latency_ms:.1f} ms "
                f"@ acc {pm.accuracy:.3f}; measured {m.latency_ms:.1f} ms "
                f"@ acc {m.accuracy:.3f}"
            )
            rec.record(profiler.evaluations, cand, m, note)
            surrogate.calibrate(cand, m)          # 4. update the world model
            measured_this_round.append((objective.score(m, base), cand, m, note))

        # Convergence guard. If the round spent no budget, every top candidate
        # was already measured (all cache hits): the beam is stuck on explored
        # territory and the next round would propose the identical set forever.
        # Without this, `while evaluations < budget` never advances -> infinite
        # loop (reproduced at layers=6, budget>=~90 and at budget=60, seed=3).
        # See tests/test_core.py::test_planner_terminates_when_beam_saturates.
        if profiler.evaluations == evals_before_round:
            trace.append(
                f"round {round_no}: converged — top candidates already "
                f"measured; stopping at {profiler.evaluations} evals"
            )
            break

        if not measured_this_round:
            break

        # 5. Re-form the beam from the best measured configs seen so far.
        all_pts = sorted(rec.history, key=lambda p: p.score, reverse=True)
        beam = [pt.state for pt in all_pts[:beam_width]]

        best = measured_this_round[0]
        trace.append(
            f"round {round_no}: best move -> {best[3]} "
            f"[score {best[0]:+.3f}, k_lat={surrogate.k_lat:.3f}]"
        )

    best = rec.best
    if best is None:
        return SearchResult("anvil", base, OptState.baseline(spec), base,
                            0.0, rec.history, rec.convergence, trace)
    trace.append(
        f"result: {base.latency_ms / best.metrics.latency_ms:.2f}x faster, "
        f"acc {best.metrics.accuracy:.3f}, {profiler.evaluations} evals"
    )
    return SearchResult(
        method="anvil",
        baseline=base,
        best_state=best.state,
        best_metrics=best.metrics,
        best_score=best.score,
        history=rec.history,
        convergence=rec.convergence,
        trace=trace,
    )
