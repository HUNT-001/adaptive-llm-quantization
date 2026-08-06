"""Baseline search strategies — the "everybody brute-forces" methods we beat.

  * random_search : sample configurations uniformly, keep the best feasible one.
  * greedy_search : hill climbing. From the current config, measure every
                    single-step neighbor and move to the best one; stop at a
                    local optimum. Greedy gets trapped precisely because some
                    globally-good moves look locally bad (quantizing a sensitive
                    block only pays off once you compensate elsewhere).

Both share the planner's profiler, budget accounting, and recorder, so the
comparison is fair.
"""
from __future__ import annotations

import random
from typing import List

from .actions import neighbors
from .model import PRECISIONS, THREAD_OPTIONS, ModelSpec, OptState
from .objective import Objective
from .profiler import Profiler
from .search import Recorder, SearchResult


def _random_state(spec: ModelSpec, rng: random.Random) -> OptState:
    return OptState(
        precision=[rng.choice(PRECISIONS) for _ in range(spec.n_blocks)],
        fused=[rng.random() < 0.5 for _ in range(spec.n_blocks)],
        threads=rng.choice(THREAD_OPTIONS),
    )


def random_search(
    spec: ModelSpec,
    profiler: Profiler,
    objective: Objective,
    budget: int,
    seed: int = 0,
) -> SearchResult:
    rng = random.Random(seed)
    base = profiler.evaluate(OptState.baseline(spec))
    rec = Recorder(objective, base)
    while profiler.evaluations < budget:
        s = _random_state(spec, rng)
        m = profiler.evaluate(s)
        rec.record(profiler.evaluations, s, m)
    return _finish("random", spec, base, rec)


def greedy_search(
    spec: ModelSpec,
    profiler: Profiler,
    objective: Objective,
    budget: int,
) -> SearchResult:
    base = profiler.evaluate(OptState.baseline(spec))
    rec = Recorder(objective, base)
    current = OptState.baseline(spec)
    cur_score = objective.score(base, base)
    while profiler.evaluations < budget:
        best_neighbor = None
        best_neighbor_score = cur_score
        improved = False
        for act in neighbors(spec, current):
            if profiler.evaluations >= budget:
                break
            cand = act.apply(current)
            m = profiler.evaluate(cand)
            sc = rec.record(profiler.evaluations, cand, m).score
            if sc > best_neighbor_score:
                best_neighbor_score = sc
                best_neighbor = cand
                improved = True
        if not improved:
            break  # local optimum reached; greedy is stuck
        current = best_neighbor
        cur_score = best_neighbor_score
    return _finish("greedy", spec, base, rec)


def _finish(name: str, spec: ModelSpec, base, rec: Recorder) -> SearchResult:
    best = rec.best
    if best is None:  # nothing feasible found; report the fp16 baseline itself
        return SearchResult(name, base, OptState.baseline(spec), base,
                            0.0, rec.history, rec.convergence)
    return SearchResult(
        method=name,
        baseline=base,
        best_state=best.state,
        best_metrics=best.metrics,
        best_score=best.score,
        history=rec.history,
        convergence=rec.convergence,
    )
