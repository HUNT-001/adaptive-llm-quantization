"""Shared search bookkeeping used by every method (planner + baselines).

Keeping this common guarantees an apples-to-apples comparison: all methods
share the same profiler, the same evaluation-budget accounting, and record the
same telemetry, so the convergence chart is fair by construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .model import Metrics, OptState
from .objective import Objective
from .profiler import Profiler


@dataclass
class MeasuredPoint:
    evals: int          # profiler evaluations spent when this point was taken
    state: OptState
    metrics: Metrics
    score: float
    note: str = ""      # planner reasoning ("why this move")


@dataclass
class SearchResult:
    method: str
    baseline: Metrics
    best_state: OptState
    best_metrics: Metrics
    best_score: float
    history: List[MeasuredPoint]              # every measured config, in order
    convergence: List[Tuple[int, float]]      # (evals, best feasible latency)
    trace: List[str] = field(default_factory=list)  # human-readable decisions

    def speedup(self) -> float:
        return self.baseline.latency_ms / self.best_metrics.latency_ms

    def summary(self) -> str:
        m = self.best_metrics
        return (
            f"{self.method}: {self.speedup():.2f}x faster "
            f"({self.baseline.latency_ms:.1f} -> {m.latency_ms:.1f} ms), "
            f"acc {m.accuracy:.3f}, mem {m.memory_mb:.0f} MB, "
            f"in {self.history[-1].evals if self.history else 0} evals"
        )


class Recorder:
    """Tracks the running best *feasible* configuration and the convergence
    curve as evaluations are spent."""

    def __init__(self, objective: Objective, baseline: Metrics):
        self.obj = objective
        self.baseline = baseline
        self.history: List[MeasuredPoint] = []
        self.convergence: List[Tuple[int, float]] = []
        self.best: Optional[MeasuredPoint] = None

    def record(self, evals: int, state: OptState, m: Metrics, note: str = "") -> MeasuredPoint:
        score = self.obj.score(m, self.baseline)
        pt = MeasuredPoint(evals, state.copy(), m, score, note)
        self.history.append(pt)
        feasible = self.obj.feasible(m)
        if feasible and (self.best is None or m.latency_ms < self.best.metrics.latency_ms):
            self.best = pt
        best_lat = self.best.metrics.latency_ms if self.best else self.baseline.latency_ms
        self.convergence.append((evals, best_lat))
        return pt
