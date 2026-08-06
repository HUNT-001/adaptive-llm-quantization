"""Objective / reward.

We frame optimization as constrained search: minimize latency subject to an
accuracy floor, while also valuing memory savings. This maps to the write-up's

    R = a*dThroughput + b*dLatency + g*dMemory + d*dAccuracy

but expressed as a single scalar `score` (higher = better) so any search method
can rank configurations, plus a hard `feasible` flag for the accuracy floor.
The Pareto story in the demo comes from plotting latency vs accuracy for every
configuration each method actually measured.
"""
from __future__ import annotations

from dataclasses import dataclass

from .model import Metrics


@dataclass
class Objective:
    accuracy_floor: float = 0.95     # do not ship below this quality
    w_latency: float = 1.0           # weight on latency reduction
    w_memory: float = 0.25           # weight on memory reduction
    infeasible_penalty: float = 1e3

    def feasible(self, m: Metrics) -> bool:
        return m.accuracy >= self.accuracy_floor

    def score(self, m: Metrics, baseline: Metrics) -> float:
        """Higher is better. Rewards fractional latency and memory reduction
        vs the fp16 baseline; anything under the accuracy floor is dominated by
        a large penalty so the search is pulled back to feasible territory."""
        lat_gain = (baseline.latency_ms - m.latency_ms) / baseline.latency_ms
        mem_gain = (baseline.memory_mb - m.memory_mb) / baseline.memory_mb
        s = self.w_latency * lat_gain + self.w_memory * mem_gain
        if not self.feasible(m):
            deficit = self.accuracy_floor - m.accuracy
            s -= self.infeasible_penalty * deficit
        return s
