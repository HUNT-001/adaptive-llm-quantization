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
    size_budget_mb: float = None     # optional hard cap on memory_mb (mobile:
                                     # the model must fit the device). None on
                                     # the cloud track, so behavior is unchanged.

    def feasible(self, m: Metrics) -> bool:
        if m.accuracy < self.accuracy_floor:
            return False
        if self.size_budget_mb is not None and m.memory_mb > self.size_budget_mb:
            return False
        return True

    def score(self, m: Metrics, baseline: Metrics) -> float:
        """Higher is better. Rewards fractional latency and memory reduction
        vs the fp16 baseline; anything infeasible (under the accuracy floor, or
        over the size budget when one is set) is dominated by a large penalty so
        the search is pulled back to feasible territory.

        Note the metric fields are reinterpreted on the Mobile track: latency_ms
        is time-to-first-token, memory_mb is on-disk model size, accuracy is a
        quality score, and throughput is tokens/sec. Minimizing latency is then
        equivalent to maximizing tokens/sec, so the objective needs no change of
        form — only the size budget, which is off (None) for cloud."""
        lat_gain = (baseline.latency_ms - m.latency_ms) / baseline.latency_ms
        mem_gain = (baseline.memory_mb - m.memory_mb) / baseline.memory_mb
        s = self.w_latency * lat_gain + self.w_memory * mem_gain
        if m.accuracy < self.accuracy_floor:
            s -= self.infeasible_penalty * (self.accuracy_floor - m.accuracy)
        if self.size_budget_mb is not None and m.memory_mb > self.size_budget_mb:
            over = (m.memory_mb - self.size_budget_mb) / self.size_budget_mb
            s -= self.infeasible_penalty * over
        return s
