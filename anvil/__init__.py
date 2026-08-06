"""Anvil — a planning-based optimization agent for AI models on Arm.

Anvil treats model optimization as a sequential decision problem: it profiles a
model on Arm (via Arm Performix), reasons about which optimization to apply
next, applies it, re-benchmarks, and converges toward a better latency/accuracy
tradeoff in far fewer trials than brute-force tuning.
"""
from .model import Block, Metrics, ModelSpec, OptState, search_space_size
from .objective import Objective
from .profiler import MockProfiler, PerformixProfiler, Profiler
from .planner import anvil_plan
from .baselines import greedy_search, random_search
from .models import demo_transformer

__version__ = "0.1.0"

__all__ = [
    "Block", "Metrics", "ModelSpec", "OptState", "search_space_size",
    "Objective", "Profiler", "MockProfiler", "PerformixProfiler",
    "anvil_plan", "greedy_search", "random_search", "demo_transformer",
]
