"""Model description + optimization state.

A `ModelSpec` is a lightweight, hardware-agnostic description of a neural
network as a sequence of blocks (e.g. transformer layers). Anvil never needs
the real weights to *plan* — it plans over this graph description and only the
profiler backend touches real bytes. That separation is what lets us develop
and validate the whole search algorithm on a laptop and then swap in the real
Arm/Performix profiler with zero changes to the planner.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import List


# Precision options an action may assign to a block. Lower bits = faster +
# smaller, but potentially worse accuracy depending on block sensitivity.
PRECISIONS = ("fp16", "int8", "int4")

# How many threads the runtime may use. Real values come from the target's
# core count; these are placeholders for the mock backend.
THREAD_OPTIONS = (1, 2, 4, 8)


@dataclass
class Block:
    """One optimizable unit of the model (e.g. a transformer layer)."""
    name: str
    base_latency_ms: float       # latency at fp16, single thread, unfused
    base_memory_mb: float        # weight memory at fp16
    quant_sensitivity: float     # 0..1, how much accuracy this block loses
                                 # when heavily quantized (attention/embeddings
                                 # are typically sensitive; MLP less so)
    fusible: bool = True         # can this block be fused with the next one


@dataclass
class ModelSpec:
    """Immutable description of the model to be optimized."""
    name: str
    blocks: List[Block]
    base_accuracy: float = 1.0   # reference quality score (e.g. 1 - normalized
                                 # perplexity); actions can only reduce it

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)


@dataclass
class OptState:
    """A concrete optimization configuration = one point in the search space.

    This is the `S` in the write-up: the full set of decisions the planner has
    made so far. The profiler maps an OptState to measured metrics.
    """
    precision: List[str]         # per-block precision, len == n_blocks
    fused: List[bool]            # per-block "fuse with next" flag
    threads: int

    @staticmethod
    def baseline(spec: ModelSpec) -> "OptState":
        """The un-optimized starting point: everything fp16, nothing fused,
        max threads. Every method starts here for a fair comparison."""
        return OptState(
            precision=["fp16"] * spec.n_blocks,
            fused=[False] * spec.n_blocks,
            threads=max(THREAD_OPTIONS),
        )

    def copy(self) -> "OptState":
        return OptState(
            precision=list(self.precision),
            fused=list(self.fused),
            threads=self.threads,
        )

    def key(self) -> str:
        """Stable hash so we can cache profiler results and never pay to
        measure the same configuration twice."""
        blob = json.dumps(
            {"p": self.precision, "f": self.fused, "t": self.threads},
            sort_keys=True,
        )
        return hashlib.sha1(blob.encode()).hexdigest()[:12]


@dataclass
class Metrics:
    """What the profiler returns for a given OptState."""
    latency_ms: float
    memory_mb: float
    accuracy: float
    throughput: float            # inferences/sec (or tokens/sec)

    def as_dict(self) -> dict:
        return {
            "latency_ms": round(self.latency_ms, 4),
            "memory_mb": round(self.memory_mb, 3),
            "accuracy": round(self.accuracy, 5),
            "throughput": round(self.throughput, 4),
        }


def search_space_size(spec: ModelSpec) -> int:
    """Size of the configuration space. This is deliberately huge — that is the
    whole point. Brute force is impossible, so strategic search has something
    real to prove versus random/greedy."""
    per_block = len(PRECISIONS) * 2  # precision choices x fuse/no-fuse
    return (per_block ** spec.n_blocks) * len(THREAD_OPTIONS)
