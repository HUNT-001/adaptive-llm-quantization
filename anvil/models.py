"""Sample model specs for development against the mock profiler.

On real hardware you build a ModelSpec from one profiling sweep of your model
(per-layer latency/memory from Performix, sensitivity from a quick quant probe).
Here we synthesize a transformer-shaped spec so the whole pipeline runs on a
laptop. `demo_transformer` deliberately mixes sensitive blocks (attention,
embeddings) with robust ones (MLP) so greedy has local optima to fall into.
"""
from __future__ import annotations

import random

from .model import Block, ModelSpec


def demo_transformer(n_layers: int = 6, seed: int = 7) -> ModelSpec:
    rng = random.Random(seed)
    blocks = []
    # embedding: cheap latency, big memory, very quant-sensitive
    blocks.append(Block("embed", base_latency_ms=1.2, base_memory_mb=180.0,
                        quant_sensitivity=0.9, fusible=False))
    for i in range(n_layers):
        # each transformer layer = attention (sensitive) + MLP (robust)
        blocks.append(Block(
            f"attn_{i}",
            base_latency_ms=rng.uniform(2.5, 4.0),
            base_memory_mb=rng.uniform(40, 70),
            quant_sensitivity=rng.uniform(0.55, 0.85),
        ))
        blocks.append(Block(
            f"mlp_{i}",
            base_latency_ms=rng.uniform(3.5, 5.5),
            base_memory_mb=rng.uniform(90, 140),
            quant_sensitivity=rng.uniform(0.05, 0.25),
        ))
    # lm head: moderate
    blocks.append(Block("lm_head", base_latency_ms=1.8, base_memory_mb=180.0,
                        quant_sensitivity=0.5, fusible=False))
    return ModelSpec(name=f"demo_transformer_{n_layers}L", blocks=blocks,
                     base_accuracy=1.0)
