# Action Space Design

The single most important design decision in Anvil is that **actions are
per-block**, which makes the search space combinatorial. This is what gives a
strategic planner something to prove against brute force.

## State

An `OptState` is one full configuration:

- `precision[i] ∈ {fp16, int8, int4}` for each of the N blocks
- `fused[i] ∈ {true, false}` for each block (fuse with the next block)
- `threads ∈ {1, 2, 4, 8}` (one global choice)

## Actions (moves from a state)

- set block `i` to a different precision
- toggle fusion on block `i`
- change the global thread count

Only moves that change something are emitted (no no-ops), keeping the branching
factor honest.

## Why combinatorial matters

Space size = `(|precisions| · 2)^N · |threads| = 6^N · 4`.

| N blocks | configurations |
|----------|----------------|
| 6  | ~1.9 × 10⁵ |
| 14 | ~3.1 × 10¹¹ |
| 24 | ~1.7 × 10¹⁹ |

If the action space were instead ~8 global presets, exhaustive search would win
and a planner would add nothing. By making decisions per block, the space
explodes and **the interesting question becomes *which* few configurations are
worth spending a real benchmark on** — which is exactly what the surrogate-guided
planner answers.

## Why greedy gets trapped (and the planner doesn't)

Some blocks are quantization-sensitive (attention, embeddings); others are robust
(MLP). Greedy hill-climbing won't quantize a sensitive block because it looks
locally bad — even when doing so, *combined* with compensating moves elsewhere,
is globally optimal. The planner's beam + surrogate explores those non-greedy
sequences, which is why it escapes the local optimum greedy settles into. You can
see this directly in `convergence.png`: greedy plateaus, Anvil keeps improving.

## Extending the space

Add actions by extending `anvil/actions.py::neighbors` and teaching the surrogate
(and `PerformixProfiler._apply_config_remote`) how to realize them. Natural
additions for future work: structured pruning, per-block sparsity, KV-cache
precision, operator reordering. Keep them per-block to preserve the property that
makes the planner valuable.
