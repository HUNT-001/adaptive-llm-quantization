"""The action space: the moves the planner can make from any state.

Actions are *per-block* (and one global thread action). That is a deliberate
design choice: it makes the space combinatorial (see model.search_space_size),
which is exactly the regime where a strategic planner beats random/greedy. If
the action set were a flat list of ~8 global configs, brute force would win and
the planner would prove nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

from .model import PRECISIONS, THREAD_OPTIONS, ModelSpec, OptState


@dataclass
class Action:
    """A named transformation S_t -> S_{t+1}."""
    name: str
    apply: Callable[[OptState], OptState]


def _set_precision(block: int, prec: str) -> Action:
    def fn(s: OptState) -> OptState:
        ns = s.copy()
        ns.precision[block] = prec
        return ns
    return Action(f"quant[{block}]={prec}", fn)


def _toggle_fuse(block: int) -> Action:
    def fn(s: OptState) -> OptState:
        ns = s.copy()
        ns.fused[block] = not ns.fused[block]
        return ns
    return Action(f"fuse[{block}]", fn)


def _set_threads(t: int) -> Action:
    def fn(s: OptState) -> OptState:
        ns = s.copy()
        ns.threads = t
        return ns
    return Action(f"threads={t}", fn)


def neighbors(spec: ModelSpec, state: OptState) -> List[Action]:
    """All single-step moves reachable from `state`.

    We only emit moves that actually change something (e.g. we don't offer to
    set a block to the precision it already has), so the branching factor stays
    honest and the planner isn't handed free no-ops.
    """
    acts: List[Action] = []
    for b in range(spec.n_blocks):
        for prec in PRECISIONS:
            if state.precision[b] != prec:
                acts.append(_set_precision(b, prec))
        if spec.blocks[b].fusible and b < spec.n_blocks - 1:
            acts.append(_toggle_fuse(b))
    for t in THREAD_OPTIONS:
        if state.threads != t:
            acts.append(_set_threads(t))
    return acts
