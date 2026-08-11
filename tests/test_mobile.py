"""Hardware-free tests for the Mobile backend (MobileMockProfiler).

Mirrors tests/test_core.py but on the mobile spec + mock, where metric fields
carry their Mobile meanings (latency_ms = TTFT, memory_mb = GGUF size,
accuracy = quality, throughput = tokens/sec). No phone required.
"""
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anvil import (Objective, anvil_plan, greedy_search, random_search,
                   mobile_llm_spec, MobileMockProfiler)
from anvil.model import OptState, PRECISIONS, THREAD_OPTIONS
from anvil.planner import Surrogate
import random as _random


def _setup(layers=16, size_budget_mb=None):
    spec = mobile_llm_spec(n_layers=layers)
    return spec, Objective(accuracy_floor=0.95, size_budget_mb=size_budget_mb)


def test_baseline_is_feasible():
    spec, obj = _setup()
    prof = MobileMockProfiler(spec, seed=0)
    base = prof.evaluate(OptState.baseline(spec))
    assert obj.feasible(base)          # F16 baseline is full quality
    assert base.throughput > 0


def test_budget_is_respected():
    spec, obj = _setup()
    prof = MobileMockProfiler(spec, seed=0)
    anvil_plan(spec, prof, obj, budget=30)
    assert prof.evaluations <= 30


def test_result_speeds_up_and_shrinks():
    spec, obj = _setup()
    base_prof = MobileMockProfiler(spec, seed=0)
    base = base_prof.evaluate(OptState.baseline(spec))
    r = anvil_plan(spec, MobileMockProfiler(spec, seed=0), obj, budget=40)
    assert obj.feasible(r.best_metrics)
    assert r.best_metrics.throughput > base.throughput   # faster tokens/sec
    assert r.best_metrics.memory_mb < base.memory_mb      # smaller on disk


def test_anvil_beats_baselines_across_seeds():
    """Core claim on the mobile mock: same on-device budget, Anvil's best
    feasible config is at least as fast as random and greedy, every seed."""
    spec, obj = _setup(layers=16)
    budget = 40
    for seed in range(10):
        a = anvil_plan(spec, MobileMockProfiler(spec, seed=seed), obj, budget)
        g = greedy_search(spec, MobileMockProfiler(spec, seed=seed), obj, budget)
        r = random_search(spec, MobileMockProfiler(spec, seed=seed), obj, budget, seed=seed)
        # higher tokens/sec is better -> Anvil's best latency (per-token) <= both
        assert a.best_metrics.latency_ms <= g.best_metrics.latency_ms + 1e-6, seed
        assert a.best_metrics.latency_ms <= r.best_metrics.latency_ms + 1e-6, seed


def test_size_budget_is_enforced():
    """With a size budget below the F16 baseline, the planner must ship a config
    that both fits the budget and holds quality. (Smaller model here so the
    feasible region is reachable within budget; on a big model + tiny eval
    budget the planner may not cross into feasibility, and then it correctly
    reports no feasible config rather than an over-budget one.)"""
    spec = mobile_llm_spec(n_layers=6)
    base = MobileMockProfiler(spec, seed=0).evaluate(OptState.baseline(spec))
    budget_mb = base.memory_mb * 0.75           # force real quantization
    obj = Objective(accuracy_floor=0.95, size_budget_mb=budget_mb)
    r = anvil_plan(spec, MobileMockProfiler(spec, seed=0), obj, budget=60)
    assert obj.feasible(r.best_metrics)          # found a genuinely feasible config
    assert r.best_metrics.memory_mb <= budget_mb + 1e-6
    assert r.best_metrics.accuracy >= 0.95


def test_planner_terminates_when_beam_saturates():
    """Hang regression, mobile mock: a budget it cannot fully spend still returns."""
    spec, obj = _setup()
    done = {}

    def go():
        prof = MobileMockProfiler(spec, seed=0)
        anvil_plan(spec, prof, obj, budget=100)
        done["evals"] = prof.evaluations

    t = threading.Thread(target=go, daemon=True)
    t.start()
    t.join(timeout=20)
    assert not t.is_alive(), "anvil_plan hung on the mobile mock"
    assert done["evals"] <= 100


def test_surrogate_calibrates_to_mobile_mock():
    """The phone mock's true constants differ from the surrogate priors (decode
    is bandwidth-bound: threads help little, quant helps a lot), so calibration
    must move k_lat/k_acc away from 1.0."""
    spec, _ = _setup()
    prof = MobileMockProfiler(spec, seed=0)
    surr = Surrogate(spec)
    rng = _random.Random(0)
    for _ in range(25):
        s = OptState(
            precision=[rng.choice(PRECISIONS) for _ in range(spec.n_blocks)],
            fused=[False] * spec.n_blocks,
            threads=rng.choice(THREAD_OPTIONS),
        )
        surr.calibrate(s, prof.evaluate(s))
    assert abs(surr.k_lat - 1.0) > 0.03, surr.k_lat
    assert abs(surr.k_acc - 1.0) > 0.03, surr.k_acc


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    raise SystemExit(1 if failed else 0)
