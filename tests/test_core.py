"""Fast, hardware-free tests for the search engine."""
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anvil import (Objective, anvil_plan, demo_transformer, greedy_search,
                   random_search)
from anvil.model import OptState, PRECISIONS, THREAD_OPTIONS
from anvil.planner import Surrogate
from anvil.profiler import MockProfiler
import random as _random


def _setup(layers=6):
    spec = demo_transformer(n_layers=layers)
    return spec, Objective(accuracy_floor=0.95)


def test_budget_is_respected():
    spec, obj = _setup()
    prof = MockProfiler(spec)
    anvil_plan(spec, prof, obj, budget=40)
    assert prof.evaluations <= 40


def test_baseline_is_feasible_and_slow():
    spec, obj = _setup()
    prof = MockProfiler(spec)
    base = prof.evaluate(OptState.baseline(spec))
    assert obj.feasible(base)  # fp16 baseline must satisfy the floor


def test_result_is_feasible():
    spec, obj = _setup()
    r = anvil_plan(spec, MockProfiler(spec), obj, budget=60)
    assert obj.feasible(r.best_metrics)
    assert r.speedup() > 1.0  # it actually sped the model up


def test_anvil_beats_or_matches_baselines():
    """Core claim: on the same budget, Anvil's best feasible latency is no
    worse than random or greedy (and usually strictly better)."""
    spec, obj = _setup(layers=6)
    budget = 60
    a = anvil_plan(spec, MockProfiler(spec, seed=0), obj, budget)
    g = greedy_search(spec, MockProfiler(spec, seed=0), obj, budget)
    r = random_search(spec, MockProfiler(spec, seed=0), obj, budget, seed=0)
    assert a.best_metrics.latency_ms <= g.best_metrics.latency_ms + 1e-6
    assert a.best_metrics.latency_ms <= r.best_metrics.latency_ms + 1e-6


def test_cache_prevents_double_pay():
    spec, obj = _setup()
    prof = MockProfiler(spec)
    s = OptState.baseline(spec)
    prof.evaluate(s)
    prof.evaluate(s)  # same key
    assert prof.evaluations == 1


def test_anvil_beats_baselines_across_seeds():
    """Robustness: the headline win is not a single-seed fluke. On the same
    budget, Anvil's best feasible latency is <= both baselines for every seed."""
    spec, obj = _setup(layers=6)
    budget = 60
    for seed in range(12):
        a = anvil_plan(spec, MockProfiler(spec, seed=seed), obj, budget)
        g = greedy_search(spec, MockProfiler(spec, seed=seed), obj, budget)
        r = random_search(spec, MockProfiler(spec, seed=seed), obj, budget, seed=seed)
        assert a.best_metrics.latency_ms <= g.best_metrics.latency_ms + 1e-6, seed
        assert a.best_metrics.latency_ms <= r.best_metrics.latency_ms + 1e-6, seed


def test_planner_terminates_when_beam_saturates():
    """Regression: anvil_plan once looped forever when a round's top predicted
    candidates were all cache hits, so `evaluations` never reached the budget.
    Give it a budget it cannot fully spend and assert it still returns."""
    spec, obj = _setup(layers=6)
    done = {}

    def go():
        prof = MockProfiler(spec, seed=0)
        anvil_plan(spec, prof, obj, budget=100)  # more than it can spend
        done["evals"] = prof.evaluations

    t = threading.Thread(target=go, daemon=True)
    t.start()
    t.join(timeout=20)
    assert not t.is_alive(), "anvil_plan hung: did not terminate (budget unreached)"
    assert done["evals"] <= 100


def test_surrogate_calibrates_to_mock():
    """The surrogate's priors deliberately differ from the mock's true cost
    constants, so online calibration must move k_lat/k_acc away from 1.0.
    (If they matched, the surrogate would be an oracle and calibration a no-op.)"""
    spec, _ = _setup()
    prof = MockProfiler(spec, seed=0)
    surr = Surrogate(spec)
    rng = _random.Random(0)
    for _ in range(25):
        s = OptState(
            precision=[rng.choice(PRECISIONS) for _ in range(spec.n_blocks)],
            fused=[rng.random() < 0.5 for _ in range(spec.n_blocks)],
            threads=rng.choice(THREAD_OPTIONS),
        )
        surr.calibrate(s, prof.evaluate(s))
    assert abs(surr.k_lat - 1.0) > 0.03, surr.k_lat  # latency prior corrected
    assert abs(surr.k_acc - 1.0) > 0.03, surr.k_acc  # accuracy prior corrected


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
