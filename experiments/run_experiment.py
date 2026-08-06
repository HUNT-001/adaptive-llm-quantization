"""The headline experiment: Anvil vs greedy vs random, same evaluation budget.

Run:
    python experiments/run_experiment.py --budget 60 --layers 6

Each method gets a FRESH profiler (so budgets are independent and fair) but the
same model, same objective, and same accuracy floor. Outputs a summary table,
the convergence + Pareto charts, and results.json (including Anvil's decision
trace). This is the artifact the demo video is built around.

Swap MockProfiler -> PerformixProfiler (same interface) to run the identical
experiment on the Arm target with real measurements.
"""
from __future__ import annotations

import argparse
import os
import sys

# allow running from repo root without installing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anvil import (Objective, anvil_plan, demo_transformer, greedy_search,
                   random_search, search_space_size)
from anvil.profiler import MockProfiler
from anvil import report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=60,
                    help="profiler evaluations allowed per method")
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--acc-floor", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    spec = demo_transformer(n_layers=args.layers)
    obj = Objective(accuracy_floor=args.acc_floor)

    print(f"model: {spec.name}  ({spec.n_blocks} blocks)")
    print(f"search space: ~{search_space_size(spec):.2e} configurations")
    print(f"budget: {args.budget} evaluations/method   acc floor: {args.acc_floor}\n")

    results = {}
    # fresh profiler per method => independent, fair budgets
    results["random"] = random_search(
        spec, MockProfiler(spec, seed=args.seed), obj, args.budget, seed=args.seed)
    results["greedy"] = greedy_search(
        spec, MockProfiler(spec, seed=args.seed), obj, args.budget)
    results["anvil"] = anvil_plan(
        spec, MockProfiler(spec, seed=args.seed), obj, args.budget)

    print("results")
    print("-" * 64)
    for name in ("random", "greedy", "anvil"):
        print("  " + results[name].summary())
    print("-" * 64)

    anvil = results["anvil"]
    best_baseline = min(r.best_metrics.latency_ms for r in results.values())
    if anvil.best_metrics.latency_ms <= best_baseline + 1e-9:
        print("PASS: Anvil found the best (or tied-best) feasible config.\n")
    else:
        print("NOTE: a baseline matched Anvil this run; try more layers/lower budget.\n")

    print("Anvil decision trace:")
    for line in anvil.trace:
        print("  " + line)

    report.write_json(results, os.path.join(args.out, "results.json"))
    imgs = report.plot(results, args.out, accuracy_floor=args.acc_floor)
    print("\nwrote:", os.path.join(args.out, "results.json"))
    for p in imgs:
        print("wrote:", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
