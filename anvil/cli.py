"""Anvil command-line interface — the reusable artifact judges can run.

    anvil optimize --budget 60        # run the planner, print the plan + result
    anvil compare  --budget 60        # planner vs greedy vs random + charts
    anvil space    --layers 6         # show how large the search space is

By default everything runs against the mock profiler so it works with zero
setup. Point it at real hardware by wiring PerformixProfiler (see
docs/BUILD_PLAN.md) behind the --backend performix flag.
"""
from __future__ import annotations

import argparse
import sys

from . import report
from .models import demo_transformer
from .objective import Objective
from .planner import anvil_plan
from .baselines import greedy_search, random_search
from .profiler import MockProfiler
from .model import search_space_size


def _spec(args):
    return demo_transformer(n_layers=args.layers)


def cmd_optimize(args) -> int:
    spec = _spec(args)
    obj = Objective(accuracy_floor=args.acc_floor)
    r = anvil_plan(spec, MockProfiler(spec, seed=args.seed), obj, args.budget)
    print(r.summary(), "\n")
    print("plan:")
    for line in r.trace:
        print("  " + line)
    return 0


def cmd_compare(args) -> int:
    spec = _spec(args)
    obj = Objective(accuracy_floor=args.acc_floor)
    results = {
        "random": random_search(spec, MockProfiler(spec, seed=args.seed), obj, args.budget, seed=args.seed),
        "greedy": greedy_search(spec, MockProfiler(spec, seed=args.seed), obj, args.budget),
        "anvil": anvil_plan(spec, MockProfiler(spec, seed=args.seed), obj, args.budget),
    }
    for name in ("random", "greedy", "anvil"):
        print("  " + results[name].summary())
    if args.out:
        report.write_json(results, f"{args.out}/results.json")
        report.plot(results, args.out, accuracy_floor=args.acc_floor)
        print(f"\nartifacts written to {args.out}/")
    return 0


def cmd_space(args) -> int:
    spec = _spec(args)
    print(f"{spec.name}: {spec.n_blocks} blocks -> "
          f"~{search_space_size(spec):.2e} configurations")
    return 0


def main(argv=None) -> int:
    # Shared flags live on a parent parser attached to each subcommand, so the
    # flags come AFTER the command, matching the README quickstart:
    #   anvil optimize --budget 60
    # (Previously the flags sat on the top-level parser only, so that exact
    # documented form failed with "unrecognized arguments". Attaching them to
    # the top parser too would instead let the subcommand's default silently
    # overwrite a pre-command value, so we keep them on the subcommands alone.)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--layers", type=int, default=6)
    common.add_argument("--budget", type=int, default=60)
    common.add_argument("--acc-floor", type=float, default=0.95)
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--out", default=None, help="dir for charts + results.json")

    ap = argparse.ArgumentParser(prog="anvil", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("optimize", "compare", "space"):
        sub.add_parser(name, parents=[common])
    args = ap.parse_args(argv)
    return {"optimize": cmd_optimize, "compare": cmd_compare,
            "space": cmd_space}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
