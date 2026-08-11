#!/usr/bin/env python3
"""Anvil on-device search — Qwen2.5-1.5B on the iQOO Z7s (Snapdragon 695).

Self-contained (stdlib only). Runs the SAME surrogate-guided beam search as the
offline engine, specialised to the Arm-real action space:

  * Per-layer precision in {Q8_0, Q4_0} -- the two KleidiAI-accelerated GGUF
    types (measured: Q4_K/Q6_K are NOT accelerated on this phone). token_embd is
    pinned to Q8_0. 28 layers -> 2^28 configurations.
  * Objective: MINIMISE on-disk size subject to a quality floor (perplexity <=
    target). Every config is KleidiAI-fast, so speed is a free win, not a search
    variable -- which lets us skip the slow benchmark during search and pay only
    quantize + perplexity (~80 s/config).
  * Anvil vs greedy vs random on the same evaluation budget; results checkpoint
    after every measurement; the three winners are benchmarked at the end.

Run on the phone (Termux), from ~ with qwen15b-f16.gguf present:

    termux-wake-lock
    python run_z7s.py --budget 14 --chunks 4

Use --mock to dry-run the whole search off-device (no llama.cpp needed).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
HOME = os.path.expanduser("~")
BIN = os.path.join(HOME, "llama.cpp", "build", "bin")
BASE_GGUF = os.path.join(HOME, "qwen15b-f16.gguf")
WIKI = os.path.join(HOME, "llama.cpp", "wikitext-2-raw", "wiki.test.raw")
WORK = HOME

N_LAYERS = 28
TYPES = {"q8": "Q8_0", "q4": "Q4_0"}          # the KleidiAI-accelerated types
# Per-layer F16 weight (MiB) and a rough size multiplier per type (from measured
# all-Q8_0 = 1565 MiB, all-Q4_0 = 886 MiB). Used only to RANK candidates; real
# size is read from the file.
LAYER_F16_MB = 89.25
SIZE_MULT = {"q8": 0.53, "q4": 0.30}
# Prior belief about per-layer quant sensitivity (first/last layers matter more).
# Only seeds the surrogate; it is calibrated from measured perplexity.
def _sensitivity_prior(i: int) -> float:
    edge = min(i, N_LAYERS - 1 - i)          # 0 at the ends, grows to centre
    return 0.9 - 0.6 * (edge / (N_LAYERS / 2))   # ~0.9 at ends, ~0.3 in middle


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
class State:
    __slots__ = ("prec",)

    def __init__(self, prec):
        self.prec = list(prec)               # list of "q8"/"q4", len N_LAYERS

    def copy(self):
        return State(self.prec)

    def key(self):
        return "".join("8" if p == "q8" else "4" for p in self.prec)

    @staticmethod
    def all_q8():
        return State(["q8"] * N_LAYERS)

    @staticmethod
    def all_q4():
        return State(["q4"] * N_LAYERS)


def neighbors(s: State):
    """Single-layer precision flips (the whole 2-type action space)."""
    out = []
    for i in range(N_LAYERS):
        ns = s.copy()
        ns.prec[i] = "q4" if s.prec[i] == "q8" else "q8"
        out.append((ns, f"layer{i}->{'Q4_0' if ns.prec[i]=='q4' else 'Q8_0'}"))
    return out


# --------------------------------------------------------------------------- #
# Profiler backends
# --------------------------------------------------------------------------- #
class Profiler:
    def __init__(self):
        self.evals = 0
        self.cache = {}

    def measure(self, s: State):
        k = s.key()
        if k in self.cache:
            return self.cache[k]
        self.evals += 1
        m = self._measure(s)
        self.cache[k] = m
        return m


class RealProfiler(Profiler):
    """Quantise with llama-quantize (per-layer --tensor-type), score quality
    with llama-perplexity. Size read from the GGUF; benchmark skipped."""

    def __init__(self, base_ppl, chunks):
        super().__init__()
        self.base_ppl = base_ppl
        self.chunks = chunks

    def _tt(self, s: State):
        args = []
        for i, p in enumerate(s.prec):
            args += ["--tensor-type", f"blk\\.{i}\\..*={TYPES[p]}"]
        args += ["--tensor-type", "token_embd=Q8_0"]   # pin embeddings
        return args

    def _measure(self, s: State):
        out = os.path.join(WORK, f"z7s_{s.key()}.gguf")
        try:
            subprocess.run(
                [os.path.join(BIN, "llama-quantize"), *self._tt(s),
                 BASE_GGUF, out, "Q8_0"],
                check=True, capture_output=True, text=True)
            size_mb = os.path.getsize(out) / (1024 * 1024)
            r = subprocess.run(
                [os.path.join(BIN, "llama-perplexity"), "-m", out,
                 "-f", WIKI, "--chunks", str(self.chunks), "-t", "4"],
                check=True, capture_output=True, text=True)
            ppl = _parse_ppl(r.stdout + r.stderr)
        finally:
            if os.path.exists(out):
                os.remove(out)
        quality = 1.0 - (ppl - self.base_ppl) / self.base_ppl
        return {"size_mb": size_mb, "ppl": ppl, "quality": quality}

    def bench(self, s: State):
        """Full quantize + llama-bench for a final winner (speed table)."""
        out = os.path.join(WORK, f"z7s_win_{s.key()}.gguf")
        subprocess.run(
            [os.path.join(BIN, "llama-quantize"), *self._tt(s),
             BASE_GGUF, out, "Q8_0"],
            check=True, capture_output=True, text=True)
        try:
            r = subprocess.run(
                [os.path.join(BIN, "llama-bench"), "-m", out, "-t", "4",
                 "-o", "json"],
                check=True, capture_output=True, text=True)
            gen, prefill, size = _parse_bench(r.stdout)
        finally:
            if os.path.exists(out):
                os.remove(out)
        return {"prefill_ts": prefill, "gen_ts": gen, "size_mb": size / (1024 * 1024)}


class MockProfiler(Profiler):
    """Off-device dry run. Analytic size + a synthetic perplexity so the search
    logic can be exercised with --mock (no llama.cpp needed)."""

    def __init__(self, base_ppl=10.9, chunks=4, seed=0):
        super().__init__()
        self.base_ppl = base_ppl
        self.rng = random.Random(seed)
        self._sens = [0.02 + 0.05 * _sensitivity_prior(i) + self.rng.uniform(-0.004, 0.004)
                      for i in range(N_LAYERS)]

    def _measure(self, s: State):
        size = 236.0  # token_embd pinned Q8_0
        loss = 0.0
        for i, p in enumerate(s.prec):
            size += LAYER_F16_MB * SIZE_MULT[p]
            if p == "q4":
                loss += self._sens[i]
        ppl = self.base_ppl + loss
        ppl *= 1.0 + self.rng.uniform(-0.01, 0.01)
        return {"size_mb": size, "ppl": ppl, "quality": 1.0 - (ppl - self.base_ppl) / self.base_ppl}

    def bench(self, s: State):
        q4 = sum(1 for p in s.prec if p == "q4")
        return {"prefill_ts": 49 + 18 * (1 - q4 / N_LAYERS),
                "gen_ts": 9.4 - 1.6 * (1 - q4 / N_LAYERS),
                "size_mb": self._measure(s)["size_mb"]}


def _parse_ppl(text):
    m = re.search(r"PPL\s*=\s*([0-9]+\.?[0-9]*)", text)
    if not m:
        raise ValueError("no PPL in perplexity output")
    return float(m.group(1))


def _parse_bench(stdout):
    data = json.loads(stdout)
    gen = prefill = 0.0
    size = 0
    for e in data:
        size = e.get("model_size", size)
        if e.get("n_gen", 0) > 0:
            gen = float(e["avg_ts"])
        elif e.get("n_prompt", 0) > 0:
            prefill = float(e["avg_ts"])
    return gen, prefill, size


# --------------------------------------------------------------------------- #
# Objective + surrogate
# --------------------------------------------------------------------------- #
class Objective:
    def __init__(self, ppl_target, f16_size_mb):
        self.ppl_target = ppl_target
        self.f16 = f16_size_mb

    def feasible(self, m):
        return m["ppl"] <= self.ppl_target + 1e-9

    def score(self, m):
        # higher = better: fractional size reduction vs F16, minus a big penalty
        # for exceeding the perplexity target.
        s = (self.f16 - m["size_mb"]) / self.f16
        if m["ppl"] > self.ppl_target:
            s -= 100.0 * (m["ppl"] - self.ppl_target)
        return s


class Surrogate:
    """Predicts size (near-exact) and perplexity (from a calibrated per-layer
    sensitivity prior) so the planner can rank candidates before paying to
    measure them."""

    def __init__(self, base_ppl):
        self.base_ppl = base_ppl
        self.sens = [0.05 * _sensitivity_prior(i) for i in range(N_LAYERS)]
        self.k = 1.0

    def predict(self, s: State):
        size = 236.0
        loss = 0.0
        for i, p in enumerate(s.prec):
            size += LAYER_F16_MB * SIZE_MULT[p]
            if p == "q4":
                loss += self.sens[i]
        return {"size_mb": size, "ppl": self.base_ppl + self.k * loss,
                "quality": 1.0}

    def calibrate(self, s: State, m):
        pred = self.predict(s)
        pl = pred["ppl"] - self.base_ppl
        ml = m["ppl"] - self.base_ppl
        if pl > 1e-6:
            self.k *= 0.7 + 0.3 * max(0.0, ml / pl)


# --------------------------------------------------------------------------- #
# Search methods (share profiler + objective + budget)
# --------------------------------------------------------------------------- #
def _feasible_best(history, obj):
    feas = [h for h in history if obj.feasible(h[1])]
    if not feas:
        return None
    return min(feas, key=lambda h: h[1]["size_mb"])   # smallest feasible


def _convergence(history, obj):
    curve, best = [], None
    for ev, (state, m) in enumerate(history, 1):
        if obj.feasible(m) and (best is None or m["size_mb"] < best):
            best = m["size_mb"]
        curve.append([ev, best])
    return curve


def _config_upgrading(order, k):
    """Config that upgrades the top-k most-sensitive layers to Q8_0 (rest Q4_0)."""
    up = set(order[:k])
    return State(["q8" if i in up else "q4" for i in range(N_LAYERS)])


def anvil(prof, obj, budget, checkpoint, base_ppl):
    """Surrogate-guided search. Instead of crawling one layer at a time, Anvil
    uses its calibrated per-layer sensitivity model to PREDICT the minimal set of
    layers to protect at Q8_0, measures that config, then binary-searches the
    count -- reaching the smallest feasible model in a handful of evaluations.
    Anchors on both extremes (all-Q4, all-Q8) to calibrate, then predicts."""
    surr = Surrogate(base_ppl)
    history, trace = [], []

    q4, q8 = State.all_q4(), State.all_q8()
    m4 = prof.measure(q4); surr.calibrate(q4, m4); history.append((q4, m4))
    m8 = prof.measure(q8); surr.calibrate(q8, m8); history.append((q8, m8))
    trace.append(f"anchors: all-Q4_0 {m4['size_mb']:.0f}MiB ppl {m4['ppl']:.3f}"
                 f" ({'ok' if obj.feasible(m4) else 'infeasible'}); "
                 f"all-Q8_0 {m8['size_mb']:.0f}MiB ppl {m8['ppl']:.3f}")

    order = sorted(range(N_LAYERS), key=lambda i: surr.sens[i], reverse=True)
    # predicted minimal feasible k (upgrade fewest sensitive layers)
    kpred = next((k for k in range(N_LAYERS + 1)
                  if surr.predict(_config_upgrading(order, k))["ppl"] <= obj.ppl_target),
                 N_LAYERS)
    trace.append(f"surrogate predicts {kpred}/{N_LAYERS} Q8_0 upgrades needed")

    lo, hi, best_k, tried = 0, N_LAYERS, None, {}
    k = kpred
    while prof.evals < budget:
        k = max(lo, min(hi, k))
        if k in tried:                       # pick nearest untried k in [lo,hi]
            cand_ks = [x for x in range(lo, hi + 1) if x not in tried]
            if not cand_ks:
                break
            k = min(cand_ks, key=lambda x: abs(x - k))
        cand = _config_upgrading(order, k)
        m = prof.measure(cand); surr.calibrate(cand, m); history.append((cand, m))
        tried[k] = m
        feas = obj.feasible(m)
        trace.append(f"try {k} upgrades: {m['size_mb']:.0f}MiB ppl {m['ppl']:.3f}"
                     f" {'OK -> shrink' if feas else 'X -> add'}")
        checkpoint(history)
        if feas:
            best_k = k if best_k is None else min(best_k, k)
            hi = k - 1; k = k - 1              # try to protect fewer layers
        else:
            lo = k + 1; k = k + 1              # need to protect more
        if lo > hi:
            break
    return history, trace


def greedy(prof, obj, budget, checkpoint):
    cur = State.all_q4()
    base = prof.measure(cur)
    history = [(cur, base)]
    cur_score = obj.score(base)
    while prof.evals < budget:
        best = None
        for cand, _ in neighbors(cur):
            if prof.evals >= budget:
                break
            m = prof.measure(cand)
            history.append((cand, m))
            checkpoint(history)
            sc = obj.score(m)
            if sc > cur_score:
                cur_score, best = sc, cand
        if best is None:
            break
        cur = best
    return history


def random_search(prof, obj, budget, checkpoint, seed=0):
    rng = random.Random(seed)
    base = prof.measure(State.all_q4())
    history = [(State.all_q4(), base)]
    while prof.evals < budget:
        s = State([rng.choice(("q8", "q4")) for _ in range(N_LAYERS)])
        m = prof.measure(s)
        history.append((s, m))
        checkpoint(history)
    return history


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=14, help="evals per method")
    ap.add_argument("--chunks", type=int, default=4, help="perplexity chunks")
    ap.add_argument("--ppl-target", type=float, default=0.0,
                    help="quality floor: max perplexity. 0 = auto-measure the "
                         "Q4_K_M default at these chunks (recommended, honest bar)")
    ap.add_argument("--mock", action="store_true", help="dry run, no llama.cpp")
    ap.add_argument("--out", default=os.path.join(WORK, "results_z7s.json"))
    args = ap.parse_args()

    def make_prof(base_ppl=None):
        if args.mock:
            return MockProfiler(chunks=args.chunks)
        return RealProfiler(base_ppl, args.chunks)

    # F16 baseline perplexity + the Q4_K_M reference target, both measured at the
    # SAME chunk count as the search so the "beat the default" bar is honest.
    def _ppl_of(gguf):
        r = subprocess.run(
            [os.path.join(BIN, "llama-perplexity"), "-m", gguf,
             "-f", WIKI, "--chunks", str(args.chunks), "-t", "4"],
            check=True, capture_output=True, text=True)
        return _parse_ppl(r.stdout + r.stderr)

    if args.mock:
        base_ppl, f16_size = 10.9, 2945.0
        ppl_target = args.ppl_target if args.ppl_target > 0 else 11.88
    else:
        print("measuring F16 baseline perplexity ...", flush=True)
        base_ppl = _ppl_of(BASE_GGUF)
        f16_size = os.path.getsize(BASE_GGUF) / (1024 * 1024)
        print(f"F16 baseline: ppl {base_ppl:.3f}, {f16_size:.0f} MiB", flush=True)
        if args.ppl_target > 0:
            ppl_target = args.ppl_target
        else:
            print("measuring Q4_K_M reference (the popular default) ...", flush=True)
            ref = os.path.join(WORK, "z7s_q4km_ref.gguf")
            subprocess.run([os.path.join(BIN, "llama-quantize"),
                            BASE_GGUF, ref, "Q4_K_M"],
                           check=True, capture_output=True, text=True)
            try:
                ppl_target = _ppl_of(ref)
                ref_size = os.path.getsize(ref) / (1024 * 1024)
            finally:
                if os.path.exists(ref):
                    os.remove(ref)
            print(f"Q4_K_M default: ppl {ppl_target:.3f}, {ref_size:.0f} MiB "
                  f"(this is the quality bar to beat)", flush=True)

    obj = Objective(ppl_target, f16_size)
    results = {"meta": {"base_ppl": base_ppl, "f16_size_mb": f16_size,
                        "ppl_target": ppl_target, "budget": args.budget,
                        "chunks": args.chunks}, "methods": {}}

    def save():
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    def ck(method):
        return lambda hist: results["methods"].setdefault(method, {}).__setitem__(
            "evals", len(hist)) or save()

    t0 = time.time()
    for name, fn in (("anvil", anvil), ("greedy", greedy), ("random", random_search)):
        print(f"\n=== {name} (budget {args.budget}) ===", flush=True)
        prof = make_prof(base_ppl)
        if name == "anvil":
            history, trace = fn(prof, obj, args.budget, ck(name), base_ppl)
        else:
            history = fn(prof, obj, args.budget, ck(name))
            trace = []
        best = _feasible_best(history, obj)
        entry = {
            "evals": len(history),
            "best": None if best is None else {
                "size_mb": round(best[1]["size_mb"], 1),
                "ppl": round(best[1]["ppl"], 4),
                "config": best[0].key(),
                "n_q4": best[0].key().count("4"),
            },
            "convergence": _convergence(history, obj),
            "trace": trace,
        }
        results["methods"][name] = entry
        save()
        if best:
            print(f"  best feasible: {best[1]['size_mb']:.0f} MiB, "
                  f"ppl {best[1]['ppl']:.3f}, {best[0].key().count('4')}/28 layers Q4_0",
                  flush=True)
        else:
            print("  no feasible config found in budget", flush=True)

    # Benchmark each method's winner for the real speed table.
    print("\n=== benchmarking winners ===", flush=True)
    bprof = make_prof(base_ppl)
    for name in ("anvil", "greedy", "random"):
        b = results["methods"][name]["best"]
        if not b:
            continue
        st = State(["q4" if c == "4" else "q8" for c in b["config"]])
        try:
            speed = bprof.bench(st)
            b.update({"prefill_ts": round(speed["prefill_ts"], 1),
                      "gen_ts": round(speed["gen_ts"], 2)})
            print(f"  {name}: {speed['prefill_ts']:.1f} tok/s prefill, "
                  f"{speed['gen_ts']:.1f} gen, {b['size_mb']:.0f} MiB, ppl {b['ppl']}",
                  flush=True)
        except Exception as e:
            print(f"  {name}: bench failed ({e})", flush=True)
        save()

    print(f"\ndone in {(time.time()-t0)/60:.1f} min -> {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
