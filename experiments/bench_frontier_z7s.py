#!/usr/bin/env python3
"""Measure TTFT/decode for the whole frontier, with an explicit thermal control.

The layer sweep gave size+perplexity for every configuration but benchmarked only
a few. This fills in prefill (TTFT) and decode tokens/sec across the frontier so
a genuine quality-vs-latency Pareto front can be computed.

THERMAL CONTROL: the Snapdragon 695 throttles, so a long benchmarking session
drifts. We bench a fixed reference configuration FIRST and LAST. If the two
readings agree, the intervening points are mutually comparable; if they diverge,
the drift is reported and conclusions are qualified. Nothing is hidden.

Run cooled, plugged in, in Termux:

    termux-wake-lock
    python bench_frontier_z7s.py

Writes bench_frontier_z7s.json after every point (safe to interrupt/resume).
Needs ~2 GB free beyond the F16 base at any moment (one config at a time).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

HOME = os.path.expanduser("~")
BIN = os.path.join(HOME, "llama.cpp", "build", "bin")
BASE = os.path.join(HOME, "qwen15b-f16.gguf")
N_LAYERS = 28


def sensitivity_prior(i):
    edge = min(i, N_LAYERS - 1 - i)
    return 0.9 - 0.6 * (edge / (N_LAYERS / 2))


ORDER = sorted(range(N_LAYERS), key=sensitivity_prior, reverse=True)


def q4_layers_for(k):
    protect = set(ORDER[:k])
    return [i for i in range(N_LAYERS) if i not in protect]


def quantize_k(k, out):
    """k most-sensitive layers Q8_0, rest Q4_0, token_embd Q8_0."""
    tt = []
    for i in q4_layers_for(k):
        tt += ["--tensor-type", f"blk\\.{i}\\..*=Q4_0"]
    tt += ["--tensor-type", "token_embd=Q8_0"]
    subprocess.run([os.path.join(BIN, "llama-quantize"), *tt, BASE, out, "Q8_0"],
                   check=True, capture_output=True, text=True)


def quantize_preset(preset, out):
    subprocess.run([os.path.join(BIN, "llama-quantize"), BASE, out, preset],
                   check=True, capture_output=True, text=True)


def bench(gguf, threads=4):
    r = subprocess.run([os.path.join(BIN, "llama-bench"), "-m", gguf,
                        "-t", str(threads), "-o", "json"],
                       check=True, capture_output=True, text=True)
    data = json.loads(r.stdout)
    prefill = gen = 0.0
    size = 0
    for e in data:
        size = e.get("model_size", size)
        if e.get("n_gen", 0) > 0:
            gen = float(e["avg_ts"])
        elif e.get("n_prompt", 0) > 0:
            prefill = float(e["avg_ts"])
    return prefill, gen, size / (1024 * 1024)


def measure(label, build_fn, store, path):
    out = os.path.join(HOME, f"bf_{label}.gguf")
    print(f"\n[{label}] quantizing ...", flush=True)
    try:
        build_fn(out)
        print(f"[{label}] benchmarking ...", flush=True)
        prefill, gen, size = bench(out)
    finally:
        if os.path.exists(out):
            os.remove(out)
    store[label] = {"prefill_ts": round(prefill, 2),
                    "gen_ts": round(gen, 2),
                    "size_mib": round(size, 1),
                    "t": round(time.time(), 1)}
    json.dump(store, open(path, "w"), indent=2)
    print(f"[{label}] prefill {prefill:.1f} t/s  gen {gen:.2f} t/s  {size:.0f} MiB",
          flush=True)
    return store[label]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="0,8,12,16,20,24,28")
    ap.add_argument("--out", default=os.path.join(HOME, "bench_frontier_z7s.json"))
    args = ap.parse_args()

    store = {}
    if os.path.exists(args.out):
        try:
            store = json.load(open(args.out))
        except Exception:
            store = {}

    t0 = time.time()
    REF_K = 20  # reference configuration for the thermal control

    # --- thermal control: reference FIRST ---
    if "ref_start" not in store:
        measure("ref_start", lambda o: quantize_k(REF_K, o), store, args.out)

    # --- the frontier ---
    for k in [int(x) for x in args.ks.split(",")]:
        label = f"k{k}"
        if label in store:
            print(f"[{label}] cached, skipping", flush=True)
            continue
        measure(label, (lambda kk: (lambda o: quantize_k(kk, o)))(k), store, args.out)

    # --- the unaccelerated default, same session ---
    if "q4_k_m" not in store:
        measure("q4_k_m", lambda o: quantize_preset("Q4_K_M", o), store, args.out)

    # --- thermal control: reference LAST ---
    if "ref_end" not in store:
        measure("ref_end", lambda o: quantize_k(REF_K, o), store, args.out)

    # --- report ---
    a, b = store.get("ref_start"), store.get("ref_end")
    print("\n=== thermal control ===")
    if a and b:
        drift = (b["prefill_ts"] - a["prefill_ts"]) / a["prefill_ts"] * 100
        print(f"  reference prefill: start {a['prefill_ts']:.1f} -> end "
              f"{b['prefill_ts']:.1f} t/s  ({drift:+.1f}%)")
        if abs(drift) <= 10:
            print("  drift within 10% -> points are mutually comparable")
        else:
            print("  DRIFT > 10% -> treat absolute values cautiously; "
                  "re-run cooled if possible")

    print("\n=== frontier ===")
    print(f"  {'config':10s} {'prefill':>9s} {'decode':>8s} {'size MiB':>10s}")
    for label in sorted(store, key=lambda s: (s[0] != "k", s)):
        e = store[label]
        print(f"  {label:10s} {e['prefill_ts']:9.1f} {e['gen_ts']:8.2f} {e['size_mib']:10.1f}")
    print(f"\ndone in {(time.time()-t0)/60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()
