#!/usr/bin/env python3
"""k-sweep: map the full size/quality frontier on the Z7s.

For k = 0, 4, 8, ... 28, protect the k most quant-sensitive layers at Q8_0 and
leave the rest at Q4_0 (both KleidiAI-accelerated), then measure on-disk size and
perplexity. This traces the frontier Anvil searched, so we can show that the
planner located the minimal feasible configuration in far fewer measurements
than mapping the curve requires.

Run in Termux from ~ (needs qwen15b-f16.gguf):

    python sweep_z7s.py --chunks 4

Writes/updates sweep_z7s.json after every point, so it is safe to interrupt and
re-run (already-measured k values are skipped).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time

HOME = os.path.expanduser("~")
BIN = os.path.join(HOME, "llama.cpp", "build", "bin")
BASE = os.path.join(HOME, "qwen15b-f16.gguf")
WIKI = os.path.join(HOME, "llama.cpp", "wikitext-2-raw", "wiki.test.raw")
N_LAYERS = 28


def sensitivity_prior(i):
    edge = min(i, N_LAYERS - 1 - i)
    return 0.9 - 0.6 * (edge / (N_LAYERS / 2))


# same deterministic ordering the planner used
ORDER = sorted(range(N_LAYERS), key=sensitivity_prior, reverse=True)


def config_for(k):
    """k most-sensitive layers -> Q8_0, the rest -> Q4_0."""
    protect = set(ORDER[:k])
    return ["q8" if i in protect else "q4" for i in range(N_LAYERS)]


def tensor_args(prec):
    args = []
    for i, p in enumerate(prec):
        if p == "q4":                     # base type is Q8_0, override the Q4s
            args += ["--tensor-type", f"blk\\.{i}\\..*=Q4_0"]
    args += ["--tensor-type", "token_embd=Q8_0"]
    return args


def parse_ppl(text):
    m = re.search(r"PPL\s*=\s*([0-9]+\.?[0-9]*)", text)
    if not m:
        raise ValueError("no PPL found")
    return float(m.group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--ks", default="0,4,8,12,16,20,24,28")
    ap.add_argument("--out", default=os.path.join(HOME, "sweep_z7s.json"))
    args = ap.parse_args()

    data = {"chunks": args.chunks, "points": {}}
    if os.path.exists(args.out):
        try:
            data = json.load(open(args.out))
            data.setdefault("points", {})
        except Exception:
            pass

    ks = [int(x) for x in args.ks.split(",")]
    t0 = time.time()
    for k in ks:
        if str(k) in data["points"]:
            print(f"k={k}: cached, skipping", flush=True)
            continue
        prec = config_for(k)
        out = os.path.join(HOME, f"sweep_k{k}.gguf")
        print(f"k={k}: quantizing ({prec.count('q4')} layers Q4_0) ...", flush=True)
        try:
            subprocess.run(
                [os.path.join(BIN, "llama-quantize"), *tensor_args(prec),
                 BASE, out, "Q8_0"],
                check=True, capture_output=True, text=True)
            size_mib = os.path.getsize(out) / (1024 * 1024)
            print(f"k={k}: {size_mib:.0f} MiB, measuring perplexity ...", flush=True)
            r = subprocess.run(
                [os.path.join(BIN, "llama-perplexity"), "-m", out, "-f", WIKI,
                 "--chunks", str(args.chunks), "-t", "4"],
                check=True, capture_output=True, text=True)
            ppl = parse_ppl(r.stdout + r.stderr)
        finally:
            if os.path.exists(out):
                os.remove(out)
        data["points"][str(k)] = {"k_protected": k,
                                  "n_q4_layers": prec.count("q4"),
                                  "size_mib": round(size_mib, 1),
                                  "ppl": round(ppl, 4)}
        json.dump(data, open(args.out, "w"), indent=2)
        print(f"k={k}: ppl {ppl:.4f}  ({(time.time()-t0)/60:.1f} min elapsed)\n",
              flush=True)

    print("frontier:")
    for k in sorted(data["points"], key=int):
        p = data["points"][k]
        print(f"  {p['k_protected']:2d} protected -> {p['size_mib']:7.1f} MiB   ppl {p['ppl']:.4f}")
    print(f"\ndone in {(time.time()-t0)/60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()
