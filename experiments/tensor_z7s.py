#!/usr/bin/env python3
"""Tensor-granularity probe: can a near-free per-TENSOR protection beat the
Q4_K_M default on quality *at or below its size*?

The layer-granularity sweep forced all 7 tensors of a block to the same type.
But sensitivity and size are wildly uneven within a block: attn_v/attn_k are
0.75 MiB each (protecting them costs ~0.19 MiB/layer -- essentially free) while
ffn_* are 26.25 MiB each. This probes the cheap protections first.

Everything stays inside the KleidiAI-accelerated types {Q4_0, Q8_0}.

    python tensor_z7s.py --chunks 4

Writes tensor_z7s.json after every config (safe to interrupt/re-run).
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

# name -> list of --tensor-type patterns promoted to Q8_0 (base type = Q4_0).
# token_embd is pinned to Q8_0 in every config, as in the layer sweep.
CONFIGS = {
    "v_only":        ["attn_v"],
    "v_k":           ["attn_v", "attn_k"],
    "v_k_out":       ["attn_v", "attn_k", "attn_output"],
    "v_k_ffndown":   ["attn_v", "attn_k", "ffn_down"],
}


def parse_ppl(text):
    m = re.search(r"PPL\s*=\s*([0-9]+\.?[0-9]*)", text)
    if not m:
        raise ValueError("no PPL found")
    return float(m.group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(HOME, "tensor_z7s.json"))
    ap.add_argument("--only", default="", help="comma-separated config names")
    args = ap.parse_args()

    data = {"chunks": args.chunks, "configs": {}}
    if os.path.exists(args.out):
        try:
            data = json.load(open(args.out))
            data.setdefault("configs", {})
        except Exception:
            pass

    names = ([n for n in args.only.split(",") if n] or list(CONFIGS))
    t0 = time.time()
    for name in names:
        if name in data["configs"]:
            print(f"{name}: cached, skipping", flush=True)
            continue
        pats = CONFIGS[name]
        out = os.path.join(HOME, f"tt_{name}.gguf")
        tt = []
        for p in pats:
            tt += ["--tensor-type", f"{p}=Q8_0"]
        tt += ["--tensor-type", "token_embd=Q8_0"]
        print(f"\n{name}: base Q4_0, promoting {pats} -> Q8_0", flush=True)
        try:
            subprocess.run([os.path.join(BIN, "llama-quantize"), *tt,
                            BASE, out, "Q4_0"],
                           check=True, capture_output=True, text=True)
            size = os.path.getsize(out) / (1024 * 1024)
            print(f"{name}: {size:.1f} MiB, measuring perplexity ...", flush=True)
            r = subprocess.run([os.path.join(BIN, "llama-perplexity"),
                                "-m", out, "-f", WIKI,
                                "--chunks", str(args.chunks), "-t", "4"],
                               check=True, capture_output=True, text=True)
            ppl = parse_ppl(r.stdout + r.stderr)
        finally:
            if os.path.exists(out):
                os.remove(out)
        data["configs"][name] = {"promoted": pats,
                                 "size_mib": round(size, 1),
                                 "ppl": round(ppl, 4)}
        json.dump(data, open(args.out, "w"), indent=2)
        print(f"{name}: ppl {ppl:.4f}   ({(time.time()-t0)/60:.1f} min)", flush=True)

    print("\n--- results (bar to beat: Q4_K_M = 935 MiB, ppl 9.605) ---")
    for n, c in data["configs"].items():
        verdict = "BEATS DEFAULT" if c["ppl"] <= 9.605 else ""
        smaller = "and SMALLER" if c["size_mib"] <= 940 and c["ppl"] <= 9.605 else ""
        print(f"  {n:14s} {c['size_mib']:7.1f} MiB   ppl {c['ppl']:.4f}  {verdict} {smaller}")
    print(f"\ndone in {(time.time()-t0)/60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()
