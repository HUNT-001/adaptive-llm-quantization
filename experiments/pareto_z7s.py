#!/usr/bin/env python3
"""Compute and plot the quality/latency Pareto frontier from measured data.

Merges the layer sweep (size + perplexity) with the frontier benchmark
(prefill/decode tokens/sec) and answers the question the whole project is about:

    Among configurations that run on Arm-accelerated kernels, which are
    Pareto-optimal in (quality, time-to-first-token)? And where does the
    unaccelerated Q4_K_M default sit relative to that frontier?

Usage:
    python pareto_z7s.py --bench bench_frontier_z7s.json

If --bench is omitted it falls back to whatever TTFT values are already recorded
in results_z7s.json, so the script still runs (with fewer points).
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = json.load(open(os.path.join(HERE, "results_z7s.json")))

ACC = "#177245"     # accelerated
FRONT = "#d62728"   # pareto-optimal
DEF = "#1f77b4"     # the unaccelerated default


def load_points(bench_path):
    """-> list of dicts with ppl, prefill_ts, size_mib, label, accelerated."""
    ppl_by_k = {p["k_protected"]: p for p in RES["frontier_sweep"]["points"]}
    pts = []

    if bench_path and os.path.exists(bench_path):
        b = json.load(open(bench_path))
        for label, e in b.items():
            if label.startswith("k") and label[1:].isdigit():
                k = int(label[1:])
                if k in ppl_by_k:
                    pts.append({"label": f"{k} protected",
                                "k": k,
                                "ppl": ppl_by_k[k]["ppl"],
                                "prefill_ts": e["prefill_ts"],
                                "gen_ts": e["gen_ts"],
                                "size_mib": e["size_mib"],
                                "accelerated": True})
        if "q4_k_m" in b:
            pts.append({"label": "Q4_K_M default", "k": None,
                        "ppl": RES["search"]["reference_ppl_4chunk"]["q4_k_m"],
                        "prefill_ts": b["q4_k_m"]["prefill_ts"],
                        "gen_ts": b["q4_k_m"]["gen_ts"],
                        "size_mib": b["q4_k_m"]["size_mib"],
                        "accelerated": False})
    else:
        # fallback: the few points already measured
        h = RES["head_to_head"]
        pts.append({"label": "20 protected (Anvil)", "k": 20,
                    "ppl": h["anvil"]["ppl_4chunk"],
                    "prefill_ts": h["anvil"]["prefill_ts"],
                    "gen_ts": h["anvil"]["gen_ts"],
                    "size_mib": h["anvil"]["size_mib"], "accelerated": True})
        pts.append({"label": "Q4_K_M default", "k": None,
                    "ppl": h["q4_k_m"]["ppl_4chunk"],
                    "prefill_ts": h["q4_k_m"]["prefill_ts"],
                    "gen_ts": h["q4_k_m"]["gen_ts"],
                    "size_mib": h["q4_k_m"]["size_mib"], "accelerated": False})
    return pts


# Objectives: (key, better_is) where better_is = +1 to maximise, -1 to minimise.
OBJECTIVES = [("prefill_ts", +1), ("gen_ts", +1), ("ppl", -1), ("size_mib", -1)]


def _dominates(q, p, objectives):
    """q dominates p: at least as good on every objective, strictly better on one."""
    better_somewhere = False
    for key, sign in objectives:
        qv, pv = sign * q[key], sign * p[key]
        if qv < pv - 1e-9:
            return False
        if qv > pv + 1e-9:
            better_somewhere = True
    return better_somewhere


def pareto(pts, objectives=None, among_accelerated=True):
    """Pareto-optimal set. Default is the full 4-objective problem: maximise
    prefill and decode tokens/sec, minimise perplexity and size.

    Using a single speed axis would be misleading on this hardware, because
    prefill and decode favour OPPOSITE quantizations (prefill is compute-bound
    and likes Q8_0's KleidiAI int8 kernels; decode is bandwidth-bound and likes
    Q4_0's smaller weights). A configuration that looks dominated on prefill can
    be Pareto-optimal once decode and size are counted."""
    objectives = objectives or OBJECTIVES
    pool = [p for p in pts if p["accelerated"]] if among_accelerated else list(pts)
    front = [p for p in pool
             if not any(_dominates(q, p, objectives) for q in pool if q is not p)]
    return sorted(front, key=lambda p: -p["prefill_ts"])


def workload_optima(pts):
    """Best accelerated config for each workload profile a developer might care
    about. This is what makes the frontier actionable rather than decorative."""
    acc = [p for p in pts if p["accelerated"]]
    if not acc:
        return {}
    return {
        "prompt_heavy (best TTFT)": max(acc, key=lambda p: p["prefill_ts"]),
        "generation_heavy (best decode)": max(acc, key=lambda p: p["gen_ts"]),
        "storage_constrained (smallest)": min(acc, key=lambda p: p["size_mib"]),
        "quality_first (lowest ppl)": min(acc, key=lambda p: p["ppl"]),
    }


def efficiency_table(pts):
    """Quality bought per unit of latency given up, relative to the fastest
    accelerated config. Answers: 'if I spend more time-to-first-token, how much
    quality do I actually get back?'"""
    acc = sorted([p for p in pts if p["accelerated"]],
                 key=lambda p: -p["prefill_ts"])
    if not acc:
        return []
    base = acc[0]
    rows = []
    for p in acc[1:]:
        d_ppl = base["ppl"] - p["ppl"]            # positive = quality gained
        d_ms = (1000.0 / p["prefill_ts"]) - (1000.0 / base["prefill_ts"])
        rows.append({"config": p["label"], "d_quality_ppl": round(d_ppl, 4),
                     "d_ttft_ms_per_token": round(d_ms, 3),
                     "ppl_gained_per_ms": round(d_ppl / d_ms, 4) if d_ms > 1e-9 else None})
    return rows


def plot(pts, front, out):
    """Two panels, because prefill and decode disagree about the best config."""
    acc = [p for p in pts if p["accelerated"]]
    unacc = [p for p in pts if not p["accelerated"]]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), sharey=True)

    for ax, xkey, title in (
        (axes[0], "prefill_ts", "Prefill / time-to-first-token (tok/s)"),
        (axes[1], "gen_ts", "Decode / sustained generation (tok/s)"),
    ):
        ax.scatter([p[xkey] for p in acc], [p["ppl"] for p in acc],
                   s=80, color=ACC, edgecolor="black", linewidth=0.6, zorder=3,
                   label="KleidiAI-accelerated")
        for p in acc:
            ax.annotate(str(p["k"]), (p[xkey], p["ppl"]),
                        textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=8, color="#444")
        f = sorted([p for p in front], key=lambda p: p[xkey])
        if len(f) > 1:
            ax.plot([p[xkey] for p in f], [p["ppl"] for p in f], "--",
                    color=FRONT, linewidth=2.0, zorder=2,
                    label="Pareto-optimal (4-objective)")
        for p in unacc:
            ax.scatter([p[xkey]], [p["ppl"]], marker="s", s=170, color=DEF,
                       edgecolor="black", linewidth=0.8, zorder=5,
                       label="Q4_K_M (NOT accelerated)")
        # mark the winner on this axis
        if acc:
            best = max(acc, key=lambda p: p[xkey])
            ax.annotate(f"best here:\n{best['k']} protected",
                        (best[xkey], best["ppl"]),
                        textcoords="offset points", xytext=(-12, -34),
                        fontsize=9, fontweight="bold", color=FRONT,
                        ha="center")
        ax.set_xlabel(title + "  →  faster is better")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Perplexity  →  lower is better")
    axes[0].legend(loc="upper right", fontsize=8.5)
    fig.suptitle("Prefill and decode disagree about the best quantization\n"
                 "labels = layers protected at Q8_0 · Qwen2.5-1.5B · Snapdragon 695",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=os.path.join(HERE, "bench_frontier_z7s.json"))
    args = ap.parse_args()

    pts = load_points(args.bench)
    front = pareto(pts)
    eff = efficiency_table(pts)
    optima = workload_optima(pts)

    print(f"{len(pts)} configurations · {len(front)} Pareto-optimal on "
          f"(prefill, decode, perplexity, size)\n")
    print(f"  {'config':22s} {'prefill':>8s} {'decode':>8s} {'ppl':>8s} {'size':>8s}  pareto")
    for p in sorted(pts, key=lambda p: -p["prefill_ts"]):
        mark = "  *" if p in front else ""
        print(f"  {p['label']:22s} {p['prefill_ts']:8.1f} {p['gen_ts']:8.2f} "
              f"{p['ppl']:8.3f} {p['size_mib']:8.0f}{mark}")

    if optima:
        print("\nbest accelerated configuration per workload profile:")
        for profile, p in optima.items():
            print(f"  {profile:32s} -> {p['label']}")
        pf = optima["prompt_heavy (best TTFT)"]["label"]
        gf = optima["generation_heavy (best decode)"]["label"]
        if pf != gf:
            print(f"\n  NOTE: prefill and decode select DIFFERENT configurations "
                  f"({pf} vs {gf}).\n  Prefill is compute-bound and favours Q8_0's "
                  f"KleidiAI int8 kernels; decode is bandwidth-bound and favours\n"
                  f"  Q4_0's smaller weights. 'Smaller is faster' is only true for "
                  f"decode on this device.")

    if eff:
        print("\nquality bought per millisecond of added time-to-first-token:")
        for r in eff:
            v = r["ppl_gained_per_ms"]
            print(f"  {r['config']:22s} {r['d_quality_ppl']:+7.3f} ppl for "
                  f"{r['d_ttft_ms_per_token']:+6.2f} ms/token"
                  + (f"   ({v:.4f} ppl/ms)" if v is not None else ""))

    out = plot(pts, front, os.path.join(HERE, "z7s_pareto.png"))
    summary = {"objectives": [o[0] for o in OBJECTIVES],
               "points": pts, "pareto_front": front,
               "workload_optima": {k: v["label"] for k, v in optima.items()},
               "efficiency": eff}
    json.dump(summary, open(os.path.join(HERE, "pareto_z7s.json"), "w"), indent=2)
    print("\nwrote:", out)
    print("wrote:", os.path.join(HERE, "pareto_z7s.json"))


if __name__ == "__main__":
    main()
