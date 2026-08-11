#!/usr/bin/env python3
"""Charts for the real iQOO Z7s measurements (reads experiments/results_z7s.json).

Produces:
  z7s_kleidiai.png  - the key finding: KleidiAI-accelerated types (Q4_0/Q8_0)
                      vs the popular Q4_K_M default, prefill tokens/sec.
  z7s_headtohead.png- Anvil's searched config vs the Q4_K_M default across
                      prefill / quality / size, measured back-to-back.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "results_z7s.json")))

ACC = "#d62728"     # KleidiAI-accelerated
NOT = "#7f7f7f"     # not accelerated
ANVIL = "#d62728"
DEF = "#1f77b4"


def kleidiai_chart():
    u = D["uniform_baselines"]
    names = ["F16", "Q4_K_M\n(popular default)", "Q4_0", "Q8_0"]
    vals = [u["f16"]["prefill_ts"], u["q4_k_m"]["prefill_ts"],
            u["q4_0"]["prefill_ts"], u["q8_0"]["prefill_ts"]]
    cols = [NOT, NOT, ACC, ACC]

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    bars = ax.bar(names, vals, color=cols, edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}",
                ha="center", fontweight="bold")
    ax.set_ylabel("Prefill / time-to-first-token (tokens/sec)")
    ax.set_title("KleidiAI accelerates only Q4_0 and Q8_0 — not the K-quants\n"
                 "Qwen2.5-1.5B on iQOO Z7s (Snapdragon 695), 4 threads",
                 fontsize=11)
    ax.set_ylim(0, max(vals) * 1.22)
    ax.grid(axis="y", alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=ACC),
               plt.Rectangle((0, 0), 1, 1, color=NOT)]
    ax.legend(handles, ["KleidiAI-accelerated", "not accelerated"], loc="upper left")
    p = os.path.join(HERE, "z7s_kleidiai.png")
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    return p


def head_to_head_chart():
    h = D["head_to_head"]
    a, q = h["anvil"], h["q4_k_m"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    panels = [
        ("Prefill (tok/s)\nhigher = better", a["prefill_ts"], q["prefill_ts"], False),
        ("Perplexity\nlower = better", a["ppl_4chunk"], q["ppl_4chunk"], True),
        ("Size (MiB)\nlower = better", a["size_mib"], q["size_mib"], True),
    ]
    for ax, (title, av, qv, lower_better) in zip(axes, panels):
        bars = ax.bar(["Anvil", "Q4_K_M"], [av, qv],
                      color=[ANVIL, DEF], edgecolor="black", linewidth=0.6)
        for b, v in zip(bars, [av, qv]):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.01, f"{v:g}",
                    ha="center", fontweight="bold")
        anvil_wins = (av < qv) if lower_better else (av > qv)
        ax.set_title(title + ("   ✓ Anvil" if anvil_wins else "   ✗ Anvil"),
                     fontsize=10,
                     color=("#177245" if anvil_wins else "#a33"))
        ax.set_ylim(0, max(av, qv) * 1.2)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Anvil's searched per-layer config vs the Q4_K_M default "
                 "(back-to-back on a cooled iQOO Z7s)", fontsize=11)
    p = os.path.join(HERE, "z7s_headtohead.png")
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    return p


def frontier_chart():
    f = D["frontier_sweep"]
    pts = f["points"]
    bar = D["search"]["reference_ppl_4chunk"]["q4_k_m"]
    q4km_size = D["uniform_baselines"]["q4_k_m"]["size_mib"]

    xs = [p["size_mib"] for p in pts]
    ys = [p["ppl"] for p in pts]
    sel = next(p for p in pts if p.get("anvil_selected"))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, "-", color="#999", zorder=1, linewidth=1.6)
    for p in pts:
        ok = p["feasible"]
        ax.scatter(p["size_mib"], p["ppl"], s=70, zorder=3,
                   color=("#177245" if ok else "#bbb"),
                   edgecolor="black", linewidth=0.6)
        ax.annotate(f"{p['k_protected']}", (p["size_mib"], p["ppl"]),
                    textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8, color="#555")

    # quality bar = the popular default
    ax.axhline(bar, ls="--", color=DEF, linewidth=1.8)
    ax.text(xs[0], bar - 0.045, f"  Q4_K_M default (ppl {bar}) — must be at or below this",
            color=DEF, fontsize=9, va="top")
    ax.scatter([q4km_size], [bar], marker="s", s=90, color=DEF,
               edgecolor="black", zorder=4, label="Q4_K_M default")

    # Anvil's pick
    ax.scatter([sel["size_mib"]], [sel["ppl"]], s=340, marker="*",
               color=ANVIL, edgecolor="black", linewidth=0.8, zorder=5,
               label="Anvil's choice (smallest feasible)")
    ax.annotate("Anvil", (sel["size_mib"], sel["ppl"]),
                textcoords="offset points", xytext=(14, -16),
                fontsize=11, fontweight="bold", color=ANVIL)

    ax.set_xlabel("Model size on disk (MiB)  →  smaller is better")
    ax.set_ylabel("Perplexity  →  lower is better")
    ax.set_title("The size/quality frontier Anvil searched\n"
                 "point labels = layers protected at Q8_0 (rest Q4_0); "
                 "green = beats the default's quality", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    p = os.path.join(HERE, "z7s_frontier.png")
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    return p


if __name__ == "__main__":
    for p in (kleidiai_chart(), head_to_head_chart(), frontier_chart()):
        print("wrote:", p)
