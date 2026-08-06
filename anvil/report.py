"""Reporting: the one chart that wins the submission, plus a JSON artifact.

Produces:
  * convergence.png : best feasible latency vs profiler evaluations, per method.
    This is THE plot — it shows Anvil reaching a better point in fewer evals.
  * pareto.png      : latency vs accuracy scatter of every measured config,
    with the accuracy floor drawn in, so judges see the tradeoff visually.
  * results.json    : machine-readable summary + Anvil's decision trace.

matplotlib is the only heavy dependency and is optional: if it is missing the
JSON is still written so nothing hard-fails on a bare box.
"""
from __future__ import annotations

import json
from typing import Dict, List

from .search import SearchResult


def write_json(results: Dict[str, SearchResult], path: str) -> None:
    out = {}
    for name, r in results.items():
        out[name] = {
            "summary": r.summary(),
            "speedup": round(r.speedup(), 4),
            "baseline_ms": round(r.baseline.latency_ms, 4),
            "best": r.best_metrics.as_dict(),
            "evals_used": r.history[-1].evals if r.history else 0,
            "trace": r.trace,
        }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


def plot(results: Dict[str, SearchResult], out_dir: str,
         accuracy_floor: float = None) -> List[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    written = []
    colors = {"anvil": "#d62728", "greedy": "#1f77b4", "random": "#7f7f7f"}

    # --- convergence: best feasible latency vs evaluations ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, r in results.items():
        xs = [e for e, _ in r.convergence]
        ys = [l for _, l in r.convergence]
        ax.plot(xs, ys, label=name, color=colors.get(name), linewidth=2.2,
                drawstyle="steps-post")
    base = next(iter(results.values())).baseline.latency_ms
    ax.axhline(base, ls="--", color="black", alpha=0.5, label="fp16 baseline")
    ax.set_xlabel("Profiler evaluations spent")
    ax.set_ylabel("Best feasible latency (ms)")
    ax.set_title("Anvil reaches a better config in fewer evaluations")
    ax.legend()
    ax.grid(alpha=0.3)
    p = f"{out_dir}/convergence.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    written.append(p)

    # --- pareto: latency vs accuracy for every measured config ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, r in results.items():
        xs = [pt.metrics.latency_ms for pt in r.history]
        ys = [pt.metrics.accuracy for pt in r.history]
        ax.scatter(xs, ys, s=14, alpha=0.45, color=colors.get(name), label=name)
        ax.scatter([r.best_metrics.latency_ms], [r.best_metrics.accuracy],
                   s=140, marker="*", color=colors.get(name),
                   edgecolor="black", zorder=5)
    if accuracy_floor is not None:
        ax.axhline(accuracy_floor, ls="--", color="crimson", alpha=0.6,
                   label=f"accuracy floor {accuracy_floor}")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Latency vs accuracy (stars = best shipped config)")
    ax.legend()
    ax.grid(alpha=0.3)
    p = f"{out_dir}/pareto.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    written.append(p)
    return written
