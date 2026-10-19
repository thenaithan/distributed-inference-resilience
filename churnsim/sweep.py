"""Sweep per-hop failure probability and plot throughput, Petals Appendix F style.

Run:  python -m churnsim.sweep [--seeds 30] [--out out]
Reads fitted parameters from out/calibration.json (run calibrate first), writes
out/sweep.csv and out/throughput_vs_failure.png with the paper's Table 1 points overlaid.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Optional

from .model import Strategy, run_many
from .petals import SEQ_LENS, TABLE1, base_costs_from_p0, config_for

P_GRID = [0.0, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 3e-2, 5e-2]

# Default categorical palette, fixed slot order; see dataviz palette reference.
COLORS = {Strategy.REPLAY: "#2a78d6", Strategy.RESTART: "#eb6834", Strategy.CACHELESS: "#1baf7a"}
LABELS = {Strategy.REPLAY: "reroute + replay (Alg. 1)",
          Strategy.RESTART: "caching with restarts",
          Strategy.CACHELESS: "cache-less"}


def reconnect_for(strat: Strategy, params: Dict[str, float]) -> float:
    return params[f"reconnect_{strat.value}"]


def run_sweep(params: Dict[str, float], seeds: int, time_budget: float) -> List[dict]:
    base = base_costs_from_p0()
    rows = []
    for strat in Strategy:
        for T in SEQ_LENS:
            for p in P_GRID:
                cfg = config_for(strat, T, p, reconnect_time=reconnect_for(strat, params),
                                 replay_per_token=params["replay_per_token"], base=base,
                                 time_budget=time_budget)
                summ = run_many(cfg, strat, seeds=seeds)
                v = summ.mean_steps_per_s
                rows.append(dict(strategy=strat.value, seq_len=T, p=p,
                                 steps_per_s=v, finished=summ.n_finished, seeds=seeds,
                                 censored=summ.censored, observed_p=summ.mean_observed_p))
                print(f"{strat.value:<10} T={T:<5} p={p:<7.0e} -> "
                      f"{'DNF' if v is None else f'{v:6.2f}'} steps/s "
                      f"({summ.n_finished}/{seeds} finished)")
    return rows


def plot(rows: List[dict], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, T in zip(axes, SEQ_LENS):
        for strat in Strategy:
            sel = [r for r in rows if r["strategy"] == strat.value and r["seq_len"] == T]
            # p=0 cannot sit on a log axis; plot it at the left edge as the baseline
            xs = [max(r["p"], 5e-5) for r in sel]
            ys = [r["steps_per_s"] if r["steps_per_s"] is not None else float("nan") for r in sel]
            ax.plot(xs, ys, color=COLORS[strat], lw=2, marker="o", ms=4, label=LABELS[strat])
            cx = [x for x, r in zip(xs, sel) if r["censored"]]
            cy = [y for y, r in zip(ys, sel) if r["censored"]]
            if cx:
                ax.scatter(cx, cy, marker="v", color=COLORS[strat], s=40, zorder=6)
            paper = TABLE1[(strat, T)]
            px = [max(p, 5e-5) for p, v in paper.items() if v is not None]
            py = [v for v in paper.values() if v is not None]
            ax.scatter(px, py, facecolors="white", edgecolors=COLORS[strat],
                       s=70, zorder=5, linewidths=1.8)
            # label the right end of the line directly
            last = next(((x, y) for x, y in reversed(list(zip(xs, ys))) if y == y), None)
            if last:
                ax.annotate(LABELS[strat], last, xytext=(6, 0), textcoords="offset points",
                            va="center", fontsize=8, color="#444")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(4e-5, 0.12)
        ax.set_title(f"{T} tokens", fontsize=11, loc="left")
        ax.set_xlabel("per-hop failure probability p   (p = 0 shown at left edge)")
        ax.grid(True, which="major", color="#e4e4e0", lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("throughput, steps/s")
    axes[0].scatter([], [], facecolors="white", edgecolors="#444", s=70, linewidths=1.8,
                    label="Petals Table 1 (measured)")
    axes[0].scatter([], [], marker="v", color="#444", s=40,
                    label="some runs hit the 1 h budget: upper bound")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Sequential inference under per-hop failures: simulation vs Petals (BLOOM-7.1B, 4 stages)",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--out", default="out")
    ap.add_argument("--calibration", default="out/calibration.json")
    ap.add_argument("--time-budget", type=float, default=3600.0)
    args = ap.parse_args()
    with open(args.calibration) as f:
        params = json.load(f)["params"]
    os.makedirs(args.out, exist_ok=True)
    rows = run_sweep(params, args.seeds, args.time_budget)
    with open(os.path.join(args.out, "sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    plot(rows, os.path.join(args.out, "throughput_vs_failure.png"))


if __name__ == "__main__":
    main()
