"""Experiment 2: how much do you have to remove?

The same simulation read as a defender. A cut is applied `cut_time` seconds into a
generation and we ask whether the request still completes within a deadline, and how
long it takes. Three cut types, from the plan:

  nodes     a random fraction f of all nodes, removed simultaneously
  stage     every node holding one stage (the targeted cut)
  registry  a fraction g of the discovery layer: existing entries lost w.p. g, new
            joiners register w.p. 1-g; live nodes keep serving

and three swarm conditions:

  static    no churn, no joins: what is cut stays cut
  healthy   lambda = 1e-3, joins balance leaves (mean lifetime ~17 min, 3 replicas)
  busy      lambda = 1e-2, joins balance leaves (mean lifetime ~100 s)

Closed forms for the static case are in `static_p_complete`; see NOTES.md §5.

Run:  python -m churnsim.cuts [--seeds 40] [--out out]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import replace
from typing import Dict, List, Optional

from .model import Config, Cut, Strategy, run_many
from .petals import base_costs_from_p0, config_for

FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
SCENARIOS = {"static": (0.0, 0.0), "healthy": (1e-3, 3e-3), "busy": (1e-2, 3e-2)}
SEQ_LEN = 1024
CUT_TIME = 10.0          # ~100 tokens in, chain is warm
DEADLINE_FACTOR = 20.0   # a request that takes 20x the no-failure time counts as stopped


def static_p_complete(kind: str, amount: float, stages: int, replicas: int) -> float:
    """P(a request can still complete) with no joins, straight from the graph.

    nodes:    every stage must keep at least one of its n replicas: (1 - f^n)^S
    stage:    zero if the stage is fully removed
    registry: one; a chain already connected does not need discovery
    """
    if kind == "nodes":
        return (1 - amount ** replicas) ** stages
    if kind == "stage":
        return 0.0
    if kind == "registry":
        return 1.0
    raise ValueError(kind)


def registry_cut_survival(g: float, lam: float, t_remaining: float, stages: int) -> float:
    """P(complete) after losing the whole discovery layer (g = 1) under churn:
    the current chain must outlive the rest of the generation, S nodes each with
    hazard lambda. For g < 1 this is a lower bound."""
    return math.exp(-stages * lam * t_remaining)


def base_config(params: Dict[str, float], lam: float, join: float, replicas: int,
                deadline: float) -> Config:
    base = base_costs_from_p0()
    return config_for(Strategy.REPLAY, SEQ_LEN, 0.0,
                      reconnect_time=params["reconnect_replay"],
                      replay_per_token=params["replay_per_token"], base=base,
                      leave_rate=lam, join_rate=join * replicas / 3 if join else 0.0,
                      replicas=replicas, poll_interval=0.5, time_budget=deadline)


def run(params: Dict[str, float], seeds: int, replicas: int) -> List[dict]:
    nominal = SEQ_LEN / 10.7  # no-failure time at 1024 tokens, from Table 1
    deadline = DEADLINE_FACTOR * nominal
    rows: List[dict] = []
    for scen, (lam, join) in SCENARIOS.items():
        cfg0 = base_config(params, lam, join, replicas, deadline)
        for kind in ("nodes", "registry", "stage"):
            amounts = FRACTIONS if kind != "stage" else [1.0]
            for a in amounts:
                cut = Cut(time=CUT_TIME, kind=kind, amount=(1 if kind == "stage" else a))
                summ = run_many(replace(cfg0, cut=cut), Strategy.REPLAY, seeds=seeds)
                fin = [r for r in summ.runs if r.finished]
                p_done = len(fin) / len(summ.runs)
                t_done = sum(r.elapsed for r in fin) / len(fin) if fin else None
                rows.append(dict(scenario=scen, kind=kind, amount=a, replicas=replicas,
                                 p_complete=p_done,
                                 p_static_formula=static_p_complete(kind, a, cfg0.stages, replicas)
                                 if scen == "static" else "",
                                 mean_time_if_complete=t_done, nominal_time=nominal,
                                 mean_wait=sum(r.wait_time for r in summ.runs) / len(summ.runs),
                                 mean_nodes_cut=sum(r.nodes_cut for r in summ.runs) / len(summ.runs),
                                 seeds=seeds))
                print(f"{scen:<8} {kind:<9} {a:<5} P(complete)={p_done:5.2f} "
                      f"t={'-' if t_done is None else f'{t_done:7.1f}'}s "
                      f"(nominal {nominal:.0f}) wait={rows[-1]['mean_wait']:6.1f}")
    return rows


def plot(rows: List[dict], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"static": "#2a78d6", "healthy": "#eb6834", "busy": "#1baf7a"}
    labels = {"static": "static (no joins)", "healthy": "churn λ=1e-3, joins balance",
              "busy": "churn λ=1e-2, joins balance"}
    replicas = rows[0]["replicas"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=False)

    for ax, kind, xlabel in ((axes[0], "nodes", "fraction of nodes removed"),
                             (axes[1], "registry", "fraction of discovery layer removed")):
        for scen in SCENARIOS:
            sel = [r for r in rows if r["scenario"] == scen and r["kind"] == kind]
            ax.plot([r["amount"] for r in sel], [r["p_complete"] for r in sel],
                    color=colors[scen], lw=2, marker="o", ms=4, label=labels[scen])
        if kind == "nodes":
            xs = [i / 100 for i in range(101)]
            ax.plot(xs, [static_p_complete("nodes", x, 4, replicas) for x in xs],
                    color="#888", lw=1.2, ls="--", label=f"(1 − f^{replicas})^4")
        ax.set_xlabel(xlabel)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"{kind} cut", loc="left", fontsize=11)
    axes[0].set_ylabel("P(request completes within 20× nominal)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")

    ax = axes[2]
    for i, kind in enumerate(("nodes", "registry", "stage")):
        for j, scen in enumerate(SCENARIOS):
            sel = [r for r in rows if r["scenario"] == scen and r["kind"] == kind
                   and r["mean_time_if_complete"] is not None
                   and (kind == "stage" or r["amount"] in (0.5, 0.9))]
            for r in sel:
                x = i + (j - 1) * 0.25
                y = r["mean_time_if_complete"] / r["nominal_time"]
                ax.scatter([x], [y], color=colors[scen], s=36 if r["amount"] < 0.9 else 70,
                           zorder=3, edgecolors="white", linewidths=0.8)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["nodes 50% / 90%", "registry 50% / 90%", "stage"])
    ax.set_yscale("log")
    ax.set_ylabel("completion time / nominal (finished runs)")
    ax.set_title("cost when it does complete", loc="left", fontsize=11)
    ax.axhline(1.0, color="#bbb", lw=1)
    for a in axes:
        a.grid(True, color="#e4e4e0", lw=0.8); a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.suptitle(f"Experiment 2: simultaneous cuts, replay strategy, {SEQ_LEN} tokens, "
                 f"{replicas} replicas/stage, cut at t={CUT_TIME:.0f}s", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--replicas", type=int, default=3)
    ap.add_argument("--out", default="out")
    ap.add_argument("--calibration", default="out/calibration.json")
    args = ap.parse_args()
    with open(args.calibration) as f:
        params = json.load(f)["params"]
    os.makedirs(args.out, exist_ok=True)
    rows = run(params, args.seeds, args.replicas)
    tag = f"_n{args.replicas}" if args.replicas != 3 else ""
    with open(os.path.join(args.out, f"cuts{tag}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    plot(rows, os.path.join(args.out, f"cuts{tag}.png"))


if __name__ == "__main__":
    main()
