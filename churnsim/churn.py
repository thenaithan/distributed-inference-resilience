"""Bridge from node churn (leave rate lambda) to the per-hop failure probability p.

Petals varies p directly. The experiment plan and Experiment 2 are phrased in terms of nodes
leaving at a rate lambda. The two are related by how often the client touches a node:
a node contacted once per generation step, with lifetime Exp(lambda), has died since
the last contact with probability

    p_eff = 1 - exp(-lambda * tau_step)  ~=  lambda * tau_step

where tau_step is the time between successive contacts (about S hops plus whatever
recovery happened in between). This script sweeps lambda with joins balancing leaves
(join_rate = replicas * lambda per stage, so the live population is stationary),
records the failure rate the client actually observes, and compares throughput with
the p-mode curve evaluated at that observed p. If the two agree, the churn model and
the Petals model are the same experiment in different units, and a defender can
reason in either.

Run:  python -m churnsim.churn [--seeds 20] [--out out]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import List

from .model import Strategy, run_many
from .petals import base_costs_from_p0, config_for

LAMBDAS = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1]
REPLICAS = 3


def run(params, seq_len: int, seeds: int, strategy: Strategy) -> List[dict]:
    base = base_costs_from_p0()
    rc = params[f"reconnect_{strategy.value}"]
    rpt = params["replay_per_token"]
    rows = []
    for lam in LAMBDAS:
        cfg = config_for(strategy, seq_len, 0.0, reconnect_time=rc, replay_per_token=rpt,
                         base=base, leave_rate=lam, join_rate=REPLICAS * lam,
                         replicas=REPLICAS, poll_interval=0.5)
        churn = run_many(cfg, strategy, seeds=seeds)
        p_obs = churn.mean_observed_p
        # the p-mode curve at the observed p, same costs
        cfg_p = config_for(strategy, seq_len, p_obs, reconnect_time=rc,
                           replay_per_token=rpt, base=base)
        pmode = run_many(cfg_p, strategy, seeds=seeds)
        # naive: exposure = the no-failure step time
        tau_step = cfg.stages * cfg.hop_time + cfg.stages * cfg.hop_time_per_token * seq_len / 2
        p_pred = 1 - math.exp(-lam * tau_step)
        # self-consistent: exposure = the realised mean step time, recoveries included
        tau_real = sum(r.elapsed for r in churn.runs) / len(churn.runs) / seq_len
        p_pred_real = 1 - math.exp(-lam * tau_real)
        waits = sum(r.wait_time for r in churn.runs) / len(churn.runs)
        rows.append(dict(strategy=strategy.value, seq_len=seq_len, leave_rate=lam,
                         p_observed=p_obs, p_predicted=p_pred, p_predicted_realised=p_pred_real,
                         steps_per_s_churn=churn.mean_steps_per_s,
                         steps_per_s_pmode=pmode.mean_steps_per_s,
                         finished=churn.n_finished, censored=churn.censored,
                         mean_wait_s=waits))
        print(f"lambda={lam:<6.0e} p_obs={p_obs:.2e} p_naive={p_pred:.2e} p_real={p_pred_real:.2e}  "
              f"churn={_f(churn.mean_steps_per_s)} pmode={_f(pmode.mean_steps_per_s)} "
              f"wait={waits:.1f}s ({churn.n_finished}/{seeds})")
    return rows


def _f(v):
    return "  DNF" if v is None else f"{v:5.2f}"


def plot(rows: List[dict], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    lam = [r["leave_rate"] for r in rows]

    ax = axes[0]
    ax.plot(lam, [r["p_predicted"] for r in rows], color="#bbb", lw=1.5, ls=":",
            label="1 - exp(-λ·τ_step), no-failure step time")
    ax.plot(lam, [r["p_predicted_realised"] for r in rows], color="#888", lw=1.5, ls="--",
            label="1 - exp(-λ·τ), realised step time")
    ax.plot(lam, [r["p_observed"] for r in rows], color="#2a78d6", lw=2, marker="o", ms=4,
            label="observed by client")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("node leave rate λ (1/s)")
    ax.set_ylabel("per-hop failure probability")
    ax.set_title("churn → effective p", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.plot(lam, [r["steps_per_s_pmode"] or float("nan") for r in rows], color="#888",
            lw=1.5, ls="--", marker="o", ms=4, label="p-mode at observed p")
    ax.plot(lam, [r["steps_per_s_churn"] or float("nan") for r in rows], color="#2a78d6",
            lw=2, marker="o", ms=4, label="node churn")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("node leave rate λ (1/s)")
    ax.set_ylabel("throughput, steps/s")
    ax.set_title(f"replay strategy, {rows[0]['seq_len']} tokens, {REPLICAS} replicas/stage",
                 loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    for a in axes:
        a.grid(True, color="#e4e4e0", lw=0.8); a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--out", default="out")
    ap.add_argument("--calibration", default="out/calibration.json")
    args = ap.parse_args()
    with open(args.calibration) as f:
        params = json.load(f)["params"]
    os.makedirs(args.out, exist_ok=True)
    rows = run(params, args.seq_len, args.seeds, Strategy.REPLAY)
    with open(os.path.join(args.out, "churn_to_p.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    plot(rows, os.path.join(args.out, "churn_to_p.png"))


if __name__ == "__main__":
    main()
