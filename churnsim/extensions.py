"""Experiment 2 extensions: the three cheap omissions, tested.

  (a) detection latency   detect_time in {one hop, 30 s, 180 s}: throughput vs node
                          leave rate, and where the recovery cascade starts
  (b) client routing cache known_window in {0, 60 s, forever}: completion probability
                          vs fraction of the discovery layer removed, busy swarm
  (c) compute attack      degrade cut: a fraction of nodes slowed 10x, under three
                          client behaviours: never evict slow nodes (Petals),
                          naive eviction, latency-aware eviction

Run:  python -m churnsim.extensions [--seeds 100] [--out out]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import replace
from typing import List

from .cuts import CUT_TIME, DEADLINE_FACTOR, FRACTIONS, SEQ_LEN, base_config
from .model import Cut, Strategy, run_many

LAMBDAS = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2]
DETECTS = [None, 30.0, 180.0]
WINDOWS = [0.0, 60.0, 1e12]
CLIENTS = {"never evict (Petals)": dict(routing="random", slow_threshold=None),
           "naive evict": dict(routing="random", slow_threshold=3.0),
           "latency-aware evict": dict(routing="min_latency", slow_threshold=3.0)}
DEGRADE_FACTOR = 10.0


def sweep_detect(params, seeds) -> List[dict]:
    rows = []
    nominal = SEQ_LEN / 10.7
    for dt in DETECTS:
        for lam in LAMBDAS:
            cfg = replace(base_config(params, lam, 3 * lam, 3, DEADLINE_FACTOR * nominal), detect_time=dt)
            s = run_many(cfg, Strategy.REPLAY, seeds=seeds)
            C = (dt if dt is not None else cfg.hop_time) + params["reconnect_replay"] + 512 * params["replay_per_token"]
            rows.append(dict(part="detect", detect_time=dt if dt is not None else "hop", leave_rate=lam,
                             steps_per_s=s.mean_steps_per_s, finished=s.n_finished, censored=s.censored,
                             lambda_star=1 / (cfg.stages * C), seeds=seeds))
            v = s.mean_steps_per_s
            print(f"detect={rows[-1]['detect_time']:<5} λ={lam:<6.0e} -> {'DNF' if v is None else f'{v:5.2f}'} "
                  f"({s.n_finished}/{seeds}) λ*={rows[-1]['lambda_star']:.1e}")
    return rows


def sweep_cache(params, seeds) -> List[dict]:
    rows = []
    nominal = SEQ_LEN / 10.7
    for kw in WINDOWS:
        for g in FRACTIONS:
            cfg = replace(base_config(params, 1e-2, 3e-2, 3, DEADLINE_FACTOR * nominal),
                          known_window=kw, cut=Cut(time=CUT_TIME, kind="registry", amount=g))
            s = run_many(cfg, Strategy.REPLAY, seeds=seeds)
            rows.append(dict(part="cache", known_window=kw, amount=g, p_complete=s.n_finished / seeds, seeds=seeds))
            print(f"known_window={kw:<6.0f} g={g:<5} P(complete)={rows[-1]['p_complete']:.2f}")
    return rows


def sweep_degrade(params, seeds) -> List[dict]:
    rows = []
    nominal = SEQ_LEN / 10.7
    for name, kw in CLIENTS.items():
        for f in FRACTIONS:
            cfg = replace(base_config(params, 0.0, 0.0, 3, DEADLINE_FACTOR * nominal),
                          cut=Cut(time=CUT_TIME, kind="degrade", amount=f, factor=DEGRADE_FACTOR), **kw)
            s = run_many(cfg, Strategy.REPLAY, seeds=seeds)
            fin = [r for r in s.runs if r.finished]
            # unfinished runs count at the deadline, as everywhere else in the repo
            t = sum(r.elapsed for r in s.runs) / seeds
            rows.append(dict(part="degrade", client=name, amount=f, p_complete=len(fin) / seeds,
                             time_over_nominal=t / nominal, censored=len(fin) < seeds,
                             evictions=sum(r.slow_evictions for r in s.runs) / seeds, seeds=seeds))
            print(f"{name:<22} f={f:<5} P={rows[-1]['p_complete']:.2f} "
                  f"t/nominal={t / nominal:5.2f}{'+' if len(fin) < seeds else ' '} evictions={rows[-1]['evictions']:.1f}")
    return rows


def plot(det, cache, deg, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c3 = ["#2a78d6", "#eb6834", "#1baf7a"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    ax = axes[0]
    for i, dt in enumerate(["hop", 30.0, 180.0]):
        sel = [r for r in det if r["detect_time"] == dt]
        ax.plot([r["leave_rate"] for r in sel],
                [r["steps_per_s"] if r["steps_per_s"] else float("nan") for r in sel],
                color=c3[i], lw=2, marker="o", ms=4,
                label=f"detect = {'one hop' if dt == 'hop' else f'{dt:.0f} s'}")
        ax.axvline(sel[0]["lambda_star"], color=c3[i], lw=1, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("node leave rate λ (1/s)"); ax.set_ylabel("throughput, steps/s")
    ax.set_title("(a) detection latency; dotted = λ* = 1/(S·C)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    for i, kw in enumerate(WINDOWS):
        sel = [r for r in cache if r["known_window"] == kw]
        ax.plot([r["amount"] for r in sel], [r["p_complete"] for r in sel], color=c3[i], lw=2,
                marker="o", ms=4, label={0.0: "knows current chain only", 60.0: "60 s cache (Petals)",
                                         1e12: "remembers everything"}[kw])
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("fraction of discovery layer removed"); ax.set_ylabel("P(complete), busy swarm λ=1e-2")
    ax.set_title("(b) client routing cache", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    ax = axes[2]
    for i, name in enumerate(CLIENTS):
        sel = [r for r in deg if r["client"] == name]
        xs = [r["amount"] for r in sel]; ys = [r["time_over_nominal"] for r in sel]
        ax.plot(xs, ys, color=c3[i], lw=2, marker="o", ms=4, label=name)
        cx = [x for x, r in zip(xs, sel) if r["censored"]]; cy = [y for y, r in zip(ys, sel) if r["censored"]]
        if cx:
            ax.scatter(cx, cy, s=60, facecolors="white", edgecolors=c3[i], zorder=4, linewidths=1.5)
    ax.scatter([], [], s=60, facecolors="white", edgecolors="#444", linewidths=1.5,
               label="some runs hit the 20× deadline")
    ax.axhline(DEADLINE_FACTOR, color="#bbb", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel(f"fraction of nodes slowed {DEGRADE_FACTOR:.0f}x")
    ax.set_ylabel("mean time / nominal (unfinished at deadline)")
    ax.set_title("(c) compute attack: degraded but present nodes", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    for a in axes:
        a.grid(True, color="#e4e4e0", lw=0.8); a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.suptitle("Experiment 2 extensions, replay strategy, 1024 tokens, 3 replicas/stage",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--out", default="out")
    ap.add_argument("--calibration", default="out/calibration.json")
    args = ap.parse_args()
    with open(args.calibration) as f:
        params = json.load(f)["params"]
    os.makedirs(args.out, exist_ok=True)
    det = sweep_detect(params, args.seeds)
    cache = sweep_cache(params, args.seeds)
    deg = sweep_degrade(params, args.seeds)
    for name, rows in (("ext_detect", det), ("ext_cache", cache), ("ext_degrade", deg)):
        with open(os.path.join(args.out, f"{name}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    plot(det, cache, deg, os.path.join(args.out, "extensions.png"))


if __name__ == "__main__":
    main()
