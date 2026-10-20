"""Experiment 2b: the two structural omissions.

  (a) Discovery held on the server nodes (dht_k = 3). A node cut now also destroys the
      entries whose holders all died, until nodes republish. Sweep the node-cut fraction
      under static and busy swarms with republish_interval in {60 s, 600 s, never}, and
      compare with the oracle registry of Experiment 2.
  (b) Concurrent clients. M clients share the swarm; a node used by L clients runs
      (1 + alpha*(L-1)) times slower. Run the Experiment 2 cuts with M in {1, 5, 20} and
      alpha in {0.05 (Petals Table 3), 0.3 (weak batching)}; report per-client outcomes.

Run:  python -m churnsim.exp2b [--seeds 50] [--out out]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import replace
from typing import List

from .cuts import CUT_TIME, DEADLINE_FACTOR, FRACTIONS, SEQ_LEN, base_config
from .model import Cut, Strategy, simulate_all

NOMINAL = SEQ_LEN / 10.7
DEADLINE = DEADLINE_FACTOR * NOMINAL
REPUBLISH = {"60 s": 60.0, "600 s": 600.0, "never": 1e12}


def _agg(all_stats, seeds):
    flat = [st for run in all_stats for st in run]
    fin = [st for st in flat if st.finished]
    return dict(p_complete=len(fin) / len(flat),
                time_over_nominal=sum(st.elapsed for st in flat) / len(flat) / NOMINAL,
                censored=len(fin) < len(flat), n=len(flat))


def sweep_dht(params, seeds) -> List[dict]:
    rows = []
    for scen, (lam, join) in {"static": (0.0, 0.0), "busy": (1e-2, 3e-2)}.items():
        for label, rp in [("oracle", None)] + list(REPUBLISH.items()):
            for f in FRACTIONS:
                cfg = base_config(params, lam, join, 3, DEADLINE)
                cfg = replace(cfg, cut=Cut(time=CUT_TIME, kind="nodes", amount=f),
                              dht_k=0 if rp is None else 3,
                              republish_interval=rp if rp is not None else 60.0)
                runs = [simulate_all(replace(cfg, seed=i), Strategy.REPLAY) for i in range(seeds)]
                a = _agg(runs, seeds)
                rows.append(dict(part="dht", scenario=scen, registry=label, amount=f, **a))
                print(f"dht {scen:<7} {label:<7} f={f:<5} P={a['p_complete']:.2f} "
                      f"t/nom={a['time_over_nominal']:5.2f}{'+' if a['censored'] else ' '}")
    return rows


CUTS_B = [("nodes", 0.5), ("nodes", 1.0), ("stage", 1), ("registry", 0.8), ("registry", 1.0)]


def sweep_clients(params, seeds) -> List[dict]:
    rows = []
    for alpha in (0.05, 0.3):
        for M in (1, 5, 20):
            for kind, amt in CUTS_B:
                cfg = base_config(params, 1e-2, 3e-2, 3, DEADLINE)
                cfg = replace(cfg, clients=M, load_alpha=alpha,
                              cut=Cut(time=CUT_TIME, kind=kind, amount=amt))
                n_seeds = max(seeds // (M if M > 1 else 1), 10)
                runs = [simulate_all(replace(cfg, seed=i), Strategy.REPLAY) for i in range(n_seeds)]
                a = _agg(runs, n_seeds)
                rows.append(dict(part="clients", alpha=alpha, clients=M, kind=kind, amount=amt, **a))
                print(f"clients α={alpha} M={M:<3} {kind:<9} {amt:<4} P={a['p_complete']:.2f} "
                      f"t/nom={a['time_over_nominal']:5.2f}{'+' if a['censored'] else ' '} (n={a['n']})")
    return rows


def plot(dht, cl, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c4 = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, scen, ylab in ((axes[0][0], "static", "P(complete)"), (axes[0][1], "busy", "mean time / nominal")):
        for i, label in enumerate(["oracle"] + list(REPUBLISH)):
            sel = [r for r in dht if r["scenario"] == scen and r["registry"] == label]
            xs = [r["amount"] for r in sel]
            ys = [r["p_complete"] if scen == "static" else r["time_over_nominal"] for r in sel]
            ax.plot(xs, ys, color=c4[i], lw=2, marker="o", ms=4,
                    label="oracle registry" if label == "oracle" else f"DHT on servers, republish {label}")
        ax.set_xlabel("fraction of nodes removed"); ax.set_ylabel(ylab)
        if scen == "busy":
            ax.set_yscale("log"); ax.set_ylim(1, 25); ax.axhline(DEADLINE_FACTOR, color="#bbb", lw=1)
        else:
            ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"(a) node cut, {scen} swarm, discovery held on the nodes", loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=8)

    labels = [f"{k} {a}" if k != "stage" else "stage" for k, a in CUTS_B]
    x0 = list(range(len(CUTS_B)))
    w = 0.13
    for ax, key, ylab in ((axes[1][0], "p_complete", "P(complete), per client"),
                          (axes[1][1], "time_over_nominal", "mean time per client / nominal")):
        i = 0
        for alpha in (0.05, 0.3):
            for M in (1, 5, 20):
                sel = [next(r for r in cl if r["alpha"] == alpha and r["clients"] == M
                            and r["kind"] == k and r["amount"] == a) for k, a in CUTS_B]
                ax.bar([x + (i - 2.5) * w for x in x0], [r[key] for r in sel], width=w,
                       color=c4[[1, 5, 20].index(M)], alpha=1.0 if alpha == 0.05 else 0.45,
                       label=f"M={M}, α={alpha}")
                i += 1
        ax.set_xticks(x0); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(ylab)
        if key == "time_over_nominal":
            ax.set_yscale("log"); ax.set_ylim(1, 25); ax.axhline(DEADLINE_FACTOR, color="#bbb", lw=1)
        else:
            ax.set_ylim(0, 1.25)
        ax.set_title("(b) concurrent clients, busy swarm", loc="left", fontsize=10)
    axes[1][0].legend(frameon=False, fontsize=7, ncol=3, loc="upper right")
    for row in axes:
        for a in row:
            a.grid(True, color="#e4e4e0", lw=0.8); a.set_axisbelow(True)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
    fig.suptitle("Experiment 2b: discovery on the server nodes; concurrent clients  (replay, 1024 tokens, 3 replicas)",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--out", default="out")
    ap.add_argument("--calibration", default="out/calibration.json")
    args = ap.parse_args()
    with open(args.calibration) as f:
        params = json.load(f)["params"]
    os.makedirs(args.out, exist_ok=True)
    dht = sweep_dht(params, args.seeds)
    cl = sweep_clients(params, args.seeds)
    for name, rows in (("exp2b_dht", dht), ("exp2b_clients", cl)):
        with open(os.path.join(args.out, f"{name}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    plot(dht, cl, os.path.join(args.out, "exp2b.png"))


if __name__ == "__main__":
    main()
