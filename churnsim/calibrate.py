"""Fit the two free parameters of the cost model to Petals Table 1.

The no-failure costs (hop time, cache-less prefix cost) are derived exactly from the
p=0 columns in petals.py. What remains free is what a failure costs:

  reconnect_time    time to notice the failure, find a replacement and open a stream
                    (one value for the restart baseline, one for Algorithm 1: the
                    restart baseline tears down and re-opens the whole chain)
  replay_per_token  Algorithm 1 only: rebuilding one stage's KV cache per cached token

Run:  python -m churnsim.calibrate [--seeds 30]
Prints the grid-search winner and the full sim-vs-paper table for it.
"""

from __future__ import annotations

import argparse
import itertools
import math
from typing import Dict, Optional, Tuple

from .model import Strategy, run_many
from .petals import FAILURE_RATES, SEQ_LENS, TABLE1, base_costs_from_p0, config_for


def predict(reconnect_restart: float, reconnect_replay: float, replay_per_token: float,
            seeds: int, base) -> Dict[Tuple[Strategy, int, float], Optional[float]]:
    out = {}
    for strat in Strategy:
        rc = reconnect_restart if strat is Strategy.RESTART else reconnect_replay
        for T in SEQ_LENS:
            for p in FAILURE_RATES:
                if p == 0.0:
                    continue  # exact by construction
                cfg = config_for(strat, T, p, reconnect_time=rc,
                                 replay_per_token=replay_per_token, base=base)
                out[(strat, T, p)] = run_many(cfg, strat, seeds=seeds).mean_steps_per_s
    return out


def loss(pred) -> float:
    err = 0.0
    for (strat, T, p), sim in pred.items():
        paper = TABLE1[(strat, T)][p]
        if paper is None:
            # paper DNF: fine if we DNF too, mild penalty if we finish but slowly
            if sim is not None:
                err += (math.log(sim) - math.log(0.1)) ** 2 if sim > 0.1 else 0.0
            continue
        if sim is None:
            err += 9.0
            continue
        err += (math.log(sim) - math.log(paper)) ** 2
    return err


def print_table(pred, base) -> None:
    hdr = "strategy   T     " + "".join(f"{p:>8.0e}" for p in FAILURE_RATES)
    print(hdr)
    for strat in Strategy:
        for T in SEQ_LENS:
            sim_row = []
            for p in FAILURE_RATES:
                v = TABLE1[(strat, T)][0.0] if p == 0.0 else pred[(strat, T, p)]
                sim_row.append(f"{v:8.2f}" if v is not None else "     DNF")
            pap_row = []
            for p in FAILURE_RATES:
                v = TABLE1[(strat, T)][p]
                pap_row.append(f"{v:8.2f}" if v is not None else "     DNF")
            print(f"{strat.value:<10} {T:<5} sim " + "".join(sim_row))
            print(f"{'':<10} {'':<5} ppr " + "".join(pap_row))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    args = ap.parse_args()
    base = base_costs_from_p0()

    grid_rc_restart = [1.5, 2.0, 2.5, 3.0, 4.0]
    grid_rc_replay = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    grid_rpt = [0.005, 0.01, 0.015, 0.02, 0.03]
    best = None
    for a, b, c in itertools.product(grid_rc_restart, grid_rc_replay, grid_rpt):
        pred = predict(a, b, c, args.seeds, base)
        L = loss(pred)
        if best is None or L < best[0]:
            best = (L, a, b, c, pred)
            print(f"  loss={L:.3f}  reconnect_restart={a} reconnect_replay={b} replay_per_token={c}")
    L, a, b, c, pred = best
    print("\nbest:", dict(loss=round(L, 3), reconnect_restart=a, reconnect_replay=b, replay_per_token=c))
    print("derived base costs:", {k.value: {kk: round(v, 6) for kk, v in d.items()} for k, d in base.items()})
    print()
    print_table(pred, base)


if __name__ == "__main__":
    main()
