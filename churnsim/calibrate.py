"""Fit the free parameters of the cost model to Petals Table 1.

The no-failure costs (hop time, cache-less prefix cost) are derived exactly from the
p=0 columns in petals.py. What remains free is what a failure costs:

  reconnect_time    time to notice the failure, find a replacement and open a stream.
                    One value for the restart baseline (it tears down and re-opens the
                    whole chain) and one for Algorithm 1 / cache-less (one stage).
  replay_per_token  Algorithm 1 only: rebuilding one stage's KV cache, per cached token.

The strategies share no fitted parameter, so each is fitted on its own cells. The
restart cell the paper reports as DNF (1024 tokens, p=1e-2) is skipped: completing
4096 consecutive hops at p=1e-2 has probability ~1e-18, so it is DNF by construction
and simulating it only burns the time budget. Cells with p*S*T < 0.1 (128 tokens at
p=1e-4: 0.05 expected failures per run) are also skipped: with any feasible number of
repeats they measure the random number generator, not the model.

Calibration runs with no time budget, so no cell is censored, and with many seeds:
a 30-seed fit was found to be biased by ~18% on the 128-token cells (seeds 0..29
happen to draw fewer failures than nominal), which inflated the fitted reconnect.

Run:  python -m churnsim.calibrate [--seeds 200] [--json out/calibration.json]
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from typing import Dict, Optional, Tuple

from .model import Strategy, run_many
from .petals import FAILURE_RATES, SEQ_LENS, STAGES, TABLE1, base_costs_from_p0, config_for

Cells = Dict[Tuple[Strategy, int, float], Optional[float]]


def predict_strategy(strat: Strategy, reconnect: float, replay_per_token: float,
                     seeds: int, base) -> Cells:
    out: Cells = {}
    for T in SEQ_LENS:
        for p in FAILURE_RATES:
            if p == 0.0:
                continue  # exact by construction
            if TABLE1[(strat, T)][p] is None:
                out[(strat, T, p)] = None  # DNF by construction, see module docstring
                continue
            if p * STAGES * T < 0.1:
                continue  # too few failures to measure, see module docstring
            cfg = config_for(strat, T, p, reconnect_time=reconnect,
                             replay_per_token=replay_per_token, base=base,
                             time_budget=1e9)
            out[(strat, T, p)] = run_many(cfg, strat, seeds=seeds).mean_steps_per_s
    return out


def loss(pred: Cells) -> float:
    err = 0.0
    for (strat, T, p), sim in pred.items():
        paper = TABLE1[(strat, T)][p]
        if paper is None:
            continue
        if sim is None:
            err += 9.0
            continue
        err += (math.log(sim) - math.log(paper)) ** 2
    return err


def fit(seeds: int, base, verbose: bool = True) -> Tuple[Dict[str, float], Cells]:
    params: Dict[str, float] = {}
    pred: Cells = {}

    # restart: one parameter
    best = None
    for rc in [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
        P = predict_strategy(Strategy.RESTART, rc, 0.0, seeds, base)
        L = loss(P)
        if verbose:
            print(f"  restart  reconnect={rc:<4} loss={L:.3f}")
        if best is None or L < best[0]:
            best = (L, rc, P)
    params["reconnect_restart"] = best[1]
    pred.update(best[2])

    # replay: two parameters
    best = None
    for rc, rpt in itertools.product([2.0, 3.0, 4.0, 4.5, 5.0, 6.0, 8.0],
                                     [0.0, 0.004, 0.006, 0.008, 0.01, 0.012, 0.016]):
        P = predict_strategy(Strategy.REPLAY, rc, rpt, seeds, base)
        L = loss(P)
        if verbose:
            print(f"  replay   reconnect={rc:<4} replay_per_token={rpt:<6} loss={L:.3f}")
        if best is None or L < best[0]:
            best = (L, rc, rpt, P)
    params["reconnect_replay"] = best[1]
    params["replay_per_token"] = best[2]
    pred.update(best[3])

    # cache-less: one parameter. The paper's row is flat, so expect this to be small;
    # that baseline is a different implementation (Megatron-DeepSpeed) and its
    # per-failure cost evidently is not the same as Petals' reconnect.
    best = None
    for rc in [0.0, 0.25, 0.5, 1.0, 2.0]:
        P = predict_strategy(Strategy.CACHELESS, rc, 0.0, seeds, base)
        L = loss(P)
        if verbose:
            print(f"  cacheless reconnect={rc:<4} loss={L:.3f}")
        if best is None or L < best[0]:
            best = (L, rc, P)
    params["reconnect_cacheless"] = best[1]
    pred.update(best[2])
    return params, pred


def format_table(pred: Cells) -> str:
    lines = ["strategy   T        " + "".join(f"{p:>8.0e}" for p in FAILURE_RATES)]
    for strat in Strategy:
        for T in SEQ_LENS:
            sim_row, pap_row = [], []
            for p in FAILURE_RATES:
                if p == 0.0:
                    v = TABLE1[(strat, T)][0.0]
                elif (strat, T, p) not in pred:
                    sim_row.append("       -"); continue
                else:
                    v = pred[(strat, T, p)]
                sim_row.append(f"{v:8.2f}" if v is not None else "     DNF")
                v = TABLE1[(strat, T)][p]
                pap_row.append(f"{v:8.2f}" if v is not None else "     DNF")
            lines.append(f"{strat.value:<10} {T:<5} sim   " + "".join(sim_row))
            lines.append(f"{'':<10} {'':<5} paper " + "".join(pap_row))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--json", default=None, help="write fitted params + table here")
    args = ap.parse_args()
    base = base_costs_from_p0()
    t0 = time.time()
    params, pred = fit(args.seeds, base)
    print(f"\nfit took {time.time() - t0:.0f}s")
    print("derived base costs:",
          {k.value: {kk: round(v, 6) for kk, v in d.items()} for k, d in base.items()})
    print("fitted:", params)
    print()
    print(format_table(pred))
    if args.json:
        import os
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({"params": params,
                       "base": {k.value: d for k, d in base.items()},
                       "cells": [{"strategy": s.value, "seq_len": T, "p": p,
                                  "sim": v, "paper": TABLE1[(s, T)][p]}
                                 for (s, T, p), v in pred.items()]}, f, indent=2)


if __name__ == "__main__":
    main()
