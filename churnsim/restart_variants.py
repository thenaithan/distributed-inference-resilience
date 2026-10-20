"""Test three hypotheses for the restart baseline's failure cost against Table 1.

  fixed      cost = R                       (what calibrate.py fits)
  linear     cost = R + a*k                 k = tokens generated before the failure
  quadratic  cost = R + b*k^2

Each is fitted on the five finite restart cells of Table 1 (128: 1e-4, 1e-3, 1e-2;
1024: 1e-4, 1e-3). The 1024/1e-2 cell is DNF for all of them by construction.

Run:  python -m churnsim.restart_variants [--seeds 50]
"""

from __future__ import annotations

import argparse
import itertools
import math
from typing import Dict, Tuple

from .model import Strategy, run_many
from .petals import FAILURE_RATES, SEQ_LENS, TABLE1, base_costs_from_p0, config_for

CELLS = [(T, p) for T in SEQ_LENS for p in FAILURE_RATES
         if p > 0 and TABLE1[(Strategy.RESTART, T)][p] is not None]


def predict(R: float, a: float, b: float, seeds: int, base) -> Dict[Tuple[int, float], float]:
    out = {}
    for T, p in CELLS:
        cfg = config_for(Strategy.RESTART, T, p, reconnect_time=R, replay_per_token=0.0,
                         base=base, restart_cost_per_token=a, restart_cost_per_token_sq=b,
                         time_budget=1e9)
        out[(T, p)] = run_many(cfg, Strategy.RESTART, seeds=seeds).mean_steps_per_s
    return out


def loss(pred) -> float:
    return sum((math.log(v) - math.log(TABLE1[(Strategy.RESTART, T)][p])) ** 2
               for (T, p), v in pred.items() if v is not None) \
        + 9.0 * sum(1 for v in pred.values() if v is None)


def show(name, params, pred):
    print(f"\n{name}: {params}  loss={loss(pred):.3f}")
    for (T, p), v in pred.items():
        paper = TABLE1[(Strategy.RESTART, T)][p]
        vs = "DNF" if v is None else f"{v:5.2f}"
        print(f"  T={T:<5} p={p:<6.0e} sim {vs}  paper {paper:5.2f}  ratio {'' if v is None else f'{v/paper:.2f}'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    args = ap.parse_args()
    base = base_costs_from_p0()
    results = {}

    best = None
    for R in [1.0, 2.0, 3.0, 4.0, 5.0]:
        P = predict(R, 0.0, 0.0, args.seeds, base)
        if best is None or loss(P) < best[0]:
            best = (loss(P), dict(R=R), P)
    results["fixed"] = best

    best = None
    for R, a in itertools.product([0.5, 1.0, 2.0, 3.0], [0.01, 0.02, 0.04, 0.07, 0.1, 0.15]):
        P = predict(R, a, 0.0, args.seeds, base)
        if best is None or loss(P) < best[0]:
            best = (loss(P), dict(R=R, a=a), P)
    results["linear"] = best

    best = None
    for R, b in itertools.product([1.0, 2.0, 3.0], [5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3]):
        P = predict(R, 0.0, b, args.seeds, base)
        if best is None or loss(P) < best[0]:
            best = (loss(P), dict(R=R, b=b), P)
    results["quadratic"] = best

    for name, (L, params, P) in results.items():
        show(name, params, P)


if __name__ == "__main__":
    main()
