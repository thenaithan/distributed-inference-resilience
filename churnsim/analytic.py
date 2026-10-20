"""Closed-form model of the three strategies, checked against the simulation sweep.

See NOTES.md for the derivations. Notation: S stages, T tokens, h hop time (with a
per-token term), p per-hop failure probability, R reconnect cost, r replay cost per
token.

Run:  python -m churnsim.analytic          (needs out/calibration.json and out/sweep.csv)
"""

from __future__ import annotations

import csv
import json
import math
from typing import Optional

from .model import Strategy
from .petals import STAGES, base_costs_from_p0


def step_time_replay(T: int, p: float, h: float, h_tok: float, R: float, r: float) -> float:
    """Expected seconds per token. Linear in p: S*h + S*p*(R + k*r), averaged over k."""
    kbar = (T - 1) / 2
    hop = h + h_tok * kbar
    # a replay send can itself fail; the geometric number of tries adds a 1/(1-p) factor
    recovery = (R + kbar * r) / (1 - p)
    return STAGES * (hop + p * recovery)


def time_restart(T: int, p: float, h: float, h_tok: float, R: float,
                 c_lin: float = 0.0, c_quad: float = 0.0) -> Optional[float]:
    """Expected total seconds to finish T tokens with restart-from-scratch.

    Attempts are i.i.d.; one succeeds with q = (1-p)^(S*T). A failed attempt dies at
    hop H ~ Geometric(p) truncated to S*T, costing its hops plus a restart cost
    R + c_lin*k + c_quad*k^2 where k = H/S tokens of progress. Returns None when q is
    so small the expected time exceeds a day (the paper's DNF).
    """
    n = STAGES * T
    q = (1 - p) ** n
    if q < 1e-9:
        return None
    hop = h + h_tok * (T - 1) / 2
    # E[hops | failed] for a truncated geometric
    if p == 0:
        return n * hop
    e_hops_fail = (1 / p) - n * q / (1 - q) if q < 1 else 0.0
    k = e_hops_fail / STAGES
    # E[k^2] for truncated geometric, approximated as 2*k^2 (exponential moment ratio)
    cost_fail = e_hops_fail * hop + R + c_lin * k + c_quad * 2 * k * k
    cost_ok = n * hop
    expected_attempts = 1 / q
    return (expected_attempts - 1) * cost_fail + cost_ok


def step_time_cacheless(T: int, p: float, fixed: float, per_tok: float, R: float) -> float:
    kbar = (T - 1) / 2
    hop = fixed + (kbar + 1) * per_tok
    return STAGES * (hop + p * R) / (1 - p)


def crossover_p(T: int) -> float:
    """Where restart and replay cross: about one failure per generation, p*S*T = 1."""
    return 1.0 / (STAGES * T)


def churn_threshold(C: float) -> float:
    """Leave rate above which recovery cascades: S*lambda*C = 1."""
    return 1.0 / (STAGES * C)


def main() -> None:
    with open("out/calibration.json") as f:
        params = json.load(f)["params"]
    base = base_costs_from_p0()
    rows = list(csv.DictReader(open("out/sweep.csv")))

    print(f"{'strategy':<10}{'T':>6}{'p':>9}{'sim':>9}{'formula':>9}{'ratio':>8}")
    for row in rows:
        strat = Strategy(row["strategy"]); T = int(row["seq_len"]); p = float(row["p"])
        sim = float(row["steps_per_s"]) if row["steps_per_s"] not in ("", "None") else None
        if strat is Strategy.REPLAY:
            b = base[strat]
            f = 1 / step_time_replay(T, p, b["hop_time"], b["hop_time_per_token"],
                                     params["reconnect_replay"], params["replay_per_token"])
        elif strat is Strategy.RESTART:
            b = base[strat]
            t = time_restart(T, p, b["hop_time"], b["hop_time_per_token"], params["reconnect_restart"])
            f = None if t is None else T / t
        else:
            b = base[strat]
            f = 1 / step_time_cacheless(T, p, b["cacheless_fixed"], b["cacheless_per_token"],
                                        params["reconnect_cacheless"])
        fs = f"{f:9.2f}" if f is not None else "      DNF"
        ss = f"{sim:9.2f}" if sim is not None else "      DNF"
        ratio = f"{sim / f:8.2f}" if (sim and f) else "        "
        print(f"{strat.value:<10}{T:>6}{p:>9.0e}{ss}{fs}{ratio}")

    print()
    for T in (128, 1024):
        print(f"restart/replay crossover, T={T}: p ≈ {crossover_p(T):.1e}")
    C = params["reconnect_replay"] + 512 * params["replay_per_token"]
    print(f"churn cascade threshold (replay, 1024 tokens, C={C:.1f} s): λ* ≈ {churn_threshold(C):.3f} /s")
    print(f"replay knee (p ≈ h/C): {base[Strategy.REPLAY]['hop_time'] / C:.1e}")


if __name__ == "__main__":
    main()
