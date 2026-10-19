"""Reference data: Petals paper, Table 1 (arXiv:2312.08361, Section 4.1).

Sequential inference speed (steps/second) of BLOOM-7.1B over four pipeline stages
(8, 7, 8, 7 blocks each, one GeForce 1080 Ti per stage, no bandwidth limit), with a
per-send failure probability p. None = did not finish within 1 hour.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from .model import Config, Strategy

FAILURE_RATES = [0.0, 1e-4, 1e-3, 1e-2]
SEQ_LENS = [128, 1024]

# (strategy, seq_len) -> {p: steps/s}
TABLE1: Dict[Tuple[Strategy, int], Dict[float, Optional[float]]] = {
    (Strategy.RESTART, 128):   {0.0: 17.1, 1e-4: 16.7, 1e-3: 12.0, 1e-2: 0.18},
    (Strategy.RESTART, 1024):  {0.0: 15.5, 1e-4: 11.8, 1e-3: 0.48, 1e-2: None},
    (Strategy.CACHELESS, 128): {0.0: 3.44, 1e-4: 3.44, 1e-3: 3.44, 1e-2: 3.44},
    (Strategy.CACHELESS, 1024): {0.0: 0.89, 1e-4: 0.89, 1e-3: 0.89, 1e-2: 0.89},
    (Strategy.REPLAY, 128):    {0.0: 11.4, 1e-4: 11.4, 1e-3: 10.6, 1e-2: 3.38},
    (Strategy.REPLAY, 1024):   {0.0: 10.7, 1e-4: 10.7, 1e-3: 7.76, 1e-2: 2.17},
}

STAGES = 4


def _fit_linear(x0: float, y0: float, x1: float, y1: float) -> Tuple[float, float]:
    slope = (y1 - y0) / (x1 - x0)
    return y0 - slope * x0, slope


def base_costs_from_p0() -> Dict[Strategy, Dict[str, float]]:
    """Derive the no-failure cost parameters for each strategy from the p=0 columns.

    Step time at p=0 is S*(a + b*k) averaged over k in [0, T). Two sequence lengths
    give two equations, so a and b fall out directly. These are not free parameters.
    """
    out: Dict[Strategy, Dict[str, float]] = {}
    for strat in Strategy:
        t128 = 1.0 / TABLE1[(strat, 128)][0.0]
        t1024 = 1.0 / TABLE1[(strat, 1024)][0.0]
        kbar128, kbar1024 = (128 - 1) / 2, (1024 - 1) / 2
        if strat is Strategy.CACHELESS:
            # cost per hop = fixed + (k+1)*per_token  ->  mean over k uses (kbar+1)
            a, b = _fit_linear(kbar128 + 1, t128, kbar1024 + 1, t1024)
            out[strat] = {"cacheless_fixed": a / STAGES, "cacheless_per_token": b / STAGES}
        else:
            a, b = _fit_linear(kbar128, t128, kbar1024, t1024)
            out[strat] = {"hop_time": a / STAGES, "hop_time_per_token": b / STAGES}
    return out


def config_for(strategy: Strategy, seq_len: int, p: float, *,
               reconnect_time: float, replay_per_token: float,
               base: Optional[Dict[Strategy, Dict[str, float]]] = None,
               **extra) -> Config:
    base = base or base_costs_from_p0()
    kw = dict(stages=STAGES, seq_len=seq_len, hop_failure_p=p,
              reconnect_time=reconnect_time, replay_per_token=replay_per_token)
    kw.update(base[strategy])
    kw.update(extra)
    return Config(**kw)
