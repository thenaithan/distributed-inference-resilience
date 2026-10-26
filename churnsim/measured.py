"""What the measured recovery constants imply, using the model's closed forms.

The constants come from Experiment 3 (distributed-inference-recovery, RESULTS.md):
one node failure's cost on exo, vLLM and llama.cpp's RPC backend, on one CPU host.
All three are the `restart` strategy: nothing of an interrupted request survives.

    python -m churnsim.measured

Two questions are answered:

1. If each system had Petals-style replay with the same detection and reconnect cost,
   where would its recovery cascade start?  lambda* = 1/(S*C)  (NOTES.md §2)
2. As they are (restart), how reliable must a device be for a long request to finish?
   A request of duration D over N nodes completes without a failure with probability
   exp(-N*lambda*D); each failure costs the whole request again.
"""

from __future__ import annotations

import math

# (system, failure, S or N nodes in the measured setup, detect s, C s)  -- RESULTS.md §1
MEASURED = [
    ("exo bc412e2",     "worker crash/hang/partition", 3, 15.0,  275.0),
    ("exo bc412e2",     "master loss",                 3, 13.2,  300.0),
    ("vLLM 0.32 PP=2",  "worker crash",                2, 1.5,   67.0),
    ("vLLM 0.32 PP=2",  "worker hang",                 2, 300.0, 373.0),
    ("llama.cpp RPC",   "rpc host crash",              2, 0.5,   14.0),
    ("llama.cpp RPC",   "rpc host hang / partition",   2, math.inf, math.inf),
]


def main() -> None:
    print("1. cascade threshold if the system had replay recovery, lambda* = 1/(S*C)")
    for sys_, fail, S, det, C in MEASURED:
        if math.isinf(C):
            print(f"  {sys_:<16} {fail:<28} C = never detected: no threshold, every hang is permanent")
            continue
        lam = 1 / (S * C)
        print(f"  {sys_:<16} {fail:<28} C={C:6.0f} s  lambda*={lam:.1e}/s  "
              f"(mean node lifetime {S*C/60:5.1f} min)")

    print("\n2. restart strategy: a request of D hours over N nodes needs no failure at all")
    for N, D_h, label in [(3, 1, "3 nodes, 1 h request"), (15, 30, "15 boxes, 30 h window (Frontier Red Team article)")]:
        D = D_h * 3600
        print(f"  {label}:")
        for P in (0.5, 0.9, 0.99):
            lam = -math.log(P) / (N * D)
            print(f"    P(complete) = {P:4.2f} needs mean device lifetime >= {1/lam/86400:8.1f} days")
        for life_d in (1, 7, 30, 180):
            lam = 1 / (life_d * 86400)
            q = math.exp(-N * lam * D)
            print(f"    mean lifetime {life_d:4d} d: P(complete) = {q:.3f}, "
                  f"expected wall time {'inf' if q < 1e-6 else f'{D/q/3600:.0f} h'}")

    print("\n3. with replay recovery instead (Petals mechanism), the same window costs")
    N, D = 15, 30 * 3600
    for life_d in (1, 7, 30):
        lam = 1 / (life_d * 86400)
        for C in (275.0, 14.0):
            extra = N * lam * D * C
            print(f"    lifetime {life_d:3d} d, C = {C:4.0f} s: expected {N*lam*D:5.1f} failures, "
                  f"+{extra/3600:5.2f} h of recovery on a 30 h window"
                  f"{'  (cascade: S*lambda*C = %.2f)' % (N*lam*C) if N*lam*C > 0.3 else ''}")


if __name__ == "__main__":
    main()
