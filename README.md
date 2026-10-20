# distributed-inference-resilience

Experiments on how a large model running split across many machines copes with nodes
failing and leaving, studied from the operator's and defender's side. The open
questions come from the essay *Anatomy of an Asymmetric Persistence Mechanism*
(naithanweigh.com/writing/asymmetric-persistence/).

## Experiment 1: churn simulation (`churnsim/`)

A discrete-event simulation of pipeline-parallel inference under node churn,
calibrated against the fault-tolerance experiment in the Petals paper (Borzunov et
al., arXiv:2312.08361, Table 1 and Appendix F).

No real weights. Tokens move hop by hop through a pipeline of stages; hops cost time and
can fail; three recovery strategies are compared:

| strategy    | on failure                                                            |
|-------------|-----------------------------------------------------------------------|
| `restart`   | caches lost, restart the whole generation from token 0                 |
| `cacheless` | no server cache; every step re-sends the prefix; redo only this step  |
| `replay`    | Petals Algorithm 1: reroute to a replacement, replay the client-side cache to rebuild its KV state, continue |

**Done when** the curve of throughput vs per-hop failure probability reproduces the
Petals shape: graceful degradation for `replay`, a cliff for `restart`, flat-but-slow
for `cacheless`, matching the paper's numbers around p = 1e-2.

## Layout

- `churnsim/model.py` – the simulator. Plain heap-based event loop (no SimPy),
  `Registry` for discovery, `Config` for the cost model, two failure knobs:
  - `hop_failure_p`: every send fails with probability p (what Table 1 varies)
  - `leave_rate` / `join_rate`: nodes leave and join as Poisson processes; a hop fails
    when its node has left; the client waits if a stage has no live replica. This is
    the version Experiment 2 (minimum simultaneous cut) builds on.
- `churnsim/petals.py` – Table 1 data and the derivation of the no-failure costs from
  the p = 0 columns (so hop time is not a free parameter).
- `churnsim/calibrate.py` – grid search for the free parameters: reconnect cost and
  replay cost per token.
- `churnsim/sweep.py` – throughput vs p for all strategies and both sequence lengths.
- `churnsim/churn.py` – the bridge from node leave rate λ to an effective per-hop p.
- `churnsim/analytic.py` – closed-form model of all three strategies, checked against the sweep.
- `churnsim/restart_variants.py` – tests fixed / linear / quadratic restart-cost hypotheses.
- `NOTES.md` – derivations, the churn threshold λ* = 1/(S·C), discrepancies, reading list.
- `tests/` – sanity checks; `python -m tests.test_model` (or `pytest`).

## Quick start

```
python3 -c "
from churnsim.petals import config_for
from churnsim.model import Strategy, run_many
cfg = config_for(Strategy.REPLAY, 128, 1e-2, reconnect_time=4.5, replay_per_token=0.008)
print(run_many(cfg, Strategy.REPLAY, seeds=200).mean_steps_per_s)   # paper: 3.38
"
```

Throughput is reported the way the paper reports it: tokens / mean elapsed time over
repeats. Averaging steps/s per run is biased upward for `restart`, whose completion
time is heavy-tailed.

## Reproducing

```
pip install -r requirements.txt
python -m churnsim.calibrate --seeds 200 --json out/calibration.json  # ~5 min
python -m churnsim.sweep --seeds 100                                   # ~2 min
python -m churnsim.churn --seeds 50                                    # ~1 min
python -m churnsim.analytic                                            # closed forms vs sweep
python -m tests.test_model
```

`out/calibration.json` holds the fitted parameters and the sim-vs-paper cells;
`out/sweep.csv` and `out/throughput_vs_failure.png` are the sweep;
`out/churn_to_p.csv` and `out/churn_to_p.png` are the churn bridge.

## Results (2026-10-19)

Base costs from the p = 0 columns (not fitted): restart hop 14.4 ms, Algorithm 1 hop
21.7 ms (the paper notes its implementation is less mature), both growing ~3 µs per
cached token; cache-less 42.7 ms + 0.465 ms per prefix token per hop.

Fitted failure costs (200 seeds per cell, no time budget): restart reconnect 3.0 s
(whole chain re-opened), replay reconnect 4.5 s plus 8 ms per replayed token for one
stage, cache-less ~0. Cells with fewer than 0.1 expected failures per run (128 tokens
at p = 1e-4) are excluded from the fit and shown as "-".

| strategy  | T    |      | p=0   | 1e-4  | 1e-3  | 1e-2  |
|-----------|------|------|-------|-------|-------|-------|
| restart   | 128  | sim  | 17.10 | -     | 10.23 | 0.17  |
|           |      | paper| 17.10 | 16.70 | 12.00 | 0.18  |
| restart   | 1024 | sim  | 15.50 | 12.08 | 0.91  | DNF   |
|           |      | paper| 15.50 | 11.80 | 0.48  | DNF   |
| cacheless | 128  | sim  | 3.44  | -     | 3.44  | 3.41  |
|           |      | paper| 3.44  | 3.44  | 3.44  | 3.44  |
| cacheless | 1024 | sim  | 0.89  | 0.89  | 0.89  | 0.88  |
|           |      | paper| 0.89  | 0.89  | 0.89  | 0.89  |
| replay    | 128  | sim  | 11.40 | -     | 9.35  | 3.48  |
|           |      | paper| 11.40 | 11.40 | 10.60 | 3.38  |
| replay    | 1024 | sim  | 10.70 | 10.31 | 7.76  | 2.24  |
|           |      | paper| 10.70 | 10.70 | 7.76  | 2.17  |

The shape is reproduced: graceful degradation for replay, a cliff for restart between
1e-3 and 1e-2, cache-less flat. Points on the sweep plot where some runs hit the
1-hour budget are censored at the budget and marked as upper bounds.

![throughput vs per-hop failure probability](out/throughput_vs_failure.png)

### From node churn to p

Petals varies p directly; the plan (and Experiment 2) talks about nodes leaving at a
rate λ. The two are the same experiment in different units. With 3 replicas per stage,
joins balancing leaves, and the replay strategy at 1024 tokens:

![churn to effective p](out/churn_to_p.png)

- Right panel: throughput under node churn (blue) matches the p-mode curve evaluated at
  the failure rate the client actually observed (grey) to within ~20% for λ from 1e-4
  to 2e-2. So anything measured against p transfers to λ and back. (The agreement is
  partly by construction: both modes charge the same reconnect cost. What it shows is
  that failure clustering and empty-stage waits, which only churn mode has, cost at
  most ~20% in this range.)
- Left panel: the obvious estimate p ≈ 1 − exp(−λ·τ_step) with τ_step the no-failure
  step time (dotted) undercounts by 2x and more once failures are frequent. The reason
  is that a reconnect on one stage leaves the other three stages' nodes exposed for
  that whole time too, so failures cluster. Using the realised mean step time,
  recoveries included (dashed), tracks the observed p up to λ ≈ 1e-2.
- Above λ ≈ 2e-2 throughput collapses. This is not mostly waiting for empty stages
  (waits are ~6% of elapsed time there, consistent with P(stage empty) = e^(−3)); it is
  a recovery cascade: each recovery takes C ≈ 9 s, during which the other three stages'
  nodes are exposed, and once S·λ·C exceeds 1 each failure spawns more than one further
  failure on average. NOTES.md §2 derives the threshold λ* = 1/(S·C) ≈ 0.028 /s. That
  regime is the one Experiment 2 is about: it is the "every node holding one stage" cut
  happening stochastically instead of deliberately.

### Known discrepancies, and notes to ourselves

- **restart, 1024 tokens, p = 1e-3: sim 0.91 vs paper 0.48.** The closed form agrees
  with the sim (0.90), so this is a mismatch between the paper and the model, not a
  simulation artefact. A restart cost growing with progress (linear or quadratic in
  tokens generated) was tested and does not fit all five restart cells; see NOTES.md §3.
  Left open; the paper's benchmark code for this table is not public.
- **replay at 128 tokens, p = 1e-3: sim 9.35 vs paper 10.60.** The paper's 128-token
  recoveries look cheaper than its 1024-token ones in a way a fixed reconnect + linear
  replay doesn't capture; a fit on the 1024-token cells alone would want a larger R.
  Also the paper's cells at p = 1e-4 are statistically meaningless (0.05 to 0.4
  expected failures per run) and were excluded from the fit.
- At 1024 tokens the two parts of a recovery are about equal: 4.5 s fixed (detect,
  DHT lookup, reroute, connect) and ~4 s replaying a 512-token prefix on one stage. At
  128 tokens the fixed part dominates. For Experiment 2 both are levers, and under real
  churn the fixed part grows by the detection timeout (NOTES.md §4).

## Experiment 2: how much do you have to remove? (`churnsim/cuts.py`)

The same simulation read as a defender. A cut is applied 10 s into a 1024-token
generation and we measure whether the request completes within 20× its no-failure
time. Three cut types (a random fraction of nodes; every replica of one stage; a
fraction of the discovery layer) under three swarm conditions (static; churn λ = 1e-3
with joins balancing leaves; churn λ = 1e-2).

```
python -m churnsim.cuts --seeds 100                 # out/cuts.csv, out/cuts.png
python -m churnsim.cuts --seeds 100 --replicas 10   # out/cuts_n10.*
```

![simultaneous cuts](out/cuts.png)

- **Static swarm:** removal of a random fraction f of nodes follows (1 − fⁿ)^S exactly
  (dashed), so with 3 replicas the request survives f = 0.3 nine times in ten and
  f = 0.8 almost never; the targeted cut (all n replicas of one stage) stops it with n
  nodes; and removing the entire discovery layer stops nothing, because a connected
  chain does not need discovery.
- **Churning swarm:** node cuts heal. Even removing every node leaves 93% (healthy) or
  100% (busy) of requests completing, after a wait of about H_S / join_rate. The
  registry cut becomes the effective one: with all of discovery gone, completion
  requires the S chain nodes to outlive the rest of the generation,
  P = exp(−S·λ·t_remaining), which is 0.71 (healthy) and 0.03 (busy) in closed form
  against 0.63 and 0.07 measured, and is independent of the replica count. In the busy
  swarm removing 80% of the discovery layer stops more requests than removing 100% of
  the nodes.

NOTES.md §5 has the derivations, the n = 10 comparison, and what is not modelled
(client routing caches, DHT-node correlation, detection timeouts, concurrent clients).

## Status

- [x] simulator with three strategies and both failure models
- [x] Table 1 encoded, base costs derived
- [x] calibration run and sim-vs-paper table
- [x] sweep p from 0 to 5e-2 and plot (Appendix F shape)
- [x] map `leave_rate` / `join_rate` to an effective per-hop p (the bridge to Experiment 2)
- [x] tests
- [x] restart cost proportional to tokens generated before failure: tested linear and
      quadratic, neither fits all five cells (NOTES.md §3); left as fixed R, still open
- [ ] drop or heavily re-seed the p·S·T ≪ 1 cells in the calibration loss
- [ ] heavy-tailed (Weibull) node lifetimes in churn mode
- [ ] `detect_time` and `registry_lag` in `Config` (NOTES.md §4); exponential retry backoff
- [x] Experiment 2: cut sweep (nodes / stage / registry × static / healthy / busy)
- [ ] Experiment 2 follow-ups: client-side routing cache, DHT entries tied to nodes, concurrent clients
