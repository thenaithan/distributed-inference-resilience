# research-sharding-churnsim

Experiment 1 of the distributed-inference resilience plan: a discrete-event simulation
of pipeline-parallel inference under node churn, calibrated against the fault-tolerance
experiment in the Petals paper (Borzunov et al., arXiv:2312.08361, Table 1 and
Appendix F).

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
- `churnsim/calibrate.py` – grid search for the two free parameters: reconnect cost and
  replay cost per token. `python -m churnsim.calibrate --seeds 30`

## Quick start

```
python3 -c "
from churnsim.petals import config_for
from churnsim.model import Strategy, run_many
cfg = config_for(Strategy.REPLAY, 128, 1e-2, reconnect_time=2.5, replay_per_token=0.0089)
print(run_many(cfg, Strategy.REPLAY, seeds=50).mean_steps_per_s)   # paper: 3.38
"
```

Throughput is reported the way the paper reports it: tokens / mean elapsed time over
repeats. Averaging steps/s per run is biased upward for `restart`, whose completion
time is heavy-tailed.

## Status

- [x] simulator with three strategies and both failure models
- [x] Table 1 encoded, base costs derived
- [ ] calibration run and sim-vs-paper table
- [ ] sweep p from 0 to 5e-2 and plot (Appendix F shape)
- [ ] map `leave_rate` to an effective per-hop p
