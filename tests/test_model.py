"""Sanity checks. Run with `pytest` or `python -m tests.test_model`."""

import math

from churnsim.model import Config, Strategy, run_many, simulate
from churnsim.petals import TABLE1, base_costs_from_p0, config_for


def test_p0_matches_table1_exactly():
    # At p=0 the derived base costs must reproduce the paper's p=0 column by construction.
    for strat in Strategy:
        for T in (128, 1024):
            cfg = config_for(strat, T, 0.0, reconnect_time=1.0, replay_per_token=0.01)
            v = run_many(cfg, strat, seeds=1).mean_steps_per_s
            assert math.isclose(v, TABLE1[(strat, T)][0.0], rel_tol=1e-3), (strat, T, v)


def test_restart_1024_p1e2_is_dnf():
    cfg = config_for(Strategy.RESTART, 1024, 1e-2, reconnect_time=3.0, replay_per_token=0.0)
    s = run_many(cfg, Strategy.RESTART, seeds=3)
    assert s.n_finished == 0 and s.mean_steps_per_s is None


def test_replay_degrades_gracefully_and_restart_cliffs():
    base = base_costs_from_p0()
    rp = run_many(config_for(Strategy.REPLAY, 128, 1e-2, reconnect_time=6.0,
                             replay_per_token=0.005, base=base), Strategy.REPLAY, seeds=20)
    rs = run_many(config_for(Strategy.RESTART, 128, 1e-2, reconnect_time=3.0,
                             replay_per_token=0.0, base=base), Strategy.RESTART, seeds=20)
    assert rp.mean_steps_per_s > 2.0
    assert rs.mean_steps_per_s < 0.5


def test_replay_replays_exactly_the_established_prefix():
    cfg = config_for(Strategy.REPLAY, 64, 0.05, reconnect_time=0.1, replay_per_token=0.01)
    st = simulate(cfg, Strategy.REPLAY)
    assert st.finished and st.tokens == 64
    assert st.replays > 0
    # every replay rebuilds at most the tokens generated so far
    assert st.replayed_tokens <= st.replays * 64


def test_churn_mode_waits_when_stage_is_empty():
    # one replica per stage, high leave rate, no joins: the client must eventually stall
    cfg = Config(stages=2, seq_len=5000, replicas=1, leave_rate=0.5, join_rate=0.0,
                 hop_time=0.01, reconnect_time=0.1, replay_per_token=0.001,
                 time_budget=20.0, poll_interval=0.5)
    st = simulate(cfg, Strategy.REPLAY)
    assert not st.finished and st.waits > 0


def test_churn_mode_with_joins_completes():
    cfg = Config(stages=4, seq_len=200, replicas=3, leave_rate=1e-3, join_rate=3e-3,
                 hop_time=0.02, reconnect_time=1.0, replay_per_token=0.005, seed=7)
    st = simulate(cfg, Strategy.REPLAY)
    assert st.finished and st.tokens == 200


def test_summary_censoring_flag():
    cfg = config_for(Strategy.RESTART, 1024, 2e-3, reconnect_time=3.0, replay_per_token=0.0)
    s = run_many(cfg, Strategy.RESTART, seeds=10)
    assert s.censored == (0 < s.n_finished < 10)


if __name__ == "__main__":
    import sys
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok    {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL  {name}: {e}")
    sys.exit(1 if failed else 0)
