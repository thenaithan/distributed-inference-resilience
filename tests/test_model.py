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
    # in replay mode k never decreases, so successive replays rebuild a non-decreasing
    # prefix, and the total replayed is the sum of those prefixes
    sizes = st.replay_sizes
    assert all(a <= b for a, b in zip(sizes, sizes[1:]))
    assert sum(sizes) == st.replayed_tokens
    assert max(sizes) < 64


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


def test_summary_censoring_uses_budget_for_unfinished_runs():
    from churnsim.model import Stats, Summary
    cfg = Config(seq_len=100, time_budget=50.0)
    runs = [Stats(tokens=100, finished=True, elapsed=10.0),
            Stats(tokens=40, finished=False, elapsed=50.0)]
    s = Summary(Strategy.REPLAY, cfg, runs)
    assert s.censored
    assert math.isclose(s.mean_steps_per_s, 100 / 30.0)
    s_all_dnf = Summary(Strategy.REPLAY, cfg, [runs[1]])
    assert s_all_dnf.mean_steps_per_s is None and not s_all_dnf.censored


def _cut_cfg(kind, amount, lam=0.0, join=0.0, seed=0):
    from churnsim.model import Cut
    return Config(stages=4, seq_len=256, replicas=3, hop_time=0.0217, reconnect_time=4.5,
                  replay_per_token=0.008, leave_rate=lam, join_rate=join, poll_interval=0.5,
                  time_budget=400.0, cut=Cut(time=5.0, kind=kind, amount=amount), seed=seed)


def test_static_node_cut_matches_closed_form():
    from churnsim.cuts import static_p_complete
    n = 200
    done = sum(simulate(_cut_cfg("nodes", 0.5, seed=i), Strategy.REPLAY).finished for i in range(n))
    assert abs(done / n - static_p_complete("nodes", 0.5, 4, 3)) < 0.1


def test_stage_cut_stops_without_joins_and_heals_with():
    st = simulate(_cut_cfg("stage", 1), Strategy.REPLAY)
    assert not st.finished and st.nodes_cut == 3
    st = simulate(_cut_cfg("stage", 1, lam=1e-3, join=3e-2), Strategy.REPLAY)
    assert st.finished and st.waits > 0


def test_registry_cut_alone_does_not_stop_inflight_request():
    st = simulate(_cut_cfg("registry", 1.0), Strategy.REPLAY)
    assert st.finished and st.nodes_cut == 12 and st.waits == 0


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
