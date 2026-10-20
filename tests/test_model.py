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


def test_detect_time_replaces_hop_cost_for_dead_nodes():
    # every failure under churn is a send to a node that has left; with detect_time set
    # each one must cost at least detect_time
    cfg = Config(stages=2, seq_len=300, replicas=3, hop_time=0.01, reconnect_time=0.1,
                 replay_per_token=0.0001, leave_rate=0.05, join_rate=0.15,
                 detect_time=7.0, time_budget=1000.0, seed=1)
    st = simulate(cfg, Strategy.REPLAY)
    assert st.finished and st.failures >= 1
    assert st.elapsed >= 7.0 * st.failures


def test_known_window_lets_client_route_without_registry():
    # registry gone, chain node leaves; with a cache the client still finds the others
    from churnsim.model import Cut
    base = dict(stages=1, seq_len=50, replicas=3, hop_time=0.01, reconnect_time=0.1,
                replay_per_token=0.001, leave_rate=0.0, time_budget=30.0,
                cut=Cut(time=0.05, kind="registry", amount=1.0))
    # force the chain node to die: give it a lifetime via leave_rate on a single node is
    # awkward, so instead remove it with a second mechanism: hop failure p>0 reroutes
    # (cut is not None so a p-failure picks a new node rather than reconnecting)
    st0 = simulate(Config(hop_failure_p=0.2, known_window=0.0, **base), Strategy.REPLAY)
    st1 = simulate(Config(hop_failure_p=0.2, known_window=1e9, **base), Strategy.REPLAY)
    assert not st0.finished and st0.waits > 0      # nothing discoverable, stalls
    assert st1.finished and st1.waits == 0         # remembered replicas still usable


def test_degrade_cut_slows_without_failing_and_latency_aware_client_evicts():
    from churnsim.model import Cut
    base = dict(stages=2, seq_len=100, replicas=3, hop_time=0.01, reconnect_time=0.05,
                replay_per_token=0.0001, time_budget=100.0, seed=3,
                cut=Cut(time=0.1, kind="degrade", amount=1.0, factor=10.0))
    plain = simulate(Config(**base), Strategy.REPLAY)
    assert plain.finished and plain.failures == 0 and plain.slow_evictions == 0
    assert plain.elapsed > 5 * (100 * 2 * 0.01)   # at least ~10x slower after the cut
    # all nodes slowed: a latency-aware client has nowhere better to go, so no evictions
    aware = simulate(Config(routing="min_latency", slow_threshold=3.0, **base), Strategy.REPLAY)
    assert aware.slow_evictions == 0
    # half slowed: it evicts at most a few times and ends up on fast nodes
    base["cut"] = Cut(time=0.1, kind="degrade", amount=0.5, factor=10.0)
    aware = simulate(Config(routing="min_latency", slow_threshold=3.0, **base), Strategy.REPLAY)
    assert aware.finished and 0 <= aware.slow_evictions <= 4


def test_dht_entries_die_with_their_holders_and_republish_restores_them():
    from churnsim.model import Registry
    import random
    reg = Registry(dht_k=2)
    rng = random.Random(0)
    nodes = [reg.add(0, rng) for _ in range(4)]
    for n in nodes:
        reg.store(n, rng)
    target = nodes[0]
    for h in reg.holders[target.nid]:
        reg.remove(h)
    assert not reg.entry_alive(target) and target not in reg.discoverable(0)
    reg.store(target, rng)                      # what the republisher does
    assert reg.entry_alive(target) and target in reg.discoverable(0)


def test_load_nets_to_zero_after_all_clients_finish():
    from churnsim.model import Cut, Registry, simulate_all, _client  # noqa: F401
    import churnsim.model as m
    seen = {}
    orig = m.Registry.__init__
    def spy(self, dht_k=0):
        orig(self, dht_k); seen["reg"] = self
    m.Registry.__init__ = spy
    try:
        cfg = Config(stages=3, seq_len=200, replicas=2, hop_time=0.01, reconnect_time=0.1,
                     replay_per_token=0.0001, clients=6, load_alpha=0.2, leave_rate=0.02,
                     join_rate=0.04, detect_time=0.5, time_budget=2000.0,
                     cut=Cut(time=1.0, kind="nodes", amount=0.5), seed=5)
        sts = simulate_all(cfg, Strategy.REPLAY)
    finally:
        m.Registry.__init__ = orig
    assert all(st.finished for st in sts)
    assert all(n.load == 0 for n in seen["reg"].all_nodes())


def test_concurrent_clients_share_load_and_all_finish():
    from churnsim.model import simulate_all
    cfg = Config(stages=2, seq_len=100, replicas=1, hop_time=0.01, reconnect_time=0.1,
                 replay_per_token=0.0001, clients=4, load_alpha=0.5, time_budget=100.0)
    sts = simulate_all(cfg, Strategy.REPLAY)
    assert len(sts) == 4 and all(st.finished for st in sts)
    # one replica per stage, four clients: every hop runs at 1 + 0.5*3 = 2.5x
    single = simulate(Config(**{**cfg.__dict__, "clients": 1}), Strategy.REPLAY).elapsed
    assert all(2.2 * single < st.elapsed < 2.8 * single for st in sts)


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
