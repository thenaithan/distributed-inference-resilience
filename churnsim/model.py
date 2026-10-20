"""Core simulation.

A single client generates `seq_len` tokens autoregressively through a pipeline of
`stages` stages. Every token step sends activations hop by hop through the stages in
order. Each hop costs time, and each hop can fail. What happens on failure depends on
the recovery strategy, which is the whole point of the experiment:

  RESTART   "caching with restarts": servers keep KV caches; on any failure the whole
            generation restarts from token 0 (the failed server's cache is gone).
  CACHELESS no server-side cache; every step re-sends and recomputes the whole prefix;
            on failure only the current step is redone.
  REPLAY    Petals Algorithm 1: the client keeps what it sent to each stage; on failure
            it finds a replacement for that stage, replays the cached inputs so the
            replacement rebuilds its KV cache, then continues the current step.

Two failure models are supported and can be combined:

  hop_failure_p   Petals-style. Every send fails independently with probability p.
                  The stage is "reset" (its cache for this session is lost) but the
                  node stays in the registry. This is what Table 1 of the paper varies.
  leave_rate      Node churn. (Combining both knobs is allowed but note that a
                  p-failure then reroutes to a different replica, whereas in pure
                  p-mode it reconnects to the same, reset node.) Each node's lifetime is Exp(leave_rate); a leaving node is
                  removed from the registry and new nodes arrive per stage as a Poisson
                  process at join_rate. A hop fails when its node has left since the
                  client last talked to it. If a stage has no live replica the client
                  waits, polling the registry. This is the model from Nat's plan, and
                  the one Experiment 2 (minimum cut) builds on.

The event loop is a plain heap scheduler with generator-based processes (yield a
delay in seconds). No SimPy dependency; the core is standard library only.
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Callable, Dict, Generator, List, Optional


class Strategy(str, Enum):
    RESTART = "restart"
    CACHELESS = "cacheless"
    REPLAY = "replay"


@dataclass
class Config:
    # --- topology / workload -------------------------------------------------
    stages: int = 4
    seq_len: int = 128
    replicas: int = 3              # initial live nodes per stage

    # --- failure model ---------------------------------------------------------
    hop_failure_p: float = 0.0     # per-send failure probability (Petals Table 1 knob)
    leave_rate: float = 0.0        # per-node leave rate, 1/s (node churn knob)
    join_rate: float = 0.0         # per-stage node arrival rate, 1/s
    poll_interval: float = 1.0     # how often a client re-checks an empty stage

    # --- cost model, all in seconds -------------------------------------------
    # Single-token hop through one stage with a warm cache. Grows slightly with the
    # number of cached tokens (attention over the prefix).
    hop_time: float = 0.0146
    hop_time_per_token: float = 3.3e-6
    # Cost of discovering a replacement / re-establishing a stream after a failure.
    # Paid by every strategy on every failure.
    reconnect_time: float = 4.5
    # RESTART only: extra cost of a restart that depends on how far the failed attempt
    # got (k = tokens generated). Hypotheses for the paper's 1024/1e-3 cell, see NOTES.md.
    restart_cost_per_token: float = 0.0        # linear in k
    restart_cost_per_token_sq: float = 0.0     # quadratic in k
    # REPLAY only: per-token cost for a replacement stage to rebuild its KV cache from
    # the client-side cache (one batched forward over the prefix, for that stage only).
    replay_per_token: float = 0.008
    # CACHELESS only: per-hop cost of sending and recomputing the whole prefix.
    cacheless_fixed: float = 0.043
    cacheless_per_token: float = 0.00046

    # --- detection, routing, client memory (Experiment 2 extensions) ----------------
    # Time to notice that the node a hop was sent to has left. None = one hop time
    # (the send fails as fast as a normal send). Petals detects a silent crash by
    # request_timeout, 180 s.
    detect_time: Optional[float] = None
    # The client remembers nodes it has discovered or used within this many seconds
    # and can route to them without the discovery layer (Petals refreshes its routing
    # table every 60 s and keeps using what it has). 0 = knows only its current chain.
    known_window: float = 0.0
    # "random": pick any discoverable replica. "min_latency": pick the fastest known
    # (Petals routes by measured latency).
    routing: str = "random"
    # A hop slower than this multiple of its nominal time counts as a failure and the
    # client re-routes. None = never (Petals: only request_timeout, 180 s, so a node
    # slowed 10x or 100x is never replaced mid-session).
    slow_threshold: Optional[float] = None

    # --- discovery held on the server nodes (Experiment 2b) --------------------------
    # 0 = oracle registry (entries live outside the swarm). k > 0 = each node's entry is
    # stored on k other live nodes, Kademlia-style, and is discoverable only while at
    # least one holder is alive; nodes re-store their entry every republish_interval.
    dht_k: int = 0
    republish_interval: float = 60.0

    # --- concurrent clients (Experiment 2b) ----------------------------------------
    # M clients run generations on the same swarm. A hop on a node used by L clients
    # costs (1 + load_alpha * (L - 1)) times its single-client time. Petals Table 3:
    # 10 clients on 10 servers get 1.17 vs 1.65 steps/s alone, so alpha ≈ 0.045 with
    # GPU batching; without batching alpha → 1.
    clients: int = 1
    load_alpha: float = 0.0

    # --- defender's cut (Experiment 2) ------------------------------------------
    cut: Optional["Cut"] = None

    # --- bookkeeping -----------------------------------------------------------
    time_budget: float = 3600.0    # give up (DNF) beyond this many simulated seconds
    seed: int = 0


@dataclass(frozen=True)
class Cut:
    """A simultaneous removal applied at `time` seconds into the run.

    kind = "nodes":    each live node is removed independently with probability `amount`
    kind = "stage":    every live node holding stage int(amount) is removed
    kind = "registry": each node's discovery entry is lost with probability `amount`,
                       and nodes joining afterwards register only with probability
                       1 - amount. Live nodes keep serving; the client keeps using the
                       nodes it is already connected to, but can only route to nodes it
                       can discover. Removing discovery stops recovery, not inference.
    kind = "degrade":  each live node is slowed by `factor` with probability `amount`.
                       It stays alive and registered. The compute attack.
    """
    time: float
    kind: str
    amount: float
    factor: float = 10.0


@dataclass
class Node:
    nid: int
    stage: int
    alive: bool = True
    # tokens of this client's session currently in the node's KV cache
    cached: int = 0
    # multiplier on this node's compute time (degrade cut)
    speed: float = 1.0
    # number of clients currently routed through this node
    load: int = 0


class Registry:
    """Discovery layer: which live nodes hold which stage.

    `hidden` holds node ids whose entry was lost to a registry cut; `register_prob` is
    the chance a node joining after the cut gets an entry at all.
    """

    def __init__(self, dht_k: int = 0) -> None:
        self.by_stage: Dict[int, List[Node]] = {}
        self._next = 0
        self.hidden: set = set()
        self.register_prob: float = 1.0
        self.dht_k = dht_k
        self.holders: Dict[int, List[Node]] = {}  # nid -> nodes holding its entry
        self.entries_lost: int = 0

    def add(self, stage: int, rng: Optional[random.Random] = None) -> Node:
        n = Node(self._next, stage)
        self._next += 1
        self.by_stage.setdefault(stage, []).append(n)
        if self.register_prob < 1.0 and rng is not None and rng.random() >= self.register_prob:
            self.hidden.add(n.nid)
        if self.dht_k > 0:
            self.store(n, rng)
        return n

    def store(self, n: Node, rng: Optional[random.Random]) -> None:
        """(Re)publish n's entry onto dht_k live nodes other than itself."""
        pool = [m for m in self.all_nodes() if m.alive and m is not n]
        if rng is not None:
            rng.shuffle(pool)
        self.holders[n.nid] = pool[: self.dht_k] if pool else [n]

    def entry_alive(self, n: Node) -> bool:
        if self.dht_k == 0:
            return True
        return any(h.alive for h in self.holders.get(n.nid, []))

    def remove(self, n: Node) -> None:
        n.alive = False
        lst = self.by_stage.get(n.stage, [])
        if n in lst:
            lst.remove(n)

    def live(self, stage: int) -> List[Node]:
        return [n for n in self.by_stage.get(stage, []) if n.alive]

    def discoverable(self, stage: int) -> List[Node]:
        """Live nodes the client can find through the discovery layer."""
        return [n for n in self.live(stage) if n.nid not in self.hidden and self.entry_alive(n)]

    def all_nodes(self) -> List[Node]:
        return [n for lst in self.by_stage.values() for n in lst]


@dataclass
class Stats:
    tokens: int = 0
    hops: int = 0
    failures: int = 0
    restarts: int = 0
    replays: int = 0
    replayed_tokens: int = 0
    replay_sizes: list = field(default_factory=list)  # k at each replay attempt
    waits: int = 0
    wait_time: float = 0.0
    finished: bool = False
    elapsed: float = 0.0
    tokens_at_cut: int = -1
    nodes_cut: int = 0
    slow_evictions: int = 0

    @property
    def steps_per_s(self) -> Optional[float]:
        if not self.finished or self.elapsed <= 0:
            return None
        return self.tokens / self.elapsed

    @property
    def observed_hop_failure_p(self) -> float:
        return self.failures / self.hops if self.hops else 0.0


# ---------------------------------------------------------------------------
# Tiny discrete-event scheduler
# ---------------------------------------------------------------------------

Proc = Generator[float, None, None]


class Sim:
    def __init__(self) -> None:
        self.now = 0.0
        self._heap: list = []
        self._seq = 0

    def schedule(self, delay: float, fn: Callable[[], None]) -> None:
        self._seq += 1
        heapq.heappush(self._heap, (self.now + delay, self._seq, fn))

    def process(self, gen: Proc) -> None:
        def step() -> None:
            try:
                delay = next(gen)
            except StopIteration:
                return
            self.schedule(delay, step)
        self.schedule(0.0, step)

    def run(self, until: Optional[float] = None, stop: Optional[Callable[[], bool]] = None) -> None:
        while self._heap:
            t, _, fn = heapq.heappop(self._heap)
            if until is not None and t > until:
                self.now = until
                return
            self.now = t
            fn()
            if stop is not None and stop():
                return


# ---------------------------------------------------------------------------
# Node churn processes (only active when leave_rate / join_rate > 0)
# ---------------------------------------------------------------------------

def _node_lifetime(sim: Sim, reg: Registry, node: Node, rate: float, rng: random.Random) -> Proc:
    yield rng.expovariate(rate)
    reg.remove(node)


def _joiner(sim: Sim, reg: Registry, stage: int, cfg: Config, rng: random.Random) -> Proc:
    while True:
        yield rng.expovariate(cfg.join_rate)
        n = reg.add(stage, rng)
        if cfg.leave_rate > 0:
            sim.process(_node_lifetime(sim, reg, n, cfg.leave_rate, rng))


def _republisher(sim: Sim, reg: Registry, cfg: Config, rng: random.Random) -> Proc:
    while True:
        yield cfg.republish_interval
        for n in reg.all_nodes():
            if n.alive:
                if not reg.entry_alive(n):
                    reg.entries_lost += 1
                reg.store(n, rng)


def _cutter(sim: Sim, reg: Registry, cfg: Config, rng: random.Random, st: "Stats") -> Proc:
    cut = cfg.cut
    assert cut is not None
    yield cut.time
    st.tokens_at_cut = st.tokens
    if cut.kind == "nodes":
        for n in reg.all_nodes():
            if n.alive and rng.random() < cut.amount:
                reg.remove(n); st.nodes_cut += 1
    elif cut.kind == "stage":
        for n in reg.live(int(cut.amount)):
            reg.remove(n); st.nodes_cut += 1
    elif cut.kind == "registry":
        for n in reg.all_nodes():
            if rng.random() < cut.amount:
                reg.hidden.add(n.nid); st.nodes_cut += 1
        reg.register_prob = 1.0 - cut.amount
    elif cut.kind == "degrade":
        for n in reg.all_nodes():
            if n.alive and rng.random() < cut.amount:
                n.speed = cut.factor; st.nodes_cut += 1
    else:
        raise ValueError(cut.kind)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def _client(sim: Sim, cfg: Config, strategy: Strategy, reg: Registry,
            rng: random.Random, st: Stats, start_delay: float = 0.0) -> Proc:
    S, T, p = cfg.stages, cfg.seq_len, cfg.hop_failure_p

    known: Dict[int, tuple] = {}  # nid -> (node, last seen)

    def remember(nodes: List[Node]) -> None:
        if cfg.known_window > 0:
            for n in nodes:
                known[n.nid] = (n, sim.now)

    def candidates(stage: int) -> List[Node]:
        found = reg.discoverable(stage)
        remember(found)
        if cfg.known_window > 0:
            recalled = [n for n, t in known.values()
                        if n.stage == stage and n.alive and sim.now - t <= cfg.known_window]
            found = list({n.nid: n for n in found + recalled}.values())
        return found

    def pick(stage: int, avoid: Optional[Node]) -> Proc:
        """Choose a live node for `stage`, waiting if there is none. Returns via st._picked."""
        while True:
            found = candidates(stage)
            cands = [n for n in found if n is not avoid] or found
            if cands:
                if cfg.routing == "min_latency":
                    rng.shuffle(cands)
                    st._picked = min(cands, key=lambda n: n.speed)  # type: ignore[attr-defined]
                else:
                    st._picked = rng.choice(cands)  # type: ignore[attr-defined]
                remember([st._picked])  # type: ignore[attr-defined]
                return
            st.waits += 1
            st.wait_time += cfg.poll_interval
            yield cfg.poll_interval

    chain: List[Optional[Node]] = [None] * S

    def attach(stage: int, n: Node) -> None:
        old = chain[stage]
        if old is not None and old is not n:
            old.load -= 1
        if old is not n:
            n.load += 1
        chain[stage] = n

    yield start_delay
    for s in range(S):
        yield from pick(s, None)
        attach(s, st._picked)  # type: ignore[attr-defined]
        chain[s].cached = 0
    remember([n for n in chain if n is not None])

    k = 0  # tokens whose KV is established on every stage (session length so far)
    s = 0
    while k < T:
        node = chain[s]
        assert node is not None

        if strategy is Strategy.CACHELESS:
            cost = cfg.cacheless_fixed + (k + 1) * cfg.cacheless_per_token
        else:
            cost = cfg.hop_time + k * cfg.hop_time_per_token
        nominal = cost
        cost *= node.speed * (1 + cfg.load_alpha * max(node.load - 1, 0))
        dead = not node.alive
        if dead and cfg.detect_time is not None:
            cost = cfg.detect_time
        yield cost
        st.hops += 1

        too_slow = cfg.slow_threshold is not None and cost > cfg.slow_threshold * nominal
        if too_slow and cfg.routing == "min_latency":
            # a latency-aware router only switches when it knows somewhere faster
            too_slow = any(n.speed < node.speed for n in candidates(s) if n is not node)
        if too_slow:
            st.slow_evictions += 1
        failed = dead or too_slow or (p > 0 and rng.random() < p)
        if not failed:
            node.cached = k + 1
            s += 1
            if s == S:
                s = 0
                k += 1
                st.tokens = k
            continue

        # --- failure on stage s ----------------------------------------------------
        st.failures += 1
        node.cached = 0
        if node.alive and cfg.leave_rate == 0 and cfg.cut is None and not too_slow:
            # Petals-style "reset": the node is still there but its cache is gone.
            replacement = node
        else:
            yield from pick(s, node)
            replacement = st._picked  # type: ignore[attr-defined]
        attach(s, replacement)
        yield cfg.reconnect_time

        if strategy is Strategy.RESTART:
            st.restarts += 1
            yield cfg.restart_cost_per_token * k + cfg.restart_cost_per_token_sq * k * k
            for n in chain:
                if n is not None:
                    n.cached = 0
            k, s = 0, 0
        elif strategy is Strategy.CACHELESS:
            pass  # redo this hop; nothing to rebuild
        else:  # REPLAY
            # Rebuild the replacement's KV cache for the k established tokens from the
            # client-side cache. The replay send is itself a send and can fail (w.p. p),
            # and the node can leave mid-replay, in which case we pick again.
            while k > 0:
                dead = not replacement.alive
                if dead and cfg.detect_time is not None:
                    yield cfg.detect_time
                else:
                    yield (k * cfg.replay_per_token * replacement.speed
                           * (1 + cfg.load_alpha * max(replacement.load - 1, 0)))
                st.replays += 1
                st.replayed_tokens += k
                st.replay_sizes.append(k)
                st.hops += 1
                replay_failed = dead or (p > 0 and rng.random() < p)
                if not replay_failed:
                    replacement.cached = k
                    break
                st.failures += 1
                if replacement.alive and cfg.leave_rate == 0 and cfg.cut is None:
                    pass
                else:
                    yield from pick(s, replacement)
                    replacement = st._picked  # type: ignore[attr-defined]
                    attach(s, replacement)
                yield cfg.reconnect_time
            # then fall through and retry hop s for token k+1
            assert k == 0 or chain[s].cached == k, "replacement cache not rebuilt"  # type: ignore[union-attr]

    st.finished = True
    st.elapsed = sim.now
    for n in chain:
        if n is not None:
            n.load -= 1


def simulate_all(cfg: Config, strategy: Strategy) -> List[Stats]:
    """Run cfg.clients clients on one swarm; returns one Stats per client."""
    rng = random.Random(cfg.seed)
    sim = Sim()
    reg = Registry(dht_k=cfg.dht_k)
    for s in range(cfg.stages):
        for _ in range(cfg.replicas):
            n = reg.add(s, rng)
            if cfg.leave_rate > 0:
                sim.process(_node_lifetime(sim, reg, n, cfg.leave_rate, rng))
        if cfg.join_rate > 0:
            sim.process(_joiner(sim, reg, s, cfg, rng))
    if cfg.dht_k > 0:
        for n in reg.all_nodes():       # initial entries stored once the population exists
            reg.store(n, rng)
        sim.process(_republisher(sim, reg, cfg, rng))
    stats = [Stats() for _ in range(cfg.clients)]
    if cfg.cut is not None:
        sim.process(_cutter(sim, reg, cfg, rng, stats[0]))
    for i, st in enumerate(stats):
        # stagger starts across one no-failure step so clients are not in lockstep
        delay = 0.0 if i == 0 else rng.random() * cfg.stages * cfg.hop_time
        sim.process(_client(sim, cfg, strategy, reg, rng, st, start_delay=delay))
    sim.run(until=cfg.time_budget, stop=lambda: all(st.finished for st in stats))
    for st in stats:
        if not st.finished:
            st.elapsed = cfg.time_budget
    for st in stats[1:]:
        st.tokens_at_cut = stats[0].tokens_at_cut
        st.nodes_cut = stats[0].nodes_cut
    return stats


def simulate(cfg: Config, strategy: Strategy) -> Stats:
    """Single-client view: client 0 of simulate_all."""
    return simulate_all(cfg, strategy)[0]


@dataclass
class Summary:
    strategy: Strategy
    cfg: Config
    runs: List[Stats]

    @property
    def n_finished(self) -> int:
        return sum(1 for r in self.runs if r.finished)

    @property
    def mean_steps_per_s(self) -> Optional[float]:
        """Throughput as the paper reports it: tokens / mean elapsed time over repeats.

        Averaging steps/s per run instead would be biased upward for the restart
        strategy, whose completion time is heavy-tailed (geometric number of attempts).
        Runs that did not finish are censored at the time budget, so when `censored`
        is True this is an upper bound. None if no run finished at all (DNF).
        """
        if self.n_finished == 0:
            return None
        times = [r.elapsed for r in self.runs]  # unfinished runs carry elapsed=budget
        return self.cfg.seq_len / (sum(times) / len(times))

    @property
    def censored(self) -> bool:
        return 0 < self.n_finished < len(self.runs)

    @property
    def mean_observed_p(self) -> float:
        return sum(r.observed_hop_failure_p for r in self.runs) / len(self.runs)


def run_many(cfg: Config, strategy: Strategy, seeds: int = 20) -> Summary:
    runs = [simulate(replace(cfg, seed=cfg.seed + i), strategy) for i in range(seeds)]
    return Summary(strategy, cfg, runs)
