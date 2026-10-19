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
  leave_rate      Node churn. Each node's lifetime is Exp(leave_rate); a leaving node is
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
    reconnect_time: float = 2.5
    # REPLAY only: per-token cost for a replacement stage to rebuild its KV cache from
    # the client-side cache (one batched forward over the prefix, for that stage only).
    replay_per_token: float = 0.0089
    # CACHELESS only: per-hop cost of sending and recomputing the whole prefix.
    cacheless_fixed: float = 0.043
    cacheless_per_token: float = 0.00046

    # --- bookkeeping -----------------------------------------------------------
    time_budget: float = 3600.0    # give up (DNF) beyond this many simulated seconds
    seed: int = 0


@dataclass
class Node:
    nid: int
    stage: int
    alive: bool = True
    # tokens of this client's session currently in the node's KV cache
    cached: int = 0


class Registry:
    """Discovery layer: which live nodes hold which stage."""

    def __init__(self) -> None:
        self.by_stage: Dict[int, List[Node]] = {}
        self._next = 0

    def add(self, stage: int) -> Node:
        n = Node(self._next, stage)
        self._next += 1
        self.by_stage.setdefault(stage, []).append(n)
        return n

    def remove(self, n: Node) -> None:
        n.alive = False
        lst = self.by_stage.get(n.stage, [])
        if n in lst:
            lst.remove(n)

    def live(self, stage: int) -> List[Node]:
        return [n for n in self.by_stage.get(stage, []) if n.alive]

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
    waits: int = 0
    wait_time: float = 0.0
    finished: bool = False
    elapsed: float = 0.0

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
        n = reg.add(stage)
        if cfg.leave_rate > 0:
            sim.process(_node_lifetime(sim, reg, n, cfg.leave_rate, rng))


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def _client(sim: Sim, cfg: Config, strategy: Strategy, reg: Registry,
            rng: random.Random, st: Stats) -> Proc:
    S, T, p = cfg.stages, cfg.seq_len, cfg.hop_failure_p

    def pick(stage: int, avoid: Optional[Node]) -> Proc:
        """Choose a live node for `stage`, waiting if there is none. Returns via st._picked."""
        while True:
            cands = [n for n in reg.live(stage) if n is not avoid] or reg.live(stage)
            if cands:
                st._picked = rng.choice(cands)  # type: ignore[attr-defined]
                return
            st.waits += 1
            st.wait_time += cfg.poll_interval
            yield cfg.poll_interval

    chain: List[Optional[Node]] = [None] * S
    for s in range(S):
        yield from pick(s, None)
        chain[s] = st._picked  # type: ignore[attr-defined]
        chain[s].cached = 0

    k = 0  # tokens whose KV is established on every stage (session length so far)
    s = 0
    while k < T:
        node = chain[s]
        assert node is not None

        if strategy is Strategy.CACHELESS:
            cost = cfg.cacheless_fixed + (k + 1) * cfg.cacheless_per_token
        else:
            cost = cfg.hop_time + k * cfg.hop_time_per_token
        yield cost
        st.hops += 1

        failed = (not node.alive) or (p > 0 and rng.random() < p)
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
        if node.alive and cfg.leave_rate == 0:
            # Petals-style "reset": the node is still there but its cache is gone.
            replacement = node
        else:
            yield from pick(s, node)
            replacement = st._picked  # type: ignore[attr-defined]
        chain[s] = replacement
        yield cfg.reconnect_time

        if strategy is Strategy.RESTART:
            st.restarts += 1
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
                yield k * cfg.replay_per_token
                st.replays += 1
                st.replayed_tokens += k
                st.hops += 1
                replay_failed = (not replacement.alive) or (p > 0 and rng.random() < p)
                if not replay_failed:
                    replacement.cached = k
                    break
                st.failures += 1
                if replacement.alive and cfg.leave_rate == 0:
                    pass
                else:
                    yield from pick(s, replacement)
                    replacement = st._picked  # type: ignore[attr-defined]
                    chain[s] = replacement
                yield cfg.reconnect_time
            # then fall through and retry hop s for token k+1

    st.finished = True
    st.elapsed = sim.now


def simulate(cfg: Config, strategy: Strategy) -> Stats:
    rng = random.Random(cfg.seed)
    sim = Sim()
    reg = Registry()
    for s in range(cfg.stages):
        for _ in range(cfg.replicas):
            n = reg.add(s)
            if cfg.leave_rate > 0:
                sim.process(_node_lifetime(sim, reg, n, cfg.leave_rate, rng))
        if cfg.join_rate > 0:
            sim.process(_joiner(sim, reg, s, cfg, rng))
    st = Stats()
    sim.process(_client(sim, cfg, strategy, reg, rng, st))
    sim.run(until=cfg.time_budget, stop=lambda: st.finished)
    if not st.finished:
        st.elapsed = cfg.time_budget
    return st


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
