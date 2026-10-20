# Notes: what the simulation shows and why

Working notes for Experiment 1. Formulas here are implemented in `churnsim/analytic.py`
and checked against `out/sweep.csv` by `python -m churnsim.analytic`.

Notation: S stages (4), T tokens, h hop time with a small per-cached-token term,
p per-hop failure probability, R reconnect cost, r replay cost per token, k tokens
generated so far.

## 1. The three curves are two formulas

### Replay (Petals Algorithm 1): linear in p

Every hop costs h and, with probability p, a recovery R + k·r. Expected seconds per token:

    1/throughput = S·h + S·p·(R + k̄·r) / (1 − p)

(The 1/(1−p) is because a replay send can itself fail; the exact form in
`analytic.py` also retries the hop.) T=128, p=1e-2: 4×0.022 + 4×0.01×(4.5 + 64×0.008)
≈ 0.29 s/step, 3.4 steps/s; sim 3.48, paper 3.38. The curve leaves the flat region where the failure
term equals the hop term, p ≈ h/C ≈ 0.022/8.6 ≈ 2.5e-3 (C at 1024 tokens), which is where the blue line
on the sweep plot starts to bend.

### Restart from scratch: exponential in p·S·T

A generation succeeds only if all S·T hops do: q = (1−p)^(S·T) ≈ e^(−pST). Attempts are
i.i.d., so the expected number is 1/q. A failed attempt dies after a truncated-geometric
number of hops (mean ≈ 1/p) and pays R:

    E[time] = (1/q − 1)·(E[hops | fail]·h + R) + S·T·h

T=128, p=1e-2: q ≈ 0.0058, 172 attempts, ≈ 770 s, 0.17 steps/s; sim 0.19, paper 0.18.
T=1024, p=1e-2: q ≈ 1e-18. That is the DNF.

### Cache-less: flat

No state survives a step, so a failure costs one hop. With R ≈ 0 the slope is
invisible until p is of order 1e-2.

### Crossover

Restart and replay cross where a generation expects about one failure:

    p* ≈ 1 / (S·T)

T=128: 2.0e-3. T=1024: 2.4e-4. This is a rule of thumb; the exact crossing also depends
on R and on the two hop times, and with the fitted costs lands a little lower (about
1.5e-3 at T=128). Still the most reusable number from Experiment 1: S and T alone put
the failure rate above which stateless restart stops being viable within a factor of
two.

### Where this comes from

This is checkpoint/restart analysis from HPC. The Petals client-side cache is a
checkpoint taken every token at near-zero cost, Algorithm 1 is "restart from the last
checkpoint" with recovery cost R + k·r, and the restart baseline is the no-checkpoint
case. Young (1974) and Daly (2006) give the optimal checkpoint interval as
√(2·C_ckpt·MTBF); with C_ckpt ≈ 0 that is every step, which is what Petals does. The
formal tool for "expected cost per cycle over expected cycle length" is the
renewal-reward theorem (Ross, ch. 7).

### Analytic vs simulation

`python -m churnsim.analytic` reports sim/formula ratios; see the table there for the
current numbers. The formulas agree with the simulation to within a few percent on
every uncensored cell. The large misses are the two censored restart points at 1024
tokens (p = 1e-3 and 2e-3); at 2e-3 the formula says 0.03 steps/s, about 9 hours per
run, so a 1-hour budget can only ever return an upper bound there.

A note on seeds. An earlier 30-seed calibration (seeds 0..29) happened to draw ~18%
fewer failures than nominal on every 128-token cell, which made replay at 128/1e-2
look like an exact match to the paper and inflated the fitted reconnect to 6 s. With
200 seeds and no censoring the fit moved to the values quoted in the README.

The number of seeds is set by what each result has to resolve, and is stated with it:

| use                                   | seeds | why                                                                 |
|---------------------------------------|-------|---------------------------------------------------------------------|
| calibration and hypothesis fits (§3)  | 200   | fitting a per-cell throughput to ~5%; a cell with ~0.5 failures per run needs ~100 failures in total; uncensored |
| throughput sweeps (sweep, churn)      | 100   | curve shape and ~10% per point; censored points are flagged           |
| completion probabilities (cuts)       | 100   | binomial standard error ≤ 0.05                                        |
| tests                                 | 1–200 | whatever the assertion needs                                          |

Always check `observed_p` against nominal in the sweep output.

## 2. Node churn: recovery time feeds back into failure rate

A node contacted every τ seconds with Exp(λ) lifetime has died since last contact with
probability p = 1 − e^(−λτ) ≈ λτ. The naive estimate uses τ₀, the no-failure step time.
But τ is the realised step time, recoveries included, and recoveries raise τ, which
raises p. With τ = τ₀ + S·p·C and p = λτ:

    τ = τ₀ / (1 − S·λ·C)

which diverges at

    λ* = 1 / (S·C)

For replay at 1024 tokens, C ≈ 4.5 + 512×0.008 ≈ 8.6 s, so λ* ≈ 0.029 /s. The churn
sweep collapsed between λ = 2e-2 and 5e-2.

Read S·λ·C as a branching ratio: one failure, during its own recovery, leaves the other
stages' nodes untouched for C seconds and so produces S·λ·C further failures on average.
Below one, cascades die out; above one, they grow without bound. The same structure is
utilisation ρ → 1 in a queue, a supercritical Galton–Watson process, and the retry
storm in distributed systems. Bronson et al. call the class "metastable failures":
stable under nominal load, with a self-sustaining feedback loop past a threshold.

Consequences for the defender's question:

- At 1024 tokens C splits about evenly between the fixed detection-and-discovery term
  (4.5 s) and the replay (~4 s); at short sequences the fixed term dominates. Both are
  levers. Doubling discovery latency adds S·λ·4.5 s to the branching ratio, and under
  real churn the fixed term is far larger than in the Table 1 setup (§4), so in
  practice λ* is set by detection. That is "attack the discovery layer", as a number,
  and Experiment 2 can test it by varying `reconnect_time` rather than removing nodes.
- Empty stages have their own formula. With join rate n·λ balancing leave rate λ, live
  replicas per stage form an M/M/∞ birth–death process with Poisson(n) stationary
  distribution. P(stage empty) = e^(−n), about 5% for n = 3, independent of λ. λ sets
  only how long each outage lasts, about 1/(n·λ). At λ = 1e-3 an outage lasts ~5 min
  while a 128-token generation takes 12 s, so short runs rarely see one; 1024-token
  runs (100 s) start to. Replica count and churn rate are two different levers, and
  Experiment 2's "every node holding one stage" cut is e^(−n) driven to 1.

## 3. Discrepancies with the paper

### Restart, 1024 tokens, p = 1e-3: sim 0.91 (formula 0.90), paper 0.48

The formula gives ~60 attempts, each dying after ~250 tokens (15 s) plus 3 s reconnect,
≈ 1100 s. The paper's number needs each failed attempt to cost about 17 s more.

Tested (`python -m churnsim.restart_variants --seeds 200`, uncensored), fitting R and
one extra parameter on the five finite restart cells:

| hypothesis            | fit             | loss  | 128/1e-4 | 128/1e-3 | 128/1e-2 | 1024/1e-4 | 1024/1e-3 |
|-----------------------|-----------------|-------|----------|----------|----------|-----------|-----------|
| fixed R               | R=3.0           | 0.439 | 0.95     | 0.85     | 0.92     | 1.02      | **1.89**  |
| R + a·k (linear)      | R=1.0, a=0.07   | 0.126 | 0.93     | **0.74** | 1.00     | **0.85**  | 1.06      |
| R + b·k² (quadratic)  | R=3.0, b=1e-4   | 0.118 | 0.95     | 0.83     | 0.90     | **0.85**  | 1.22      |

(cells are sim/paper ratios). Both progress-dependent costs
halve the loss, but they do
it by trading the 1024/1e-3 error for errors at 1024/1e-4 (and, for the linear
variant, 128/1e-3). A failed attempt at 1e-4
typically got ~450 tokens in, further than at 1e-3, so any cost that grows with
progress hits that cell harder, and the paper says that cell barely slowed down
(15.5 → 11.8). The five cells are not jointly consistent with any one-parameter
progress-dependent cost.

Other observations:

- The paper reports an adjusted standard deviation never above 0.2% over 50 repeats.
  For the restart strategy the completion time is a geometric mixture with coefficient
  of variation near 1, so 50 repeats should give a standard error of ~14%, not 0.2%.
  Either failures were injected more regularly than i.i.d. per send, or the restart
  figures were produced some other way. Worth asking the authors.
- 1024/1e-3 at 0.48 steps/s is 2133 s per run, 30 hours for 50 repeats of one cell.

Status: left as fixed R in the calibration. Open. The Table 1 harness (BLOOM-7.1B,
four 1080 Ti stages, the restart and Megatron-DeepSpeed baselines) is not in the
Petals repository; `benchmarks/benchmark_inference.py` there is the steady-state
Tables 2/3 benchmark with no failure injection, and nothing else public turns up.
Closing this means asking the authors.

### Replay, 128 tokens, p = 1e-4: sim 10.41, paper 11.40

Sampling noise on both sides. A 512-hop generation at p = 1e-4 sees 0.05 failures on
average: ~2.5 failures across the paper's 50 repeats, ~1.5 across our 30 seeds. The
paper's 11.4 (identical to p = 0) almost certainly means zero failures happened. Cells
with p·S·T ≪ 1 should either use ~1000 seeds or be dropped from the calibration loss.
Currently they are in the loss with equal weight, which is wrong.

## 4. What the Petals client actually does on failure (from source)

Read from `src/petals/client/inference_session.py` and `routing/sequence_manager.py`
at commit 22afba6 (Aug 2024). This is what the fitted C = 4.5 s + 8 ms/token stands for.

The paper is from December 2023 and its Table 1 experiment may be older still (it uses
BLOOM-7.1B and a "less mature implementation" of Algorithm 1), so the code read here
postdates the measurements. Checked against `git log`: the mechanism (client-side
`history`, replay in one request, immediate first retry, DHT refresh when a block has
no server, ban-and-ignore-if-all-banned) has been in place since the client was
written in Nov–Dec 2022 and is unchanged in substance since. The constants have
moved: `ban_timeout` was 60 s for ten days in Dec 2022 and 15 s since;
`request_timeout` was 30 s until Feb 2023 and 180 s since; `min_backoff = 1` and
`update_period = 60` are original. All of those predate the paper, so the client
described below is the one the paper's authors had. Whether Table 1 was measured with
it or with an earlier research branch is not recoverable from the public repo.

Per hop, `InferenceSession.step` tries the server; on any exception:

1. `on_request_failure`: the peer is banned (`ban_timeout = 15 s`, doubling on repeat)
   and dropped from the routing table. If a block now has no server, the routing
   state is marked stale.
2. Retry delay: `0, 1, 2, 4, …` s, capped at 60 (`min_backoff = 1`). The first retry
   is immediate; only repeated failures on the same hop back off.
3. `_update_sequence` → `make_sequence(mode="min_latency")`. If the state is stale this
   first does a synchronous DHT refresh (`get_remote_module_infos(latest=True)`), then
   shortest-path routing, then opens a new session (`connect_timeout = 5 s`).
4. The new session inherits `history`, the client-side cache of every input sent to
   that span. Its first `step` sends the whole history in one request
   (`inputs = self.history`), so the replacement rebuilds its KV cache in a single
   batched forward over the prefix. That is the per-token r.
5. If every server for a block is banned, the ban is ignored ("the error may actually
   be client-caused"). In the Table 1 setup there is exactly one server per stage, so
   each injected failure routes straight back to the reset server after the DHT refresh.

So R ≈ DHT refresh + routing + connect, not the backoff, and the 4.5 s fit is plausible
for a hivemind DHT lookup with `latest=True` plus session setup on a local network.

Two things the Table 1 experiment does not exercise, and that matter for churn:

- **Detection.** Injected failures raise immediately. A real crash or network partition
  is detected by `request_timeout = 180 s` unless the transport resets the connection.
  With detection at 180 s, C ≈ 190 s and λ* = 1/(S·C) drops from 0.029 to about
  1.3e-3 /s, a mean node lifetime of 13 minutes. Measured P2P session lengths
  (Stutzbach & Rejaie) have medians in that range. Detection latency is the dominant
  term in C for real churn, and the obvious defender lever.
- **Discovery staleness.** Routing information is refreshed every `update_period = 60 s`
  unless a failure empties a block. A node that joined in the last minute is invisible
  to a client that hasn't been forced to refresh, which lengthens the empty-stage waits
  in §2 beyond what the M/M/∞ estimate gives.

Both are one-parameter additions to `Config` (`detect_time`, `registry_lag`) and belong
in Experiment 2's cut analysis rather than in the Table 1 calibration.

## 5. Experiment 2: how much do you have to remove?

`churnsim/cuts.py`. A cut is applied 10 s into a 1024-token generation (replay
strategy, 4 stages, 3 replicas per stage unless stated) and the question is whether
the request completes within 20× its no-failure time (96 s), and how long it takes.
100 seeds per point. Three cut types, three swarm conditions: static (nothing rejoins),
healthy churn (λ = 1e-3, joins balance leaves, mean lifetime ~17 min) and busy churn
(λ = 1e-2, mean lifetime ~100 s).

### Static swarm: the graph answers it

- **Random fraction f of nodes.** The request completes iff every stage keeps at least
  one replica: P = (1 − fⁿ)^S. With n = 3, S = 4 that is 0.90 at f = 0.3, 0.59 at 0.5,
  0.06 at 0.8. The simulation sits on this curve (test asserts it). With n = 10 the
  knee moves to f ≈ 0.8 (0.66) and f = 0.9 gives 0.22. Replication buys a lot against
  random removal, which is the Albert–Barabási error-tolerance result in a pipeline.
- **Every replica of one stage.** n nodes, chosen. P = 0 by construction. The targeted
  cut is n nodes where the random cut needs on the order of (1 − ε^(1/S))^(1/n) · S·n;
  for n = 3 that is about 9 to 10 of 12 nodes to get below 10%.
- **Discovery layer.** P = 1 at any g up to and including 1. A chain that is already
  connected needs no discovery. Removing discovery does not stop inference, it stops
  recovery.

So in a static swarm the minimum simultaneous cut that stops a request is the
targeted one: n nodes, all of one stage. That is the essay's containment question as
a number, and it is small.

### Churning swarm: the answer inverts

With joins balancing leaves, nothing that removes nodes stops a request for long:

- A 100% node cut in the healthy swarm completes 93% of the time (the rest hit the
  20× deadline), after waiting ~15 min for one joiner per stage. In the busy swarm it
  completes 100% of the time in ~3× nominal. A whole-stage cut heals the same way:
  P = 1, cost 4.5× (healthy) or 2.4× (busy) nominal. The wait is about
  E[max of S Exp(join_rate)] ≈ H_S / join_rate, 690 s at 3e-3 and 69 s at 3e-2.
- The registry cut, which did nothing in the static swarm, is now the effective one.
  Losing all of discovery in the healthy swarm drops completion to 0.63; in the busy
  swarm to 0.07. Losing 80% of it gives 0.81 and 0.38. In the busy swarm, **removing
  80% of the discovery layer stops more requests than removing 100% of the nodes.**

The mechanism is the one §4 flagged: once discovery is gone, the client can keep using
the chain it has but cannot replace a node that leaves, so completion requires the S
chain nodes to outlive the remaining generation:

    P(complete | g = 1) = exp(−S · λ · t_remaining)

Healthy: exp(−4 × 1e-3 × 86) = 0.71 vs 0.63 measured. Busy: exp(−4 × 1e-2 × 86) = 0.03
vs 0.07 (the busy generation also runs slower, lengthening t_remaining; the two effects
roughly cancel). This is independent of the replica count n: with 10 replicas the
healthy g = 1 result is 0.71 and the busy one 0.02. **Replication does not help against
a discovery cut**, because the client cannot reach replicas it cannot find.

Two asymmetries worth stating plainly:

1. The busy swarm is *more* resilient to node removal (joins are faster) and *less*
   resilient to discovery removal (its chain nodes die sooner). Churn rate flips which
   cut is the minimum one.
2. The node cut and the stage cut act instantly and heal; the registry cut acts slowly
   and does not heal, because new joiners cannot register either. Its effect is a hazard
   S·λ applied to every in-flight request from the cut onward, so its cost grows with
   the length of the request and with how long the cut is held.

For the defender's question this gives a ranking. Against a static deployment, take
out one stage (n nodes). Against anything that self-heals, take out discovery, and the
amount needed is set by the churn rate: at λ = 1e-3 even g = 1 leaves a third of
requests completing, at λ = 1e-2 g ≈ 0.8 is enough to stop most. Both λ* from §2 and
this result say the same thing from different sides: the discovery-and-detection path
is where a distributed model is thinnest.

### The three cheap omissions, tested (`churnsim/extensions.py`)

**(a) Detection latency.** `detect_time` replaces the one-hop cost of a send to a node
that has left. With Petals' `request_timeout` of 180 s, C goes from 8.6 s to 189 s and
the cascade threshold λ* = 1/(S·C) from 0.029 to 1.3e-3 /s; at 30 s it is 6.5e-3. The
churn sweep collapses where the formula says: with 180 s detection the busy swarm
(λ = 1e-2) finishes 5 of 100 runs and even λ = 1e-3 (mean lifetime 17 min) loses
two-thirds of its throughput. Detection latency is the single largest term in C
under real churn, and it is a client-side constant.

**(b) Client routing cache.** `known_window` lets the client route to nodes it has
discovered or used within that window, without the discovery layer. Under busy churn
with the whole discovery layer removed, completion goes from 0.07 (knows only its
chain) to 0.11 (60 s cache, Petals' refresh period) to 0.17 (remembers everything).
At g = 0.8: 0.38 → 0.42 → 0.48. The cache softens the discovery cut by a few points,
not by a category: a remembered replica is only useful while it is alive, and in the
busy swarm it is not alive for long. The claim "80% of discovery stops more requests
than 100% of nodes" survives for every cache setting.

**(c) The compute attack: degraded but present nodes.** A `degrade` cut slows a
fraction f of nodes by 10× while they stay alive and registered. What happens depends
entirely on the client:

| client                                          | f = 0.3 | f = 0.5 | f = 0.8 | f = 1.0 |
|-------------------------------------------------|---------|---------|---------|---------|
| never evicts slow nodes (Petals)                | 3.1×    | 4.8×    | 7.2×    | 9.1×    |
| naive: evict any hop slower than 3× nominal     | 3.2×†   | 9.1×†   | 19×†    | DNF     |
| latency-aware: evict only if a faster node is known | 1.4× | 2.2×    | 5.3×    | 9.1×    |

(mean completion time over nominal, unfinished runs at the 20× deadline; † some runs
unfinished.) Three things to take from it:

- A slowed node never fails, so a client that only re-routes on failure keeps it for
  the whole session. Petals' only slow-node mechanism is `request_timeout`, 180 s, which
  for a 20 ms hop means a 9000× slowdown. A 10× tarpit holds a Petals session until it
  ends. Cost to the attacker: the node stays up and serves, slowly.
- A naive evictor is worse than no evictor. Each eviction costs a reconnect and a
  replay (~9 s), and if every replica of a stage is slow it evicts on every hop: at
  f ≥ 0.9 no run finishes. This is the §2 cascade again, triggered by the client's own
  policy rather than by churn.
- A latency-aware evictor, which switches only when it knows somewhere faster, gets
  the degraded swarm close to the best achievable (every stage on its fastest replica)
  and converges to the never-evict line only when everything is slow.

So the compute attack does not act through computation. A slowed node hurts exactly
as much as the routing layer lets it: not at all with good routing and spare
replicas, by its full factor with Petals' routing, and catastrophically with a bad
evictor. The thinnest part is still the coordination path; the degrade cut is one
more way in, not a separate one. Note what it is *not*: it is not a test of corrupted
outputs (a Byzantine node returning wrong activations), which Petals' Appendix H
discusses and this model does not represent.

### The two structural omissions, tested (`churnsim/exp2b.py`)

**(a) Discovery held on the server nodes.** With `dht_k = 3`, each node's entry lives
on three other nodes and is discoverable only while one of them is alive; nodes
republish every 60 s (also tested: 600 s and never). A node cut then also destroys the
entries whose holders all died, a fraction f³ of the survivors' entries.

Result: almost nothing changes. Static swarm, node cut, 100 seeds: P(complete) moves
from 0.58 to 0.48 at f = 0.5 and from 0.07 to 0.02 at f = 0.8, with no republishing at
all. Busy swarm: completion stays at 1.0 and time-to-complete rises by 7 to 16%, half
of which is already present with no cut at all (1.90× → 2.10× at f = 0: ordinary
holder churn makes entries flicker). The reason is
arithmetic: to lose an entry you must kill all k of its holders, so the discovery
damage from a node cut is f^k, which at k = 3 is small until f is large, and by then
the node cut itself has done the work. libp2p's Kademlia uses k = 20; hivemind's DHT
replicates similarly.

This sharpens the Experiment 2 result rather than softening it. The oracle registry
was not hiding a vulnerability; the opposite. **A discovery layer that lives on the
nodes cannot be cut by removing nodes**: node removal reaches it only as f^k. The
registry cut in Experiment 2 (entries hidden, joiners unregistered) therefore
corresponds to an attack on the DHT *protocol*, not on its hosts: eclipse, Sybil, key
poisoning, or partitioning the overlay. Those are the cheapest cuts in this model, but
they are a different kind of operation from taking machines offline, and the model
says nothing about how hard they are.

**(b) Concurrent clients.** M clients run on the same swarm; a node serving L clients
runs (1 + α·(L − 1)) times slower, with α = 0.05 (Petals Table 3: 10 clients on 10
servers get 1.17 vs 1.65 steps/s alone, so ~0.045 per client with GPU batching) and
α = 0.3 (weak batching). Busy swarm, per-client outcomes. Runs: 100 for M = 1,
20 runs × 5 clients for M = 5, 10 runs × 20 clients for M = 20; clients within a run
share one swarm and one cut, so the per-client samples are correlated and the
effective sample size is nearer the run count.

| cut            | M=1        | M=5, α=.05 | M=20, α=.05 | M=5, α=.3  | M=20, α=.3  |
|----------------|------------|------------|-------------|------------|-------------|
| nodes 50%      | 1.00, 2.5× | 1.00, 3.0× | 1.00, 4.3×  | 1.00, 5.3× | 0.65, 17×†  |
| nodes 100%     | 1.00, 3.3× | 1.00, 3.4× | 1.00, 5.1×  | 1.00, 5.7× | 0.60, 17×†  |
| stage          | 1.00, 2.4× | 1.00, 2.9× | 1.00, 3.8×  | 1.00, 5.0× | 0.70, 16×†  |
| registry 80%   | 0.38, 15×† | 0.27, 16×† | 0.10, 19×†  | 0.15, 19×† | 0.00, DNF   |
| registry 100%  | 0.07, 19×† | 0.01, 20×† | 0.00, DNF   | 0.00, DNF  | 0.00, DNF   |

(P(complete), mean time over nominal; † unfinished clients at the 20× deadline.)

Three things:

- Node and stage cuts still heal with Petals-like batching (α = 0.05), but the heal
  costs 20 clients 3.8 to 5.1× nominal instead of 2.4 to 3.3×, because the first
  joiner on an emptied stage receives all twenty of them.
- The discovery cut gets much worse with load: at 80% removed, per-client completion
  goes from 0.38 (one client) to 0.27 (five) to 0.10 (twenty, α = 0.05), and to zero
  at twenty clients with α = 0.3. Fewer discoverable replicas means each one carries more clients,
  which slows every generation, which lengthens t_remaining in exp(−S·λ·t), which
  loses more chains. Load is a third feedback loop alongside the two in §2 and §5(c).
- With weak batching (α = 0.3), twenty clients turn even a plain node cut into a 30 to
  40% loss: the swarm heals, but the few live nodes run 6.7× slower under the pile-on,
  generations take long enough for churn to break them again, and a third of clients
  never finish. This is the §2 cascade with load as the multiplier instead of
  detection time.

So concurrency sharpens every result, and most of all the discovery one.

### Not modelled, and relevant

- Byzantine nodes (wrong outputs) are a different threat and need output validation
  in the model. Petals' Appendix H discusses validator peers and running disjoint
  chains and comparing; neither is implemented in the public client.
- The DHT protocol attacks that the registry cut stands for (eclipse, Sybil,
  poisoning) are not modelled; the model gives their effect, not their cost.
- Node lifetimes are exponential; measured P2P session lengths are heavy-tailed.

## 6. Reading

1. Ross, *Introduction to Probability Models*, ch. 7, Renewal Theory. The
   renewal-reward theorem is the basis of every expected-time-per-step line above.
2. Daly, "A higher order estimate of the optimum checkpoint interval for restart
   dumps", Future Generation Computer Systems, 2006; and Young, "A first order
   approximation to the optimum checkpoint interval", CACM, 1974.
3. Bronson, Aghayev, Charapko, Zhu, "Metastable Failures in Distributed Systems",
   HotOS 2021; Huang et al., "Metastable Failures in the Wild", OSDI 2022. Marc
   Brooker's posts on retries and metastability are a lighter companion.
4. Harchol-Balter, *Performance Modeling and Design of Computer Systems*, 2013.
   M/M/∞ and the ρ → 1 instability.
5. Stutzbach and Rejaie, "Understanding Churn in Peer-to-Peer Networks", IMC 2006;
   Rhea, Geels, Roscoe, Kubiatowicz, "Handling Churn in a DHT", USENIX ATC 2004.
   Measured session lengths are heavy-tailed, not exponential; a one-line change in
   `_node_lifetime` worth trying. Rhea is about the discovery-layer recovery cost
   that dominates C.
6. Albert, Jeong, Barabási, "Error and attack tolerance of complex networks",
   Nature 2000; Newman, *Networks*, ch. 15 (percolation). Random versus targeted
   removal, which is Experiment 2's first two cuts.
7. Chandra and Toueg, "Unreliable failure detectors for reliable distributed
   systems", JACM 1996; Hayashibara et al., "The φ accrual failure detector",
   SRDS 2004. What sets the detection term in C.
8. Dean and Barroso, "The Tail at Scale", CACM 2013. Why mean throughput hides what
   restart does to users (the heavy tail that forced censoring).
9. Borzunov et al., "Distributed Inference and Fine-tuning of Large Language Models
   Over The Internet", arXiv:2312.08361. Section 3.2, Table 1, Appendix F.
