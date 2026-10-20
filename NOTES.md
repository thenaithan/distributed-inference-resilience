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

A note on seeds: an earlier 30-seed calibration (seeds 0..29) happened to draw ~18%
fewer failures than nominal on every 128-token cell, which made replay at 128/1e-2
look like an exact match to the paper and inflated the fitted reconnect to 6 s. With
200 seeds and no censoring the fit moved to the values quoted in the README. Lesson
kept: never calibrate on fewer than a few hundred seeds, and check `observed_p`
against nominal in the output.

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

Tested (`python -m churnsim.restart_variants --seeds 50`), fitting R and one extra
parameter on the five finite restart cells:

| hypothesis            | fit             | loss  | 128/1e-4 | 128/1e-3 | 128/1e-2 | 1024/1e-4 | 1024/1e-3 |
|-----------------------|-----------------|-------|----------|----------|----------|-----------|-----------|
| fixed R               | R=3.0           | 0.413 | 0.89     | 0.94     | 0.99     | 0.98      | **1.87**  |
| R + a·k (linear)      | R=1.0, a=0.07   | 0.157 | 0.84     | 0.84     | 1.08     | **0.78**  | 1.18      |
| R + b·k² (quadratic)  | R=3.0, b=1e-4   | 0.155 | 0.88     | 0.91     | 0.97     | **0.78**  | 1.30      |

(cells are sim/paper ratios; 50 seeds with the 1 h budget, so the 1024/1e-3 row is a
slight upper bound, which does not change the picture). Both progress-dependent costs
halve the loss, but they do
it by trading the 1024/1e-3 error for a 1024/1e-4 error. A failed attempt at 1e-4
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

### Not modelled, and relevant

- Clients cache routing tables (Petals: `update_period = 60 s`) and could keep using
  known-but-unregistered nodes. Our client knows only its current chain, which makes
  the registry cut slightly more effective than it would be against Petals. The fix is
  a `known` set per client with a staleness window; cheap to add.
- The registry here is a single oracle with per-entry loss. A real DHT loses entries
  as a function of which nodes are removed, so a node cut and a registry cut are
  correlated; "remove fraction g of the DHT" is a lower bound on the damage a node
  cut does to discovery. In Petals the DHT runs on the same hivemind peers as the
  servers.
- Detection latency is one hop time (the lost node is noticed on the next send). With
  a 180 s timeout the registry-cut hazard is unchanged but every heal takes longer.
- One request, one client. Many concurrent clients contend for the same joiners.

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
