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

(The 1/(1−p) is because a replay send can itself fail.) T=128, p=1e-2: 0.34 s/step,
2.9 steps/s; sim 3.34, paper 3.38. The curve leaves the flat region where the failure
term equals the hop term, p ≈ h/C ≈ 0.022/8.6 ≈ 2.5e-3, which is where the blue line
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

T=128: 2.0e-3. T=1024: 2.4e-4. Both match the sweep plot. This is the most reusable
number from Experiment 1: for any pipeline, S and T alone give the failure rate above
which stateless restart stops being viable.

### Where this comes from

This is checkpoint/restart analysis from HPC. The Petals client-side cache is a
checkpoint taken every token at near-zero cost, Algorithm 1 is "restart from the last
checkpoint" with recovery cost R + k·r, and the restart baseline is the no-checkpoint
case. Young (1974) and Daly (2006) give the optimal checkpoint interval as
√(2·C_ckpt·MTBF); with C_ckpt ≈ 0 that is every step, which is what Petals does. The
formal tool for "expected cost per cycle over expected cycle length" is the
renewal-reward theorem (Ross, ch. 7).

### Analytic vs simulation

`python -m churnsim.analytic` reports sim/formula ratios. Replay: 0.93 to 1.15 across
all 22 cells (the formula is slightly pessimistic at 128 tokens, p around 1e-2, because
it ignores that failures during replay are rarer than failures during generation).
Restart: 0.87 to 1.16 on all uncensored cells. The one large miss, restart 1024 tokens
at p=2e-3 (ratio 11.6), is the censored point: the formula says 0.03 steps/s, about
9 hours, so a 1-hour budget could only ever return an upper bound there.

## 2. Node churn: recovery time feeds back into failure rate

A node contacted every τ seconds with Exp(λ) lifetime has died since last contact with
probability p = 1 − e^(−λτ) ≈ λτ. The naive estimate uses τ₀, the no-failure step time.
But τ is the realised step time, recoveries included, and recoveries raise τ, which
raises p. With τ = τ₀ + S·p·C and p = λτ:

    τ = τ₀ / (1 − S·λ·C)

which diverges at

    λ* = 1 / (S·C)

For replay at 1024 tokens, C ≈ 6 + 512×0.005 ≈ 8.6 s, so λ* ≈ 0.029 /s. The churn
sweep collapsed between λ = 2e-2 and 5e-2.

Read S·λ·C as a branching ratio: one failure, during its own recovery, leaves the other
stages' nodes untouched for C seconds and so produces S·λ·C further failures on average.
Below one, cascades die out; above one, they grow without bound. The same structure is
utilisation ρ → 1 in a queue, a supercritical Galton–Watson process, and the retry
storm in distributed systems. Bronson et al. call the class "metastable failures":
stable under nominal load, with a self-sustaining feedback loop past a threshold.

Consequences for the defender's question:

- C is dominated by the fixed detection-and-discovery term (6 s), not the replay (2.6 s).
  So λ* is mostly set by how fast a client can find a replacement. Doubling discovery
  latency halves λ*. That is "attack the discovery layer", as a number, and Experiment 2
  can test it by varying `reconnect_time` rather than removing nodes.
- Empty stages have their own formula. With join rate n·λ balancing leave rate λ, live
  replicas per stage form an M/M/∞ birth–death process with Poisson(n) stationary
  distribution. P(stage empty) = e^(−n), about 5% for n = 3, independent of λ. λ sets
  only how long each outage lasts, about 1/(n·λ). At λ = 1e-3 an outage lasts ~5 min
  while a 128-token generation takes 12 s, so short runs rarely see one; 1024-token
  runs (100 s) start to. Replica count and churn rate are two different levers, and
  Experiment 2's "every node holding one stage" cut is e^(−n) driven to 1.

## 3. Discrepancies with the paper

### Restart, 1024 tokens, p = 1e-3: sim 0.79 (formula 0.90), paper 0.48

The formula gives ~60 attempts, each dying after ~250 tokens (15 s) plus 3 s reconnect,
≈ 1100 s. The paper's number needs each failed attempt to cost about 17 s more.

Tested (`python -m churnsim.restart_variants --seeds 50`), fitting R and one extra
parameter on the five finite restart cells:

| hypothesis            | fit             | loss  | 128/1e-4 | 128/1e-3 | 128/1e-2 | 1024/1e-4 | 1024/1e-3 |
|-----------------------|-----------------|-------|----------|----------|----------|-----------|-----------|
| fixed R               | R=3.0           | 0.413 | 0.89     | 0.94     | 0.99     | 0.98      | **1.87**  |
| R + a·k (linear)      | R=1.0, a=0.07   | 0.157 | 0.84     | 0.84     | 1.08     | **0.78**  | 1.18      |
| R + b·k² (quadratic)  | R=3.0, b=1e-4   | 0.155 | 0.88     | 0.91     | 0.97     | **0.78**  | 1.30      |

(cells are sim/paper ratios). Both progress-dependent costs halve the loss, but they do
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

Status: left as fixed R in the calibration. Open. Next step is reading the benchmark
code in the Petals repository for how the restart baseline was actually run.

### Replay, 128 tokens, p = 1e-4: sim 10.41, paper 11.40

Sampling noise on both sides. A 512-hop generation at p = 1e-4 sees 0.05 failures on
average: ~2.5 failures across the paper's 50 repeats, ~1.5 across our 30 seeds. The
paper's 11.4 (identical to p = 0) almost certainly means zero failures happened. Cells
with p·S·T ≪ 1 should either use ~1000 seeds or be dropped from the calibration loss.
Currently they are in the loss with equal weight, which is wrong.

## 4. Reading

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
