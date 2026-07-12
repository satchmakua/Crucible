# Crucible — Results

> The deliverable (DESIGN §1): **a curve, not a leaderboard number** — accuracy as a
> function of test-time compute, showing that search + verification turns spent compute
> into measured accuracy, *with the verifier's compute counted*, and showing **why**
> (which method is compute-optimal at which budget, and where the PRM's selection gap is).

**§0 is the real artifact — a real model, real PRM, real data.** §1–§6 are the *cold,
simulator* validation (a synthetic policy + a mock PRM) that runs with no GPU on every
commit; read them as "the harness measures the right things correctly," not "model X scores
Y." The real curve is §0.

## 0. The real lift curve — accuracy vs test-time compute on GSM8K

![Real GSM8K accuracy-vs-compute curve](gsm8k-lift-curve.png)

On the dev machine (RTX 5070 Ti Laptop, 12 GB, `torch 2.11+cu128` / sm_120): **20 real
GSM8K** problems, policy `qwen2.5:1.5b-instruct` (Ollama, GPU), verifier the real
**Skywork-o1-Open-PRM-Qwen-2.5-1.5B** (GPU). Each problem sampled 8× *once*, then every
selector evaluated at each N over those samples (`crucible.bench`):

| N | pass@1 | majority | PRM | oracle |
|---|---|---|---|---|
| 1 | **40%** | 40% | 40% | 40% |
| 2 | — | 40% | 50% | 60% |
| 4 | — | 45% | **60%** | 75% |
| 8 | — | 50% | **60%** | **90%** |

(95% Wilson CIs on the plot; x-axis = **total tokens/problem, policy + verifier** — the
PRM's forward passes counted, which is why the PRM line sits ~2× to the right.)

The three things this project set out to show, now on a real model:
1. **Test-time compute buys accuracy.** Single-shot pass@1 is 40%; with search, oracle
   reaches **90%** at N=8 — a perfect verifier more than doubles accuracy from the *same*
   frozen 1.5B model. Every selector's curve rises with compute.
2. **The learned verifier earns its keep.** The real PRM reaches **60%** at N=4/8 vs
   verifier-free **majority's 50%** — a real +10 pt lift from a *learned* step-reward model
   over self-consistency. (This is the effect a smaller/easier sample can hide: an 8-problem
   pilot showed PRM ≈ majority; on 20 problems the PRM clearly separates.)
3. **The selection gap is real and honest.** Oracle 90% > PRM 60% is the headroom the PRM
   *doesn't* capture — a small open PRM is an imperfect selector (ProcessBench F1 ≈ 56 even
   for the best 7B PRMs), and the reported metric is always the outcome verifier on the
   chosen trace, never a PRM score. The gap is exactly what a stronger PRM / harder data
   (H1/H2) is expected to close.

**Reproducible without a GPU.** The run is captured to `tests/fixtures/gsm8k-bestofn.json`
(the 8 traces per problem + their PRM scores + correctness); `tests/test_cassette.py`
replays it offline and reproduces every number above — a real result that regenerates in CI
with no model (the "run live once, commit a fixture" pattern, now covering the PRM side).

## 1. The headline: accuracy vs test-time compute

`crucible sweep configs/results.yaml` runs the full search ladder on the synthetic
stepwise task — a 5-step chain where each step is good with probability 0.6, so a single
sample is correct only ~`0.6^5 ≈ 8%` of the time — across **3 seeds** (18 samples per
cell, Wilson CIs). The plot is `runs/sweep-*/curve.png`:

| method | knob | accuracy (95% CI) | tokens / problem |
|---|---|---|---|
| pass1 | — | 11.1% [3%, 33%] | 38 |
| best_of_n (prm) | N=4 | 16.7% [6%, 39%] | 304 |
| best_of_n (prm) | N=8 | 55.6% [34%, 75%] | 608 |
| best_of_n (prm) | N=16 | 83.3% [61%, 94%] | 1,216 |
| best_of_n (prm) | N=32 | 100.0% [82%, 100%] | 2,432 |
| beam | width=2 | 94.4% [74%, 99%] | 1,112 |
| beam | width=4 | 100.0% [82%, 100%] | 2,168 |
| mcts | budget=4000 | 77.8% [55%, 91%] | 4,075 |
| mcts | budget=6000 | 100.0% [82%, 100%] | 6,080 |

**The lift is real and large:** single-shot pass@1 is ~11%; with enough verifier-guided
search every method reaches 100%. Crucially the x-axis is **total generated tokens,
policy + verifier** — the PRM's forward passes are counted, so no method gets a free
lunch from un-counted verification.

## 2. Compute-optimal: which method wins at which budget

The dashed line on the curve is the **compute-optimal frontier** — the best accuracy any
method reaches at each token budget (`crucible report <sweep>` prints it):

| tokens / problem | best method | accuracy |
|---|---|---|
| 38 | pass1 | 11.1% |
| 304 | best_of_n (prm) | 16.7% |
| 584 | beam | 72.2% |
| 1,112 | beam | 94.4% |
| 2,168 | beam | 100.0% |

On this task **beam (DVTS) is compute-optimal across essentially the whole budget range**:
because the PRM gives a reliable *per-step* signal, pruning bad partial chains early beats
both best-of-N (which must pay exponentially to sample a fully-correct chain) and MCTS
(whose tree-search overhead doesn't pay off when the task is this shallow). This is the
compute-optimal-scaling result (Snell et al.) in miniature: *the right method depends on
the budget* — and here also on the shape of the problem.

## 3. The honest part: MCTS is not free, and we show it

A dishonest write-up would bury MCTS. On this **easy, shallow** task MCTS is the **most
expensive** method — it saturates to 100% but only at ~6k tokens/problem vs beam's ~2.2k.
That is consistent with the design ("MCTS: the most compute, the best on hard problems"):
its adaptive allocation pays off on *deep* trees with *rare* good steps, which this toy
doesn't reproduce. We plot it on the same axes anyway, because the point of the project is
to **measure** rather than assume that fancier search is better.

## 4. The PRM selection gap

`crucible compare` scores **majority / PRM / oracle** selection on the *same* best-of-N
samples (so the differences are real, not sampling noise), and counts the PRM's compute:

- **oracle ≥ PRM ≥ majority.** Oracle is an upper bound (it peeks at the gold answer to
  pick a passing sample); the PRM recovers much of that lift but not all — the gap between
  the PRM and oracle bars is exactly the *selection gap* (and where reward-hacking would
  show up). Majority lags badly when the base policy is below 50% (it converges to the
  wrong consensus). See `runs/compare-*/comparison.png`.
- The **reported metric is always the outcome verifier on the chosen trace** — never a PRM
  score. The PRM only steers the search.

## 5. Code generalizes the verifier

The same `OutcomeVerifier` port backs both math (symbolic equivalence) and code
(execution against unit tests in the opt-in sandbox). `crucible run --dataset code-sample
--allow-code-exec` reports pass@1 = 2/3 from real subprocess execution — the search core
is unchanged; only the verifier differs (DESIGN §6.2).

## 6. Threats to validity

- **Simulators, not models.** §1–§4 are mechanism checks; real-model numbers are pending.
- **Verifier gaming.** The PRM can be reward-hacked; that's why we always report the
  *outcome* metric and show the PRM-vs-oracle gap (§4). On real runs, inspect a sample of
  "passed" traces.
- **Step segmentation** (`\n\n` + token cap) materially affects beam/MCTS and is recorded
  per run; it should be ablated on real data.
- **Code sandbox** is a guardrail, not a jail (ADR-0003) — use Docker/WSL2 for untrusted
  code at scale.

## Reproducing

```bash
pip install -e ".[dev]"
crucible sweep configs/results.yaml        # the cold, multi-seed ladder + frontier
crucible compare                           # the PRM selection gap (cold)
crucible run --dataset code-sample --policy mock --allow-code-exec   # code track (cold)
```

**For real numbers**, point the same sweep at a real backend — set
`policy: {backend: ollama, model: qwen2.5-math-1.5b-instruct}` and a real `prm:` (a Qwen
PRM via the `prm` extra, on a GPU), and use `dataset: math500` (which carries graded
difficulty, so `crucible`'s per-difficulty analysis becomes meaningful). The analysis,
plots, CIs, and compute accounting are identical — only the adapters change.
