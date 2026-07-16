# PROGRESS — Crucible

A build log of what shipped and the notable decisions behind it. **Keep it honest** —
this is the working memory between build sessions. The forward-looking plan and
acceptance tests live in [ROADMAP.md](ROADMAP.md); this is the backward-looking
"what got done and why" companion.

**Current phase:** **M1–M7 built; the real lift curves are captured — 3-seed MATH-500, real
beam/MCTS, and a 7B baseline.** `docs/RESULTS.md §0` now **leads with a real 3-seed MATH-500
accuracy-vs-compute curve** (pass@1 38.3% → oracle 70%; PRM beats majority at every N on
identical samples) that regenerates offline in CI, with the honest caveats counted (the PRM's
~2× compute makes it ~a wash with majority at matched tokens; the 7B baseline is more
compute-efficient than 1.5B+search, so small-beats-big does **not** hold here; real beam/MCTS
are the most expensive and don't win on this non-reasoning stack). Hardening: **H1 ✓**, **H3 ✓**
(cassettes now cover policy + PRM + stepwise search), **H4 ✓**; **H2 measured (honest negative)**.
**163 tests** green. See the newest entry below.

## State of the tree

| Component | Where | Status |
|---|---|---|
| Value types + ports (the hexagon) | `domain/types.py`, `domain/ports.py` | ✅ M0 |
| Compute accounting | `domain/types.py::Compute` | ✅ M0 |
| Config (YAML ↔ dataclass) | `config.py` | ✅ M0 |
| Step segmentation + token approx | `segment.py` | ✅ M0 (used fully in M4) |
| Wilson CIs | `stats.py` | ✅ M0 |
| Mock policy (ScriptedPolicy) | `inference/mock.py` | ✅ M0 |
| Synthetic policy (seeded sim) | `inference/synthetic.py` | ✅ M2 |
| Stepwise policy + step PRM (demo pair) | `synthetic_stepwise.py` | ✅ M4 |
| Ollama policy (seeded per-request) | `inference/ollama.py` | ✅ M1 / H1 |
| Math CoT prompt builder | `prompts.py` | ✅ M1 |
| Answer extraction | `verify/answer_extract.py` | ✅ M0 |
| Math outcome verifier (+ compound-answer fix) | `verify/math_outcome.py` | ✅ M0 / H1 |
| PRM: mock + real adapters | `verify/process.py` | ✅ M3 (real, run on GPU) |
| Cassettes: policy + step + PRM record/replay | `inference/cassette.py` | ✅ H3 (both sides) |
| `crucible bench` (record/merge/curve) | `bench.py`, `cli.py` | ✅ H1/H3 (real curve) |
| `math500-hard` (levels 4–5) | `data/hf.py` | ✅ H1 |
| Code sandbox + outcome verifier | `verify/code_sandbox.py`, `verify/code_outcome.py` | ✅ M5 (opt-in) |
| pass1 strategy + registry | `search/` | ✅ M0 |
| best_of_n + selectors (maj/oracle/prm) | `search/best_of_n.py`, `search/selectors.py` | ✅ M3 |
| beam / DVTS | `search/beam.py` | ✅ M4 |
| mcts (PUCT over steps) | `search/mcts.py` | ✅ M6 |
| Sample dataset + registry | `data/` | ✅ M0 |
| GSM8K + MATH-500 loaders | `data/hf.py` | ✅ M1 |
| Code datasets: bundled + HumanEval/MBPP | `data/code_sample.py`, `data/hf.py` | ✅ M5 |
| Experiment runner | `runner.py` | ✅ M0 |
| Run records (JSON/CSV) + summary | `report.py` | ✅ M0 |
| Sweep (multi-seed) + accuracy-vs-compute curve | `sweep.py`, `report.py` | ✅ M2/M7 |
| Selection-gap comparison + bar chart | `runner.run_comparison`, `report.py` | ✅ M3 |
| Compute-optimal frontier + per-difficulty | `analyze.py` | ✅ M7 |
| Results report | `docs/RESULTS.md` | ✅ M7 |
| CLI (run/report/sweep/compare/bench/version) | `cli.py` | ✅ (all real) |

---

## 2026-07-14 — Final polish pass: a stale published number, and a Makefile

A DoD-driven polish audit (multi-agent, every finding verified by reading/running the
files). It caught a **wrong number in the results doc** — the kind of thing this project
exists to not do:

- **`docs/RESULTS.md` §1 published a stale MCTS row.** It said `mcts budget=4000 → 77.8%
  @ 4,075`; re-running the documented `crucible sweep configs/results.yaml` gives **88.9%
  [67%, 97%] @ 4,069**. 77.8% was the *pre-H4* figure — from before the fix that removed a
  wasted PRM eval. Corrected (+ the `budget=2000` row). Every other §1/§2 row reproduced
  exactly.
- **"Reproduces every number in this section" was an overclaim.** §0.1's best-of-N
  hard-subset rows (majority@8 4.2% / PRM@8 16.7% / oracle@8 45.8%) were asserted by no
  test, and the only command that produced them read from gitignored `runs/` dirs. Fixed by
  making the claim *true*: a new test derives the 8 problem ids from the beam cassette
  itself and asserts those cells.
- **Real-fixture tests silently skipped when a fixture was missing** — so CI's "it replays
  the recorded real run" (DoD §5) could pass vacuously. Guards are now hard assertions
  (verified: deleting a fixture fails loudly; restoring it passes).
- **`Makefile`** — DoD §5 asks for one `make demo`; there wasn't one. `make demo` now
  replays the real captured runs offline (no GPU/network) and prints the headline curve;
  plus `check`/`lint`/`typecheck`/`test`/`curve`. (`make` isn't installed on the Windows
  dev box, so the README gives the one-line equivalent; every recipe step was run by hand.)
- **CLI**: `run` gained the beam/MCTS knobs it never exposed (`--beam-width`,
  `--beam-expansions`, `--max-steps`, `--budget-tokens`) — ROADMAP M4/M6 documented these
  exact flags and they **errored** ("No such option"); capturing the real beam/MCTS runs had
  required hand-written YAML. Verified they drive real compute (width 1→4: 1,248→3,984 tok).
  Also: stale `--help` strings (`method` "more in later milestones", `dataset` "gsm8k/math500
  in M1", `selection` missing `prm`), a stale module docstring, and an unknown-backend error
  that omitted `synthetic`/`stepwise` while advertising unimplemented `hosted`.
- **Docs/hygiene**: README linked `CLAUDE.md` (globally gitignored → 404 in any clone);
  RESULTS' "Reproducing" block still framed real numbers as hypothetical and named a model
  (`qwen2.5-math-1.5b-instruct`) that resolves on no machine here; `§1–§6` mislabeled (§6 is
  Threats to validity); stale Tech-stack prose ("mock now"); `docs/README.md` didn't index
  RESULTS.md (the headline doc); `.gitignore` didn't cover `.claude/`/`CLAUDE.md` (relied on
  a machine-local global file); cells-`.json` provenance added for the two new figures.
- **Cassette provenance**: `RecordingPolicy.save()` recorded only `backend` — now also
  `model`, `seed`, and sampling params, so a committed fixture self-describes. Backward
  compatible (old fixtures still load).
- **Verified, not assumed**: every offline command in README/RESULTS was executed (pass1,
  code-sample, both sweeps, compare, bench curve, version) — all pass. **163 tests** green.

## 2026-07-14 — The real MATH-500 headline: `crucible bench`, real beam/MCTS, honest H1/H2

The "shinier version": the real curve is now **3-seed MATH-500** (not single-seed GSM8K), the
capture path is a **productized CLI verb**, beam/MCTS make **real-model contact** (and replay
offline), and the numbers are reported with every weakness surfaced.

**What shipped:**
- **`crucible bench` (record / merge / curve)** (`bench.py`, `cli.py`) — the capture path is
  now one command, not an interactive Python session. `record` samples N×/problem once,
  scores each with the outcome verifier + PRM, and writes a cassette (with `--offset`/`--limit`
  chunking and **incremental per-problem save** — a TDR crash saves progress and prints the
  resume offset); `merge` concatenates per-seed chunks (guarding duplicate problem ids +
  identity-meta mismatch); `curve` pools any number of cassettes into the accuracy-vs-compute
  curve + frontier offline, with `--difficulty` / `--restrict-to-runs` filters and `--run`
  overlays for beam/MCTS cells (recomputed from `results.csv` on the *matched* problem subset).
- **Step + PRM cassettes (H3 remainder)** (`inference/cassette.py`) — `RecordingPolicy` now
  records `sample_step` batches (keyed by problem + prefix hash, FIFO); new
  `RecordingProcessVerifier`/`CassetteProcessVerifier` record/replay PRM `score_steps`. A live
  **beam/MCTS run now replays offline call-for-call** (a miss raises — divergence is a failure,
  not a silent zero). `runner.run` flushes cassettes after every problem.
- **Seeded Ollama sampling** (`inference/ollama.py`) — per-request seeds derived from
  (run seed, problem, sample index), so multi-seed captures are real and reproducible;
  `sample_step` now charges the **full generation's `eval_count`** to its step, so beam/MCTS
  land on the same honest token axis as best-of-N (was a word-count undercount).
- **`math500-hard`** dataset (levels 4–5) + **`SyntheticPolicy.sample_step`** (a real stepwise
  chain, so the synthetic backend can drive beam/MCTS cold) + a **compound-answer fix** in
  `math_outcome.py` ($-wrapping so tuples/intervals/coordinates parse — this alone lifted the
  seed-0 MATH-500 pass@1 on several problems).

**The real runs (RTX 5070 Ti, Ollama `qwen2.5:1.5b-instruct` + Skywork 1.5B PRM):**
- **3-seed MATH-500** (problems 0–39 × seeds 0/1/2): pass@1 **38.3%** → oracle@8 **70%**; PRM
  beats majority at every N on identical samples (N=4: 53.3% vs 45.0%). Honest caveat: PRM's
  ~2× compute ≈ neutralizes the edge at matched tokens. Fixtures `math500-bestofn-seed{0,1,2}`.
- **Real beam + MCTS** on the 8 hardest problems (pass@1 = 0% there): beam 0/8 @ ~37k tok,
  MCTS 1/8 @ ~12k tok — honestly the most expensive, not on the frontier; the 1.5B instruct
  policy *restarts* rather than continues a partial trace. Recorded → replay offline.
- **7B baseline** (`qwen2.5:7b-instruct` pass@1): **67.5% @ 524 tok** — more compute-efficient
  than 1.5B+search, so **small-beats-big fails** on this stack (needs a stronger PRM). Recorded.

**How it was verified:** ruff + mypy(strict) clean; **162 tests** (bench CLI cold, step/PRM
cassette round-trips, seeded-Ollama payloads, compound-answer equivalence, and offline
reproduction of every §0 number). Before the GPU runs, a **multi-agent adversarial review** of
the diff surfaced 21 confirmed findings (silent fixture-corruption class: mixed-PRM pooling,
unvalidated `--n-values`, blind merge, unfiltered overlays, non-incremental capture save) — all
fixed and regression-tested. The incremental-save fix then paid off live: seed 2 TDR-crashed at
problem 37/40 and lost nothing (resumed from the recorded offset).

**Gotchas / honest notes:** seed pooling treats 40 problems × 3 seeds as 120 obs (matches
`sweep.py`; CIs mildly optimistic since seeds share problems). The token axis is raw tokens,
not FLOPs (flatters small models — the H2 negative holds anyway). Beam/MCTS are bounded by the
non-reasoning policy; a reasoning policy is the fair test of DVTS.

## The shiny artifact — a real GSM8K lift curve · 2026-07-12

**The headline the whole project was built for, on a real model.** `crucible.bench`
(generate-once, evaluate-at-many-N — avoids a 3-hour re-sampling marathon) captured **20
real GSM8K** problems × 8 samples via `qwen2.5:1.5b-instruct` (Ollama, GPU), each scored by
the real **Skywork-o1-Open-PRM-Qwen-2.5-1.5B** (GPU), in ~10 min, no TDR. The curve
(`docs/gsm8k-lift-curve.png`, `docs/RESULTS.md §0`):

| N | pass@1 | majority | PRM | oracle |
|---|---|---|---|---|
| 1 | 40% | 40% | 40% | 40% |
| 4 | — | 45% | **60%** | 75% |
| 8 | — | 50% | **60%** | **90%** |

Three real results: (1) **compute → accuracy** — pass@1 40% → oracle 90% at N=8 (a perfect
verifier more than doubles the frozen 1.5B); (2) **the learned PRM beats verifier-free
majority** (60% vs 50%) — the value of a step-reward model, which the earlier 8-problem
pilot was too small to show; (3) the **selection gap is honest** (oracle 90% > PRM 60%),
with the PRM's forward-pass tokens counted (its line sits ~2× right).

**Closes ROADMAP H3.** The run is captured to `tests/fixtures/gsm8k-bestofn.json` (traces +
PRM scores + correctness); `tests/test_cassette.py` replays it offline and reproduces every
number — the real curve regenerates in CI with no GPU. **118 tests** green. (H1 is now
substantially met — RESULTS.md leads with a real curve, synthetic demoted to mechanism
validation; the full ≥3-seed MATH-500 version is the remaining polish.)

## Real PRM working on the GPU + first real selection-gap · 2026-07-12

The "clearly human-owned" GPU/PRM blocker is **resolved** — the real learned verifier now
runs end-to-end on the dev machine (RTX 5070 Ti Laptop, 12 GB).

**Setup (the fiddly parts, now documented):**
- **Blackwell (sm_120) torch:** `pip install torch --index-url .../cu128` → torch
  2.11.0+cu128; `torch.cuda.get_device_capability() == (12, 0)`, GPU matmul confirmed. The
  Blackwell-wheel risk was real but is solved by the cu128 index.
- **transformers pinned `<5`:** the open PRMs' custom modeling code calls
  `DynamicCache.from_legacy_cache`, removed in transformers 5.x → downgraded to 4.46.3
  (capped in the `prm` extra).
- **`PRMVerifier` rewired to the real convention.** My M3 generic implementation was wrong
  (mean-pooled logits). The real Skywork-o1-Open-PRM (a `Qwen2ForRewardModel`) exposes a
  per-token value head (`v_head`); the correct scoring is `bos + problem + "\n"`, each
  step's trailing newline a reward position, `sigmoid(v_head(hidden))` read there. Probed
  it directly: a correct trace scores [0.92, 0.86, 0.68], a wrong one [0.27, 0.16, 0.02] —
  it flags the bad first step. 3.16 GB VRAM.

**First real result** — `crucible compare` on real GSM8K, `qwen2.5:1.5b-instruct` (Ollama,
100% GPU) + the real 1.5B Skywork PRM, 8 problems × N=8: **majority 62.5%, PRM 62.5%,
oracle 87.5%** (PRM tokens counted: 4,075 vs 2,591). Honest read: the stack works and the
selection gap is real (oracle − prm = 25%), but the **small PRM doesn't beat majority on
easy GSM8K** — the lift needs a stronger PRM / harder data (MATH-500), per DESIGN. Recorded
in `docs/RESULTS.md §0`. This executes the *mechanism* of ROADMAP H1's real-model variant;
the full multi-seed MATH-500 curve (H1) + the small-beats-big headline (H2) remain.

## H3 (partial) — Real-run cassettes: a live run replays offline · 2026-06-28

The standing pattern (run live once, commit a fixture, CI reproduces without a GPU) now
works for the **generation** side. `inference/cassette.py`: `RecordingPolicy` wraps any
backend and captures problems + traces to a self-contained JSON cassette; `CassettePolicy`
replays it with no network or model. `crucible run … --record <path>` writes one; the
runner saves it after the loop.

**Real captured artifact:** `crucible run --dataset gsm8k --policy ollama --model
qwen2.5:7b-instruct --limit 3 --temperature 0 --record tests/fixtures/gsm8k-m1.json`
recorded a live run — **qwen2.5:7b-instruct scored 3/3** on the first three GSM8K test
problems (greedy; gsm8k-2 now correct at 70000, 27 s warm, no TDR). `tests/test_cassette.py`
replays that 6.8 KB fixture **offline** and reproduces the 3/3 pass@1 — so this real number
regenerates in CI with no GPU. Also a mock record→replay round-trip test. **117 tests**.

**Still open on H3:** the *PRM* calls aren't cassetted (no GPU run yet), so the headline
best-of-N/beam **lift curve** doesn't yet regenerate from fixtures — that capture is a
single `--record` run on the GPU once the PRM is set up. Left unchecked in ROADMAP for that
reason; the mechanism + the M1 generation fixture are done.

## H4 — Adversarial hardening of the search core · 2026-06-28 · self-verified cold

An adversarial multi-agent stress test (9 skeptical finders over the search/verification
core, each finding re-checked by a refute-it verifier) found **8 real bugs** — and, tellingly,
**every one only manifests on the real-model path**, invisible to the synthetic suite (the
mocks always emit `\boxed`, terminate at a fixed depth, never exit early, never use a
zero-equivalent gold). One candidate ("carried-forward terminals re-scored") was correctly
**refuted** by the adversarial pass. All 8 fixed with regression tests (95 → **115 tests**).

**High severity (fixed):**
- **beam dropped completed answers** (`beam.py`): final selection was pure PRM-argmax, so a
  finished correct trace could lose to a higher-PRM *non-terminal* partial, and budget
  exhaustion returned a partial whose "answer" was a stray mid-reasoning number. Now tracks
  and returns the best *completed* trace.
- **prose answers broke terminal detection** (`answer_extract.py`): the `_ANSWER_PHRASE`
  regex required the literal text `boxe`, so "The answer is 42" never matched →
  `has_explicit_answer` missed prose → beam/MCTS never terminated early on real models
  (~15× compute inflation on the honesty axis). Fixed to require an `is/:/=` connector *or*
  `\boxed` (so a bare "answer" mid-sentence still isn't terminal).
- **sandbox clean-exit reward hack** (`code_sandbox.py`): a candidate exiting 0 before the
  asserts (`sys.exit(0)`, a `__main__` block) was scored as passing without running any test.
  Now requires an end-of-script sentinel *and* exit 0.

**Medium/low (fixed):** MCTS eager `evaluate(node)` in a `max(default=…)` burned a counted
PRM pass every expansion (now `if children else …`); sweeping a non-`_knob` compute field
merged distinct curve points (cell key is now a full config signature minus seed);
`wall_seconds` double-counted on Ollama runs (now set to true elapsed); the synthetic
`_distractor` "0" collided with a zero-equivalent gold under the symbolic verifier (now
picks a value the verifier doesn't equate).

**Verified:** ruff + mypy clean (38 files); **115 tests pass**; the synthetic sweep still
reads pass@1 11% → beam 100%, and MCTS is now slightly *more* efficient (the removed wasted
eval: 55.6%/88.9% at 2k/4k tokens vs 50%/77.8%). This closes **ROADMAP H4**. (H1–H3 remain —
they need the real GPU runs.)

## M7 — Compute-optimal & the results report · built 2026-06-28 · self-verified cold

The deliverable: the analysis layer that turns the search ladder into a *credible*
result, and the written report that interprets it honestly.

**What shipped:**
- **`analyze.py`**: `compute_optimal_frontier(cells)` (the upper-left envelope of
  accuracy vs total tokens — at each budget, the best method and accuracy; the Snell
  compute-optimal result) and `accuracy_by_difficulty(results)` (per-difficulty buckets,
  meaningful on graded datasets like MATH-500).
- **Multi-seed sweeps**: `seeds: [...]` runs each cell once per seed and **pools** the
  per-problem results (problems × seeds), so accuracy carries a tighter Wilson CI.
- **The frontier on the curve**: `render_curve` overlays the compute-optimal frontier as
  a dashed line; `report`/`sweep` print a frontier table (`tokens → best method → acc`).
- **`configs/results.yaml`**: the cold, 3-seed, full-ladder headline sweep.
- **`docs/RESULTS.md`**: the written report — the lift (pass@1 ~11% → 100% with search),
  the compute-optimal frontier (beam wins here), the PRM selection gap, the code track,
  threats to validity, and how to reproduce on real models. Honest up front that the
  numbers are from simulators, with the real-model recipe spelled out.

**How it was verified (cold):**
- `ruff` clean; `mypy src` clean (38 files); `pytest` → **95 passed** (5 new).
- `crucible sweep configs/results.yaml` (3 seeds) → `curve.png` with the dashed frontier
  and CIs; the frontier table reads pass1 (38 tok, 11%) → best_of_n (304, 17%) → beam
  (584/1112/2168 tok → 72/94/100%). Tests pin the frontier (non-dominated points only,
  monotone accuracy), per-difficulty bucketing, and multi-seed pooling (cell total =
  problems × seeds).

**Note:** this closes the roadmap's build phase. What's left is **real-model runs**:
the synthetic curves are mechanism checks; the same configs with `backend:
ollama` + a real `prm:` on MATH-500 produce the real artifact (RESULTS.md §Reproducing).

## M6 — MCTS over reasoning steps · built 2026-06-28 · self-verified cold

The top of the search ladder: PUCT Monte Carlo Tree Search over the step tree, valued by
the PRM (rStar-Math's recipe minus the training loop). Completes the ladder.

**What shipped:**
- **`MCTSStrategy`** (`search/mcts.py`): selection (PUCT: `Q + c·P·√N_parent/(1+N_child)`,
  uniform prior) → expansion (sample `beam_expansions` continuations via `sample_step`,
  evaluate **each** child with the PRM to seed its value) → backup of the best child
  value up the path. Terminal = an explicit `\boxed`/"answer is" marker. Budgeted by
  `budget_tokens` (total policy + verifier tokens) with an `mcts_max_sims` safety cap.
  Every step sampled and every PRM forward pass is counted.
- The search registry is now the **full ladder**: `pass1`, `best_of_n`, `beam`, `mcts`.
- `configs/beam-sweep.yaml` extended to plot all four methods; `sweep` knob shows the
  token budget for MCTS rows.

**How it was verified (cold, no model) — and the honest result:**
- `ruff` clean; `mypy src` clean (37 files); `pytest` → **91 passed** (4 new).
- The ladder curve: pass1 0%; **beam reaches 100% at ~1.1k tok, best-of-N at ~2.4k,
  MCTS at ~6k**. So on this *easy, shallow* synthetic task **MCTS is the most expensive**
  method — it saturates but slower than beam. This is reported honestly (not hidden): it
  matches DESIGN's framing ("MCTS: the most compute, the best on hard problems") — its
  adaptive-allocation edge over beam shows on deep/hard trees with rare good steps, a
  real-model phenomenon this toy doesn't reproduce. Tests assert the true invariants
  (needs a PRM, solves at sufficient budget, more budget → more tokens & not-worse
  accuracy, counts policy + verifier compute) — **not** an unearned "MCTS beats beam."

**Gotchas / notes:**
- First MCTS build evaluated only one child per expansion → too noisy (4/6). Fixed by
  evaluating every new child to seed its Q (expand-and-evaluate), which reliably reaches
  the all-good chain.
- MCTS always spends the full `budget_tokens` (no early stopping) — deliberately simple;
  an early-stop on PRM-confidence would make its curve more favourable but needs a
  PRM-scale threshold, deferred to avoid a magic number.

## M5 — Code track (sandboxed execution) · built 2026-06-28 · self-verified cold

The verifier abstraction generalizes from math to code — same `OutcomeVerifier` port,
"correct" now means *passing unit tests under execution* instead of symbolic equivalence.

**What shipped:**
- **`code_sandbox.run_in_sandbox`**: runs candidate + tests in an isolated subprocess
  (`python -I`), hard wall-clock timeout (kills on expiry), scratch temp dir as cwd,
  scrubbed env (proxy vars dropped), and an injected preamble that disables network
  (neuters `socket`) and caps CPU on POSIX. Guardrail, **not** a jail — see
  `docs/adr/0003-code-execution-sandbox.md`.
- **`CodeOutcomeVerifier`** (extracts the ```python block, runs the sandbox) behind the
  same port as the math verifier; **opt-in** (`allow_code_execution`). The runner
  **fails fast** on a code dataset unless `--allow-code-exec` is passed.
- **Datasets**: a bundled `code-sample` (3 problems + scripted solutions) and
  HuggingFace **HumanEval** / **MBPP** loaders (HumanEval `test` + `check(entry_point)`
  folded into the test tuple; MBPP `test_list`). Ollama uses a code-specific prompt for
  code datasets. `predicted` is reported as None for code (no math answer to extract).

**How it was verified (cold, no model):**
- `ruff` clean; `mypy src` clean (36 files); `pytest` → **87 passed** (15 new).
- Sandbox tests cover the security contract: correct code passes; wrong/raising/syntax-
  error code fails; an infinite loop **times out**; a `socket.create_connection` is
  **blocked**. `crucible run --dataset code-sample --policy mock --allow-code-exec` →
  **2/3** from real execution (c1, c2 pass; c3's buggy reverse fails); without the flag
  it refuses with a clear message (exit 1). HumanEval/MBPP row mappers tested on fixtures.

**Gotchas / notes:**
- True isolation needs Docker `--network none`/WSL2; the in-Python network/CPU guards can
  be bypassed by native code and Windows grandchildren may outlive the timeout (ADR-0003).
- The suite is slightly slower (~4s) because the sandbox tests spawn real subprocesses.
- best-of-N on code should use `--selection oracle` (run tests, pick a passer); majority
  voting on raw code isn't meaningful.

## M4 — Step segmentation + PRM-guided beam (DVTS) · built 2026-06-27 · self-verified cold

The first strategy that searches over *steps* rather than whole traces — and the first
to reproduce the headline test-time-search result (beam > best-of-N at matched compute).

**What shipped:**
- **`BeamStrategy`** (`search/beam.py`): each round, expand every non-terminal partial
  by `beam_expansions` continuations via `policy.sample_step`, score each partial with
  the PRM, keep the top `beam_width`; carry terminal partials forward; stop when the
  beam is all-terminal or `max_steps` is hit. Compute counts **every** policy step
  sampled and **every** PRM forward pass (its tokens on the honest axis).
- **A synthetic stepwise task** (`synthetic_stepwise.py`): `StepwisePolicy` (a D-step
  process, each step good w.p. `step_accuracy`, emits the gold answer only if *all*
  steps are good) + `StepRewardPRM` (scores partials by their step markers). This is the
  regime where step-level pruning wins, so beam's advantage is demonstrable cold.
- **`has_explicit_answer`** in `answer_extract` — a terminal check that looks for an
  actual `\boxed`/"answer is" marker, not the bare-number fallback.
- Config: `beam_width` / `beam_expansions` / `max_steps`, `step_accuracy` / `step_depth`
  / `step_prm_accuracy`; backends `stepwise` and PRM `step` wired into the runner.

**How it was verified (cold, no model):**
- `ruff` clean; `mypy src` clean (32 files); `pytest` → **72 passed** (9 new).
- `crucible sweep configs/beam-sweep.yaml` (5-step task, p=0.6/step) → on the curve,
  **beam reaches 100% at ~1.1k tokens/problem (width 2) while best-of-N needs ~2.4k
  (N=32)**, and pass1 is 0%. Beam sits above best-of-N at matched compute — the Snell
  result. Tests assert beam needs a PRM, solves the task (≥80%), out-scores best-of-N,
  and counts policy + verifier compute.

**Gotchas (a real bug caught here):**
- The first beam build mistook every partial for terminal because `extract_final_answer`'s
  number fallback matched "Step 2" — beam never expanded and scored 1/6 (only the problem
  whose gold is "1"). Fixed with the explicit-answer terminal check; **note for the real
  policy**: terminal detection keys on `\boxed`/"answer is", so the CoT prompt must elicit
  one (it does).
- width=1 beam is greedy and can get stuck (66.7% vs 100% for width≥2) — a nice
  illustration that beam needs width to be robust.

## M3 — PRM integration + the selection gap · built 2026-06-27 · self-verified cold

The learned verifier and the honest selection-gap comparison the whole project is
built to expose (DESIGN.md §4.4).

**What shipped:**
- **`ProcessVerifier` adapters** (`verify/process.py`): `MockProcessVerifier` (seeded,
  imperfect signal — correct traces score higher with noise, skill set by
  `prm_accuracy`) and `PRMVerifier` (real open PRM via `transformers`, lazy, `prm`
  extra; targets the Qwen-PRM convention, flagged to verify on first real run).
  `aggregate_scores` (mean/min/last/prod) reduces per-step scores.
- **Selectors** (`search/selectors.py`): `majority` / `oracle` / `prm`, each returning
  the chosen trace + its own selection compute. `best_of_n` was refactored to use them
  (so `--selection prm` works) — M2's curve path is unchanged.
- **PRM compute is counted**: `select_prm` adds a verifier forward call *and the trace's
  tokens* per candidate, so the PRM line costs more on the compute axis (the honesty
  point made concrete).
- **Same-samples comparison** (`runner.run_comparison`): generate N once per problem,
  then score with every selector — so majority/PRM/oracle differences are real, not a
  sampling artifact. New `crucible compare` command → a table + `comparison.png` bar
  chart; `crucible run`/sweeps gained `--prm`/`--prm-accuracy`.
- The synthetic policy now stamps each sample with an attempt nonce so candidates have
  distinct text (a PRM scores them individually rather than seeing N identical strings).

**How it was verified (cold, no model):**
- `ruff` clean; `mypy src` clean (30 files); `pytest` → **64 passed** (11 new).
- `crucible compare` (defaults: synthetic@0.3, mock-PRM@0.3, N=8) produces the textbook
  ordering **oracle 83% ≥ prm 67% ≥ majority 17%** with the PRM bar at 400 vs 200
  tokens/problem (its forward passes counted). Tests assert the structural invariants:
  oracle upper-bounds the others, a perfect PRM equals oracle, all selectors share
  identical generation compute, and PRM adds verifier tokens while majority adds none.

**Gotchas / notes:**
- best-of-N saturates: with many correct samples even a weak PRM ≈ oracle, so the *gap*
  only shows when correct samples are scarce (low policy accuracy / fewer N) — the demo
  defaults are tuned for that. Real gaps show on hard problems with a real PRM.
- The real `PRMVerifier` tensor wiring is model-specific and **unrun here** (no GPU);
  treat it like the Ollama adapter — structurally complete, confirm on first real use.
- The three-line *curve* (gap vs compute) and per-difficulty analysis are M7's report.

## M2 — Best-of-N + the accuracy-vs-compute curve · built 2026-06-27 · self-verified cold

The first **measured lift** and the project's headline artifact — the
accuracy-vs-compute curve — now exist, demonstrated end-to-end without a model.

**What shipped:**
- **`best_of_n` strategy** with two selectors: **majority** (vote on extracted answers,
  verifier-free) and **oracle** (first trace the outcome verifier passes — an upper
  bound that "cheats" with gold). The returned trace's `Compute` accounts for *all N*
  samples plus any selection-time verifier calls, so the curve's x-axis is honest.
- **`SyntheticPolicy`** — a seeded simulator of a policy with a known accuracy (correct
  trace w.p. `synthetic_accuracy`, else a distractor). Makes test-time scaling
  deterministic and analysable (pass@1 ≈ p; oracle@N ≈ 1-(1-p)^N) so the curve can be
  produced and unit-tested cold.
- **`sweep`** (`sweep.py`) — expands a base config + `grid:` (cartesian over list-valued
  fields like `n: [4,8,16]`) into runs, writes each record under one sweep dir, and
  aggregates `sweep.json`.
- **The curve** (`report.render_curve`) — matplotlib (Agg, headless) accuracy-vs-**total
  tokens/problem** (log x), one line per method/selector, with Wilson error bars →
  `curve.png`. `report` re-renders it from a sweep dir; `crucible run` gained
  `--selection` and `--synthetic-accuracy`.
- `configs/sample-sweep.yaml` — an offline lift-curve demo (synthetic backend).

**How it was verified (cold, no model):**
- `ruff` clean; `mypy src` clean (28 files); `pytest` → **53 passed** (10 new).
- `crucible sweep configs/sample-sweep.yaml` produces a valid 840×540 `curve.png`:
  pass@1 ≈ 83% (6-problem noise), oracle best-of-N → 100% by N=4, majority → 100% by
  N=32, tokens/problem scaling ~N× — i.e. accuracy rising with compute, the headline
  result. Selection logic + compute accounting + grid expansion are unit-tested; an
  end-to-end sweep test asserts oracle@16 ≥ pass@1 and uses more tokens.

**Gotchas / notes:**
- Matplotlib yerr must be non-negative; a clamped Wilson bound can land a hair past an
  accuracy of exactly 1.0, so `render_curve` clamps the error bars with `max(0, …)`.
- The demo uses a >50% synthetic policy so **majority** converges to the *correct*
  answer; with <50% it converges to the *wrong* one (self-consistency's known failure)
  — `oracle` is the reliable upper-bound line. Curves are noisy on only 6 problems.
- PRM-weighted selection (the third line) is M3; the majority-vs-oracle-vs-PRM
  comparison on the *same* samples will likely want a runner tweak then.

## M1 — Ollama backend + real pass@1 on GSM8K · built 2026-06-27 · ✓ confirmed 2026-06-28

**Live confirmation (2026-06-28):** `crucible run --method pass1 --dataset gsm8k --policy
ollama --model qwen2.5:7b-instruct --limit 3` → **2/3** on real GSM8K (gsm8k-0 18=18,
gsm8k-1 3=3, gsm8k-2 195000≠70000 — a genuine model miss). GSM8K downloaded via HF; the
answer extraction, `math-verify` equivalence, honest token counts (847), and Wilson CI all
worked on real output; no GPU TDR crash on the short run (~42 s/problem, incl. cold load).
A fuller `--limit 20` run for a tighter CI is left to the human (longer GPU run vs the
documented TDR risk). The `qwen2.5:7b-instruct` policy is a non-reasoning instruct model,
so the external-search lift stays visible (the -math variant is a drop-in upgrade).


The first real-model slice: the engine can now read GSM8K/MATH-500 from HuggingFace
and run pass@1 through a live Ollama server — no change to the search core, just two new
adapters (a dataset loader and the already-present `OllamaPolicy`, now wired with a
proper prompt).

**What shipped:**
- **`prompts.build_cot_prompt`** — a zero-shot math CoT prompt that asks for the answer
  in `\boxed{}` (designed in lockstep with the extractor).
- **`data/hf.py`** — GSM8K (`openai/gsm8k`, `main`/`test`; gold parsed from the
  `#### …` tail, commas/`$` stripped) and MATH-500 (`HuggingFaceH4/MATH-500`; LaTeX
  `answer`, `level` → difficulty). `datasets` is imported lazily; row→`Problem` mapping
  and gold extraction are pure functions. Wired into `data/registry.py`.
- **`OllamaPolicy`** now builds the CoT prompt, reads real token counts from Ollama's
  `eval_count` (whitespace approximation as fallback), and accepts an injectable
  `httpx.Client` for testing.
- **CLI** now fails *gracefully* on a down/unreachable backend (`httpx.HTTPError` →
  one-line message, exit 1) instead of dumping a traceback.

**How it was verified (without a live server):**
- `ruff` clean; `mypy src` clean (25 files); `pytest` → **43 passed** (10 new).
- `OllamaPolicy` is tested end-to-end against an **`httpx.MockTransport`**: correct
  `/api/generate` payload (model, `\boxed{}` prompt, options), response parsed into a
  `Trace` with `eval_count` tokens, and the resulting trace verified correct by the
  math verifier. GSM8K gold extraction + row mapping tested on fixtures.
- Live checks: `import datasets` (3.6.0) works; `crucible run --policy ollama` with no
  server prints the clean backend error; `--dataset humaneval` returns the M5 message.

**Still pending (the M1 acceptance test):** with Ollama running and a small
instruct model pulled, `crucible run --method pass1 --dataset gsm8k --policy ollama
--model <m> --limit 20` should print a real pass@1 + Wilson CI and write a record.

**Gotchas:** GSM8K/MATH-500 schemas are coded from their known HuggingFace layouts and
read defensively, but haven't been hit live here (no network) — worth a glance on the
first real run. `--dataset gsm8k --policy mock` yields 0% (no scripted outputs); that's
expected, mock is for the bundled `sample` set only.

## M0 — Skeleton & it runs · built 2026-06-27 · ✓ verified at scaffold

The full vertical spine, end-to-end, with **zero external dependencies** — a fresh
session can `pip install -e ".[dev]"` and immediately run and test it.

**What shipped:**
- Hexagonal core: `domain/types.py` (`Problem`, `Step`, `Trace`, `Compute`, `Result`,
  `Verdict`) and `domain/ports.py` (the `PolicyModel` / `OutcomeVerifier` /
  `ProcessVerifier` / `SearchStrategy` protocols). The search core depends only on
  these.
- **Compute accounting** as a first-class value type (`Compute`, additive, with
  `total_tokens` = policy + verifier) — the honesty layer from day one.
- A deterministic **`ScriptedPolicy`** mock backend, plus a real **`OllamaPolicy`**
  (httpx) written and ready to wire in M1.
- Math **answer extraction** (`\boxed{}`, "answer is …", number fallback) +
  **`math-verify`** outcome verifier with a SymPy/string fallback.
- A bundled **`sample`** dataset (6 GSM8K-style problems + canned outputs: 4 right, 2
  wrong) that exercises every path, including symbolic equivalence (`1/4 ≡ 0.25`).
- `runner.py` (problem × method → `Result`), `report.py` (JSON/CSV records + a rich
  console summary with **Wilson CIs**), and a Typer **CLI** (`run`/`report`/`version`;
  `sweep` stubbed with an M2 message).
- Project hygiene: `pyproject.toml` (light base deps; `datasets`/`prm`/`vllm`/`dev`
  extras), ruff + mypy(strict) config, GitHub Actions CI, `.env.example`, two ADRs.

**How it was verified (concrete evidence):**
- `ruff check .` → clean; `mypy src` → no issues in 23 files; `pytest` → **33 passed**.
- `python -m crucible run --method pass1 --dataset sample --policy mock` →
  **66.7% (4/6)** pass@1, Wilson CI [30.0%, 90.3%], 105 total tokens, 6 verifier calls;
  problem 6 (`\boxed{1/4}` vs gold `0.25`) correctly marked correct by symbolic
  equivalence; a record was written under `runs/`.
- `crucible report <run>` and `crucible run --config configs/sample.yaml` both work;
  `--method mcts` returns the milestone-aware error (exit 1).

**Gotchas for future sessions:**
- `math-verify` 0.9.0's built-in timeout uses signals/subprocesses that are fragile on
  Windows (and crash under `python -c`). We **disable** the timeout (`parsing_timeout=
  None`, `timeout_seconds=None`) and silence its warning logger in `math_outcome.py`.
  Inputs are short, so this is safe; revisit if very long expressions appear.
- Editable install must happen **after** `src/crucible/` exists, or the package path
  won't register (hit this once; `pip install -e .` again fixed it).
- Console output is kept ASCII (Windows cp1252 can't encode `✓`/`·`); `cli.main()` also
  reconfigures stdout to UTF-8 defensively.
- `runs/` is gitignored — run records are reproducible from `config.json`.
