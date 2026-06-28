# PROGRESS — Crucible

A build log of what shipped and the notable decisions behind it. **Keep it honest** —
this is the working memory between build sessions. The forward-looking plan and
acceptance tests live in [ROADMAP.md](ROADMAP.md); this is the backward-looking
"what got done and why" companion.

**Current phase:** M1–M5 built (M1 awaits a live-Ollama test; M2–M5 self-verified cold).
Next up: **M6** (MCTS over reasoning steps — the headline method).

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
| Ollama policy | `inference/ollama.py` | ✅ M1 |
| Math CoT prompt builder | `prompts.py` | ✅ M1 |
| Answer extraction | `verify/answer_extract.py` | ✅ M0 |
| Math outcome verifier | `verify/math_outcome.py` | ✅ M0 |
| PRM: mock + real adapters | `verify/process.py` | ✅ M3 (real PRM unrun) |
| Code sandbox + outcome verifier | `verify/code_sandbox.py`, `verify/code_outcome.py` | ✅ M5 (opt-in) |
| pass1 strategy + registry | `search/` | ✅ M0 |
| best_of_n + selectors (maj/oracle/prm) | `search/best_of_n.py`, `search/selectors.py` | ✅ M3 |
| beam / DVTS | `search/beam.py` | ✅ M4 |
| mcts | `search/` | ⬜ M6 |
| Sample dataset + registry | `data/` | ✅ M0 |
| GSM8K + MATH-500 loaders | `data/hf.py` | ✅ M1 |
| Code datasets: bundled + HumanEval/MBPP | `data/code_sample.py`, `data/hf.py` | ✅ M5 |
| Experiment runner | `runner.py` | ✅ M0 |
| Run records (JSON/CSV) + summary | `report.py` | ✅ M0 |
| Sweep + accuracy-vs-compute curve | `sweep.py`, `report.py` | ✅ M2 |
| Selection-gap comparison + bar chart | `runner.run_comparison`, `report.py` | ✅ M3 |
| CLI (run/report/sweep/compare/version) | `cli.py` | ✅ M3 (all real) |

---

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

## M1 — Ollama backend + real pass@1 on GSM8K · built 2026-06-27 · awaiting test

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
