# ROADMAP — Crucible

The milestone checklist. Standing instruction: **"continue"** → build the next
unchecked milestone.

**Rules of the road:**
- Each milestone is an **independently runnable** slice — something the human can
  actually test, not an internal-only refactor.
- Every milestone ends with explicit **Test** steps: what to do and what should
  happen. These are the acceptance criteria.
- Build **top-down**: a thin end-to-end slice first, then deepen each rung of the
  search ladder. Counts/scopes are budgets, not promises — split if one grows too big.
- Check a box **only after the human confirms its Test passes**, then add a
  `PROGRESS.md` entry.

> These mirror the ladder in [DESIGN.md §8](DESIGN.md). M0 here is the **offline
> skeleton** the scaffold delivered; the live-Ollama pass@1 that DESIGN folds into its
> "M0" is **M1** here. The headline deliverable is **M7** (the lift curve + report).

---

## Phase 0 — Walking skeleton

- [x] **M0 — Skeleton & it runs.** The vertical spine, end-to-end, with **zero
  external dependencies**: Typer CLI + YAML config; the ports/protocols and compute
  accounting; a deterministic **mock** policy; math answer-extraction + `math-verify`
  outcome check; a tiny bundled `sample` dataset; JSON/CSV run records with Wilson CIs;
  ruff + mypy(strict) + pytest wired up.
  **Test:** `pytest` → green (33 tests); `python -m crucible run --method pass1
  --dataset sample --policy mock` → prints a 4/6 (66.7%) pass@1 table, including the
  `1/4 ≡ 0.25` equivalence case, and writes a record under `runs/`. _(Verified at
  scaffold time.)_

## Phase 1 — Measured lift on math

- [ ] **M1 — Ollama backend + real pass@1 on GSM8K.** Wire the real `OllamaPolicy`
  into the run loop and load GSM8K via the HuggingFace `datasets` extra. Generate one
  CoT per problem, extract + verify, report pass@1 on ~10–50 problems.
  _(Built and unit-tested 2026-06-27 against a mocked transport; run the Test below to
  confirm against a live server, then check this box.)_
  **Test:** with Ollama running and a small instruct model pulled (e.g.
  `qwen2.5-math-1.5b-instruct`), `crucible run --method pass1 --dataset gsm8k
  --policy ollama --model <m> --limit 20` prints a pass@1 with a Wilson CI and writes
  a record. (Install: `pip install -e ".[datasets]"`.)

- [ ] **M2 — Best-of-N + the lift curve.** Add the `best_of_n` strategy (majority@N
  and oracle-best-of-N selection) and the `sweep`/`report` accuracy-vs-compute curve
  (pandas + matplotlib), plotting accuracy against **total tokens** (compute
  accounting made visible). Run on GSM8K + MATH-500.
  **Test:** `crucible sweep configs/lift-curve.yaml` (a small grid) then
  `crucible report <run>` renders a PNG curve where best-of-N@16 > pass@1, with the
  verifier's compute counted on the x-axis.

- [ ] **M3 — PRM integration (learned verifier).** Add a `ProcessVerifier` adapter for
  an open PRM (transformers/`prm` extra); **PRM-weighted best-of-N**; report
  PRM-selected vs majority vs oracle (exposes the PRM selection gap).
  **Test:** `crucible run --method best_of_n --prm <prm-id> ...` reports all three
  selection lines on MATH-500; oracle ≥ PRM ≥ majority is visible.

- [ ] **M4 — Step segmentation + beam/DVTS.** Use the step abstraction for real:
  PRM-guided beam search (Diverse Verifier Tree Search). Plot beam vs best-of-N at
  matched compute (expect beam to win on harder MATH-500).
  **Test:** `crucible run --method beam --beam-width 4 --dataset math500 ...` beats
  best-of-N at the same token budget on the hard subset.

## Phase 2 — Code, then the hardest search

- [ ] **M5 — Code track.** A sandboxed code-execution `OutcomeVerifier` (subprocess,
  hard timeout, no network, temp dir) + HumanEval/MBPP loaders; run best-of-N/beam
  with execution feedback. Proves the verifier abstraction generalizes math → code.
  **Test:** `crucible run --dataset humaneval --method best_of_n ...` reports pass@1
  and best-of-N from real test execution, with the sandbox enforced.

- [ ] **M6 — MCTS over reasoning steps.** UCT/PUCT with the PRM as value:
  selection / expansion / rollout / backup. Compare MCTS vs beam vs best-of-N at
  **matched compute** — the most expressive, most expensive method.
  **Test:** `crucible run --method mcts --budget-tokens 8000 --dataset math500 ...`
  matches or beats beam on the hard subset at equal total tokens.

## Phase 3 — The deliverable

- [ ] **M7 — Compute-optimal & the report.** Per-difficulty strategy selection
  (Snell-style), full accuracy-vs-compute curves with seeds + CIs, ablations (verifier
  on/off, PRM vs outcome, segmentation), and a written results report with the
  headline plot.
  **Test:** `crucible sweep configs/lift-curve.yaml && crucible report <run>` produces
  the final accuracy-vs-compute figure + tables; a short `docs/RESULTS.md` interprets
  the lift honestly.

---

**North star:** a credible **accuracy-vs-compute curve** showing search + verification
lifts a small open model well above its pass@1 on math (and code), with the verifier's
compute counted — the result is honest enough to trust.
