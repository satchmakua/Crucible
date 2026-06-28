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
  (matplotlib), plotting accuracy against **total tokens** (compute accounting made
  visible).
  _(Built and self-verified 2026-06-27 on the offline synthetic backend — `curve.png`
  shows best-of-N rising above pass@1 with compute on the x-axis; review it / run a
  real-model sweep, then check this box.)_
  **Test (offline, runs cold):** `crucible sweep configs/sample-sweep.yaml` then
  `crucible report <sweep_dir>` → a `curve.png` where best-of-N (oracle, and majority
  for a >50% policy) rises above pass@1, with tokens on the x-axis. **Real-model
  variant:** point a sweep at `dataset: gsm8k`, `backend: ollama` (needs M1's setup).

- [ ] **M3 — PRM integration (learned verifier).** Add a `ProcessVerifier` adapter for
  an open PRM (transformers/`prm` extra); **PRM-weighted best-of-N**; the `compare`
  path scores majority / PRM / oracle on the *same* samples to expose the selection gap.
  _(Built and self-verified 2026-06-27 with a mock PRM — `comparison.png` shows
  oracle ≥ prm ≥ majority with PRM compute counted; run a real PRM, then check this box.)_
  **Test (offline, runs cold):** `crucible compare` → a table + `comparison.png` where
  **oracle ≥ prm ≥ majority** and the prm bar costs more tokens/problem (its forward
  passes counted). **Real-model variant:** `crucible compare --policy ollama --model
  <m> --dataset gsm8k --prm <qwen-prm-id>` (needs the `prm` extra + a GPU).

- [ ] **M4 — Step segmentation + beam/DVTS.** PRM-guided beam search over the step port
  (`sample_step`): expand top-k partials, score with the PRM, prune; plot beam vs
  best-of-N at matched compute.
  _(Built and self-verified 2026-06-27 on a synthetic stepwise task — the curve shows
  beam reaching 100% at ~half the tokens best-of-N needs; run a real-model beam, then
  check this box.)_
  **Test (offline, runs cold):** `crucible sweep configs/beam-sweep.yaml` → a `curve.png`
  where the **beam** line sits above **best_of_n** at matched tokens (beam hits 100% at
  ~1.1k tok/problem vs ~2.4k for best-of-N; pass1 ≈ 0%). **Real-model variant:**
  `crucible run --method beam --beam-width 4 --dataset math500 --policy ollama --prm
  <qwen-prm>` should beat best-of-N at the same token budget on the hard subset.

## Phase 2 — Code, then the hardest search

- [ ] **M5 — Code track.** A sandboxed code-execution `OutcomeVerifier` (isolated
  subprocess, hard timeout, no network, scratch temp dir; **opt-in**) + HumanEval/MBPP
  loaders; run best-of-N with execution feedback. Proves the verifier abstraction
  generalizes math → code (see `docs/adr/0003-…`).
  _(Built and self-verified 2026-06-28 — the sandbox passes good code and contains
  wrong/raising/looping/network-touching code; the gate blocks code datasets unless
  `--allow-code-exec`. Run real HumanEval, then check this box.)_
  **Test (offline, runs cold):** `crucible run --dataset code-sample --policy mock
  --allow-code-exec` → pass@1 = 2/3 from real execution (and without the flag it
  refuses, exit 1). **Real-model variant:** `crucible run --dataset humaneval --method
  best_of_n --selection oracle --policy ollama --model <m> --allow-code-exec` (needs the
  `datasets` extra) reports pass@1 + best-of-N from real test execution.

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
