# ROADMAP — Crucible

The milestone checklist. Build the next unchecked milestone in order.

**Rules of the road:**
- Each milestone is an **independently runnable** slice — something actually
  testable end-to-end, not an internal-only refactor.
- Every milestone ends with explicit **Test** steps: what to do and what should
  happen. These are the acceptance criteria.
- Build **top-down**: a thin end-to-end slice first, then deepen each rung of the
  search ladder. Counts/scopes are budgets, not promises — split if one grows too big.
- Check a box **only after its Test passes**, then add a
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

- [x] **M1 — Ollama backend + real pass@1 on GSM8K.** Wire the real `OllamaPolicy`
  into the run loop and load GSM8K via the HuggingFace `datasets` extra. Generate one
  CoT per problem, extract + verify, report pass@1 on ~10–50 problems.
  _(✓ Confirmed 2026-06-28 on live Ollama — `qwen2.5:7b-instruct` scored 2/3 on real
  GSM8K with working extraction + math-verify + compute accounting. A larger `--limit`
  run for a tighter CI is optional; the path is proven.)_
  **Test:** with Ollama running and a small instruct model pulled (e.g.
  `qwen2.5-math-1.5b-instruct`), `crucible run --method pass1 --dataset gsm8k
  --policy ollama --model <m> --limit 20` prints a pass@1 with a Wilson CI and writes
  a record. (Install: `pip install -e ".[datasets]"`.)

- [x] **M2 — Best-of-N + the lift curve.** Add the `best_of_n` strategy (majority@N
  and oracle-best-of-N selection) and the `sweep`/`report` accuracy-vs-compute curve
  (matplotlib), plotting accuracy against **total tokens** (compute accounting made
  visible).
  _(✓ real 2026-07-14: the headline 3-seed MATH-500 best-of-N curve — pass@1 38.3% →
  oracle@8 70% — is captured via `crucible bench` and reproduces offline. Also self-verified
  cold on the synthetic backend.)_
  **Test (offline, runs cold):** `crucible sweep configs/sample-sweep.yaml` then
  `crucible report <sweep_dir>` → a `curve.png` where best-of-N (oracle, and majority
  for a >50% policy) rises above pass@1, with tokens on the x-axis. **Real-model
  variant:** point a sweep at `dataset: gsm8k`, `backend: ollama` (needs M1's setup).

- [x] **M3 — PRM integration (learned verifier).** Add a `ProcessVerifier` adapter for
  an open PRM (transformers/`prm` extra); **PRM-weighted best-of-N**; the `compare`
  path scores majority / PRM / oracle on the *same* samples to expose the selection gap.
  _(✓ real 2026-07-14 with the Skywork 1.5B PRM on the GPU: on identical MATH-500 samples,
  oracle (70%) ≥ prm (55%) ≥ majority (52.5%) at N=8, PRM's forward-pass compute counted;
  the selection gap is large and honest (RESULTS §0). Also self-verified cold with a mock PRM.)_
  **Test (offline, runs cold):** `crucible compare` → a table + `comparison.png` where
  **oracle ≥ prm ≥ majority** and the prm bar costs more tokens/problem (its forward
  passes counted). **Real-model variant:** `crucible compare --policy ollama --model
  <m> --dataset gsm8k --prm <qwen-prm-id>` (needs the `prm` extra + a GPU).

- [ ] **M4 — Step segmentation + beam/DVTS.** PRM-guided beam search over the step port
  (`sample_step`): expand top-k partials, score with the PRM, prune; plot beam vs
  best-of-N at matched compute.
  _(Built and self-verified 2026-06-27 on a synthetic stepwise task — the curve shows
  beam reaching 100% at ~half the tokens best-of-N needs. **Real contact 2026-07-14:** a
  recorded PRM-guided beam ran on the 8 hardest MATH-500 problems and **replays offline** —
  but the honest result is a **negative**: 0/8 at ~37k tokens/problem, because the
  non-reasoning 1.5B instruct policy *restarts* rather than continues a partial trace, so
  stepwise search can't assemble a chain (RESULTS §0.1). The beam>best-of-N crossover is a
  reasoning-policy phenomenon; box stays open until it's shown on real data.)_
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
  selection / expansion / evaluation / backup over the step tree, budgeted by total
  tokens. The full search ladder, plotted together.
  _(Built and self-verified 2026-06-28 — MCTS saturates the synthetic stepwise task; on
  this easy/shallow task it's the **most expensive** of the tree methods. **Real contact
  2026-07-14:** a recorded, budget-capped MCTS ran on the same 8 hardest MATH-500 problems
  and **replays offline** — 1/8 (12.5%) at ~12k tokens/problem: cheaper than beam (the budget
  bites) and it solves one, but still below best-of-N and not on the frontier (RESULTS §0.1).
  Its edge over beam/best-of-N is a hard-problem/reasoning-policy phenomenon; box stays open
  until it's shown on real data.)_
  **Test (offline, runs cold):** `crucible sweep configs/beam-sweep.yaml` plots all four
  methods; MCTS reaches 100% (at higher compute than beam). **Real-model variant:**
  `crucible run --method mcts --budget-tokens 8000 --dataset math500 --policy ollama
  --prm <qwen-prm>` should match or beat beam on the *hard* subset at equal tokens.

## Phase 3 — The deliverable

- [x] **M7 — Compute-optimal & the report.** The **compute-optimal frontier** (best
  method at each budget — Snell-style), multi-seed curves with Wilson CIs, per-difficulty
  analysis, and a written results report with the headline plot.
  _(✓ real 2026-07-14 — `docs/RESULTS.md §0` leads with the real 3-seed MATH-500
  accuracy-vs-compute curve + compute-optimal frontier + Wilson CIs, with beam/MCTS overlaid
  on the hard subset and the 7B baseline; the synthetic ladder is demoted to mechanism
  validation. Also self-verified cold — `analyze.compute_optimal_frontier`, multi-seed pooled
  sweeps, the frontier overlay.)_
  **Test (offline, runs cold):** `crucible sweep configs/results.yaml` (3 seeds, full
  ladder) → `curve.png` with the dashed compute-optimal frontier + a frontier table;
  [`docs/RESULTS.md`](docs/RESULTS.md) interprets the lift honestly (pass@1 ~11% → 100%
  with search; beam compute-optimal here; MCTS honestly the most expensive). **Real
  artifact:** the same sweep with `backend: ollama` + a real `prm:` on MATH-500.

---

**North star:** a credible **accuracy-vs-compute curve** showing search + verification
lifts a small open model well above its pass@1 on math (and code), with the verifier's
compute counted — the result is honest enough to trust.

---

## Review-driven hardening — from *built* to *proven* (added 2026-06-28)

> Added after an external code review (captured in `../ai-docs/project_eval/`). The
> ladder (M0–M7) is built and the **real-model variants are already written into the
> Test steps above** — they are simply unchecked. This project is a *measurement
> instrument that has not yet taken a real measurement*; these items make executing it
> the priority and define the artifact. **Standing rule:** a milestone is checked only
> when it has produced **one real, captured, reproducible artifact**, not a synthetic one.

**Definition of Done — the "Sparkle Bar"** (applies to every milestone):
1. **Real artifact captured** — produced against a real model/PRM/dataset, pinned at the top of the README with the exact reproduce command.
2. **Flagship demo in one screen** — the real accuracy-vs-compute curve + frontier.
3. **Stress-tested** — adversarial tests on the search strategies, not just happy-path units.
4. **Honest numbers** — Wilson CIs, a named baseline, an explicit "can't do" list.
5. **Cold-clone reproducible** — pinned deps, fixed seeds, one `make demo`, CI runs the real-or-recorded path.
6. **Polished** — no stray files, consistent docs, README opens with the artifact.
7. **Positioned** — one paragraph: who it's for, what it beats, why this not the obvious alternative.

**Hardening items (Crucible-specific):**
- [x] **H1 — Execute the real-model variants; lead with them.** _(✓ 2026-07-14.)_ Ran the
  real-model variants on Ollama `qwen2.5:1.5b-instruct` + the real **Skywork 1.5B PRM** on
  **MATH-500, 3 seeds** (problems 0–39). `docs/RESULTS.md §0` and the README now **lead with
  the real 3-seed MATH-500 accuracy-vs-compute curve** (Wilson CIs, frontier); synthetic
  demoted to mechanism validation. Result (honest): pass@1 38.3% → oracle@8 70%; on identical
  samples the PRM beats majority at every N (N=4: 53.3% vs 45.0%), though its ~2× compute
  makes it ~a wash with majority at matched tokens. Captured to `tests/fixtures/math500-*`,
  reproduced offline by `tests/test_cassette.py`. Delivered via the new `crucible bench` verb.
- [ ] **H2 — The "small-beats-big" headline.** _(Measured 2026-07-14 — honest **negative** on
  this stack.)_ Ran `qwen2.5:7b-instruct` pass@1 on the same 40 MATH-500 problems: **67.5% at
  ~524 tok/problem**. The 1.5B + PRM search needs ~8k tokens (16×) to reach 55% and only the
  *oracle cheat* matches the 7B — so **small does not beat big here** (RESULTS §0.2). The
  positive claim awaits a *stronger* selector (the family-matched Qwen 7B PRM); the negative
  is captured and reproducible (`tests/fixtures/math500-7b-pass1.json`). Box stays open until
  the positive result is demonstrated.
- [x] **H3 — Record real runs as fixtures.** _(✓ 2026-07-12.)_ Both the model *and* the PRM
  are cassetted: `RecordingPolicy`/`CassettePolicy` + `--record` (generation), and
  `crucible.bench` records traces + PRM scores + correctness to `tests/fixtures/`. The real
  **GSM8K lift curve** (`docs/gsm8k-lift-curve.png`) regenerates **offline in CI** from
  `gsm8k-bestofn.json` — `tests/test_cassette.py` reproduces every number with no GPU.
  *Accept met.*
- [x] **H4 — Stress the search strategies.** _(✓ 2026-06-28.)_ An adversarial multi-agent
  audit of the search/verification core found **8 real bugs** (3 high) that only manifest on
  the real-model path — all fixed with regression tests. Degenerate cases now covered:
  width-1/greedy beams, single-candidate expansion, MCTS budget exhaustion, tied/empty PRM
  scores (`tests/test_search_degenerate.py`). *Accept met:* 115 tests pass; see PROGRESS.
