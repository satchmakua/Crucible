# PROGRESS — Crucible

A build log of what shipped and the notable decisions behind it. **Keep it honest** —
this is the working memory between build sessions. The forward-looking plan and
acceptance tests live in [ROADMAP.md](ROADMAP.md); this is the backward-looking
"what got done and why" companion.

**Current phase:** M1 built (awaiting the human's live-Ollama test). Next up: **M2**
(best-of-N + the accuracy-vs-compute curve).

## State of the tree

| Component | Where | Status |
|---|---|---|
| Value types + ports (the hexagon) | `domain/types.py`, `domain/ports.py` | ✅ M0 |
| Compute accounting | `domain/types.py::Compute` | ✅ M0 |
| Config (YAML ↔ dataclass) | `config.py` | ✅ M0 |
| Step segmentation + token approx | `segment.py` | ✅ M0 (used fully in M4) |
| Wilson CIs | `stats.py` | ✅ M0 |
| Mock policy (ScriptedPolicy) | `inference/mock.py` | ✅ M0 |
| Ollama policy | `inference/ollama.py` | ✅ M1 |
| Math CoT prompt builder | `prompts.py` | ✅ M1 |
| Answer extraction | `verify/answer_extract.py` | ✅ M0 |
| Math outcome verifier | `verify/math_outcome.py` | ✅ M0 |
| PRM (process verifier) | `verify/` | ⬜ M3 |
| Code-execution verifier | `verify/` | ⬜ M5 |
| pass1 strategy + registry | `search/` | ✅ M0 |
| best_of_n / beam / mcts | `search/` | ⬜ M2 / M4 / M6 |
| Sample dataset + registry | `data/` | ✅ M0 |
| GSM8K + MATH-500 loaders | `data/hf.py` | ✅ M1 |
| Code dataset loaders (HumanEval/MBPP) | `data/` | ⬜ M5 |
| Experiment runner | `runner.py` | ✅ M0 |
| Run records (JSON/CSV) + summary | `report.py` | ✅ M0 |
| Accuracy-vs-compute curve | `report.py` | ⬜ M2 |
| CLI (run/report/sweep/version) | `cli.py` | ✅ M0 (sweep stubbed → M2) |

---

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

**Still needs the human (the M1 acceptance test):** with Ollama running and a small
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
