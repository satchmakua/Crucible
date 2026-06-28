# Crucible

> A verifier-guided reasoning engine: generate many reasoning traces from a small
> open model, score them with a verifier (programmatic checker or a process-reward
> model), and **search** — best-of-N → beam → MCTS over reasoning steps — to turn
> test-time compute into measurable accuracy on math and code.

Most people only *consume* reasoning models; Crucible builds the machinery underneath
and **measures the lift** — accuracy as a function of test-time compute over a small
open policy model. The full design and rationale live in [DESIGN.md](DESIGN.md).

**Status:** **M0 shipped; M1 built (awaiting a live-Ollama test).** The engine runs
end-to-end offline on a bundled sample, and can now run real pass@1 on GSM8K/MATH-500
through a local Ollama server. See [ROADMAP.md](ROADMAP.md) for the plan and
[PROGRESS.md](PROGRESS.md) for what's done. Next: M2 (best-of-N + the lift curve).

---

## Run it

**Prerequisites:** Python ≥ 3.11 (check: `python --version`). No GPU or network needed
for the M0 demo; real model backends (Ollama, etc.) come in from M1.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# Offline demo — generate, verify, and report pass@1 on the bundled sample set:
python -m crucible run --method pass1 --dataset sample --policy mock
```

You should see a per-problem table and a **66.7% (4/6)** pass@1 with a Wilson
confidence interval, and a run record written under `runs/`.

**Real model runs (M1):** install the dataset extra, start [Ollama](https://ollama.com)
and pull a small instruct model, then run real pass@1 on GSM8K:

```powershell
pip install -e ".[datasets]"
ollama pull qwen2.5-math-1.5b-instruct      # or any small instruct model
python -m crucible run --method pass1 --dataset gsm8k --policy ollama `
    --model qwen2.5-math-1.5b-instruct --limit 20
```

### Commands

| Command | What it does |
|---|---|
| `crucible run [...]` | Run one experiment (method × dataset × backend) and report it |
| `crucible report <run_dir>` | Print the metrics from a past run |
| `crucible sweep <config.yaml>` | Grid → the accuracy-vs-compute curve _(lands in M2)_ |
| `crucible version` | Print the version |
| `ruff check .` · `mypy src` · `pytest` | Lint · typecheck · tests |

`crucible` and `python -m crucible` are equivalent. Optional extras install per
milestone: `".[datasets]"` (M1), `".[prm]"` (M3), `".[vllm]"`.

---

## How to give feedback

You mainly **test and report**:

- Run the **Test** steps for the current milestone in [ROADMAP.md](ROADMAP.md).
- Describe what happened in plain language; paste any errors verbatim (the single most
  useful thing); include the printed metrics table for a run.

---

## Project docs

| Doc | What's in it |
|---|---|
| [DESIGN.md](DESIGN.md) | The full design and rationale — the single source of truth. |
| [ROADMAP.md](ROADMAP.md) | The milestone checklist (the plan + what's done). |
| [PROGRESS.md](PROGRESS.md) | Build log: what shipped each milestone and why. |
| [CLAUDE.md](CLAUDE.md) | Standing instructions for the AI build loop. |
| [`docs/`](docs/) | Long-form docs and architecture decisions (ADRs). |

## Tech stack

Python 3.11+ · **Typer** CLI over **YAML** config · ports-and-adapters core ·
**math-verify** + SymPy (math equivalence) · **httpx** (Ollama/hosted backends) ·
pandas + matplotlib (reporting) · pytest · ruff · mypy. Inference backends are
swappable adapters (mock now; Ollama / vLLM / hosted); PRM scoring via transformers
and datasets via HuggingFace arrive behind extras as the milestones need them.

## License

MIT — see [LICENSE](LICENSE).
