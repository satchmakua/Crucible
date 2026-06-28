# Crucible

> A verifier-guided reasoning engine: generate many reasoning traces from a small
> open model, score them with a verifier (programmatic checker or a process-reward
> model), and **search** — best-of-N → beam → MCTS over reasoning steps — to turn
> test-time compute into measurable accuracy on math and code.

Most people only *consume* reasoning models; Crucible builds the machinery underneath
and **measures the lift** — accuracy as a function of test-time compute over a small
open policy model. The full design and rationale live in [DESIGN.md](DESIGN.md).

**Status:** **M0 shipped; M1–M3 built.** The engine runs end-to-end offline, runs real
pass@1 on GSM8K/MATH-500 via a local Ollama server (M1, awaiting a live test), produces
the headline **accuracy-vs-compute curve** for best-of-N (M2), and compares
majority/PRM/oracle selection on the same samples to expose the **PRM selection gap**
(M3) — M2 and M3 self-verified on synthetic backends. See [ROADMAP.md](ROADMAP.md) and
[PROGRESS.md](PROGRESS.md). Next: M4 (step segmentation + PRM-guided beam search).

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

**The lift curve (M2):** produce the headline accuracy-vs-compute curve offline (no
model needed — it uses a synthetic policy):

```powershell
python -m crucible sweep configs/sample-sweep.yaml   # writes runs/sweep-*/curve.png
```

**The PRM selection gap (M3):** compare majority / PRM / oracle selection on the *same*
best-of-N samples (offline, mock PRM):

```powershell
python -m crucible compare   # writes runs/compare-*/comparison.png (oracle >= prm >= majority)
```

### Commands

| Command | What it does |
|---|---|
| `crucible run [...]` | Run one experiment (method × dataset × backend) and report it |
| `crucible report <run_dir>` | Print the metrics from a past run |
| `crucible sweep <config.yaml>` | Grid → the accuracy-vs-compute curve (M2) |
| `crucible compare` | Majority/PRM/oracle on the same samples → the selection gap (M3) |
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
