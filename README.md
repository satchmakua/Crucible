# Crucible

> A verifier-guided reasoning engine: generate many reasoning traces from a small
> open model, score them with a verifier (programmatic checker or a process-reward
> model), and **search** — best-of-N → beam → MCTS over reasoning steps — to turn
> test-time compute into measurable accuracy on math and code.

**Who it's for, and why this and not the obvious thing.** Crucible is for engineers and
researchers who need to *know* — not assume — what test-time search actually buys on a model
they control. The obvious alternative is to call a reasoning model and trust its benchmark
table, or to hand-roll a best-of-N script; the first tells you nothing about *your* policy,
and the second almost always reports a lift it didn't pay for. Crucible is the instrument
instead of the anecdote: it plots accuracy against **total tokens with the verifier's own
compute counted**, reports the **outcome** verifier on the chosen trace (never a PRM score),
puts Wilson CIs and a **named bigger-model baseline** next to every claim — and publishes the
results when search *loses*. It does lose here, twice; that's the point. Design rationale:
[DESIGN.md](DESIGN.md).

![Real MATH-500 accuracy-vs-compute curve](docs/math500-lift-curve.png)

**The headline — real model, real PRM, real data.** 3-seed MATH-500, frozen
`qwen2.5:1.5b-instruct` (Ollama) + a real **Skywork 1.5B PRM** (GPU): search lifts **pass@1
38.3% → 70% (oracle) at N=8**, and on *identical samples* the learned PRM beats
self-consistency majority at every N (N=4: **53.3% vs 45.0%**). The caveats, all measured, none
buried:

- the PRM's ~2× compute makes it **≈ a wash with majority at matched tokens**;
- a **7B baseline (67.5% pass@1 @ 524 tok) is more compute-efficient than 1.5B + search** —
  **small-beats-big fails** on this stack;
- real **beam (0/8)** and **MCTS (1/8)** on the hardest problems are the *most expensive*
  methods and **don't win**.

Full write-up, every number and caveat: **[docs/RESULTS.md §0](docs/RESULTS.md)**.

**Reproduce the headline in one command — no GPU, no network, no model.** It replays the
recorded real runs from committed cassettes:

```powershell
make demo   # no make on Windows? use the line below:
python -m crucible bench curve tests/fixtures/math500-bestofn-seed*.json
```

**Status:** M0–M7 built; the real curves are captured and replay in CI. Behind one
policy/verifier/search interface: **best-of-N** (M2), **PRM selection + the selection gap**
(M3), **PRM-guided beam/DVTS** (M4), a sandboxed **code track** (M5), **MCTS** (M6), the
**compute-optimal report** (M7), plus hardening (H1 ✓, H3 ✓, H4 ✓; **H2 measured — an honest
negative**). Open: a favorable real result for beam/MCTS (M4/M6) and small-beats-big (H2) both
need a stronger PRM / a reasoning policy; M5's real HumanEval run is still unrun. See
[ROADMAP.md](ROADMAP.md) · [PROGRESS.md](PROGRESS.md).

---

## Run it

**Prerequisites:** Python ≥ 3.11 (check: `python --version`). No GPU or network needed
for the M0 demo; real model backends (Ollama, etc.) come in from M1.

> **Naming:** the project is **Crucible**, distributed on PyPI as **`crucible-ttc`**
> (ttc = test-time compute), imported as `crucible`, and invoked as `crucible` on the
> command line (`pip install crucible-ttc` once published; the commands below install
> from source).

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
ollama pull qwen2.5:1.5b-instruct           # the captured runs' policy model
python -m crucible run --method pass1 --dataset gsm8k --policy ollama `
    --model qwen2.5:1.5b-instruct --limit 20
```

**The real lift curve (H1/H3):** capture it once on a GPU, regenerate it offline forever.
`bench record` samples N×/problem and scores each with the outcome verifier + PRM into a
cassette; `bench curve` computes accuracy-vs-compute for every N/selector from the cassette
(no GPU). Chunk long captures with `--offset`/`--limit` (a crash saves progress and prints
the resume offset), `merge` the chunks, and pool seeds by passing several cassettes:

```powershell
# One line per seed (needs the prm+datasets extras, Ollama, and a GPU):
crucible bench record --dataset math500 --model qwen2.5:1.5b-instruct `
    --prm Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B --max-n 8 --limit 40 --seed 0 --out runs/m500-s0.json
crucible bench curve runs/m500-s0.json runs/m500-s1.json runs/m500-s2.json   # pooled curve, offline

# Reproduce the committed headline with no GPU:
crucible bench curve tests/fixtures/math500-bestofn-seed*.json
```

**The offline lift-curve demo (M2):** the synthetic-policy curve, no model needed:

```powershell
python -m crucible sweep configs/sample-sweep.yaml   # writes runs/sweep-*/curve.png
```

**The PRM selection gap (M3):** compare majority / PRM / oracle selection on the *same*
best-of-N samples (offline, mock PRM):

```powershell
python -m crucible compare   # writes runs/compare-*/comparison.png (oracle >= prm >= majority)
```

**The full search ladder at matched compute (M4/M6):** pass1 vs best-of-N vs beam vs
MCTS on a synthetic stepwise task, offline:

```powershell
python -m crucible sweep configs/beam-sweep.yaml   # one curve, all four methods
```

**The headline report (M7):** the multi-seed ladder with Wilson CIs + the compute-optimal
frontier; interpreted in [docs/RESULTS.md](docs/RESULTS.md):

```powershell
python -m crucible sweep configs/results.yaml   # 3 seeds; curve.png has the dashed frontier
```

**The code track (M5):** execute model-generated code against unit tests in a
locked-down sandbox. Execution is **off by default** — pass `--allow-code-exec`:

```powershell
python -m crucible run --dataset code-sample --policy mock --allow-code-exec   # 2/3
```

### Commands

| Command | What it does |
|---|---|
| `crucible run [...]` | Run one experiment (method × dataset × backend) and report it |
| `crucible report <run_dir>` | Print the metrics from a past run |
| `crucible sweep <config.yaml>` | Grid → the accuracy-vs-compute curve (M2) |
| `crucible bench record/merge/curve` | Capture a real lift curve once (sample N×/problem, score offline), pool seeds, overlay beam/MCTS runs — the headline artifact (H1/H3) |
| `crucible compare` | Majority/PRM/oracle on the same samples → the selection gap (M3) |
| `crucible run --dataset code-sample --allow-code-exec` | Code track: sandboxed execution (M5) |
| `crucible run … --record <path>` | Record a live run to a cassette that replays offline in CI (H3) |
| `crucible version` | Print the version |
| `make demo` · `make check` | The offline real-results demo · lint + typecheck + test |
| `ruff check .` · `mypy src` · `pytest` | Lint · typecheck · tests |

---

## What Crucible can't do

An honest instrument publishes its limits, not just its wins:

- **It won't make a weak policy strong.** On MATH-500's hardest problems the 1.5B policy's
  pass@1 is **0%**; search lifts that to 12–17%, not to competence. A bigger model
  (7B pass@1 **67.5%**) beats 1.5B + search at a fraction of the compute — **small-beats-big
  does not hold here** (it needs a stronger PRM than the 1.5B Skywork).
- **Stepwise search (beam/MCTS) doesn't help a non-reasoning instruct policy.** Asked to
  *continue* a partial trace, `qwen2.5:1.5b-instruct` **restarts** instead — so beam scores
  0/8 at ~37k tokens/problem. Beam/DVTS is a reasoning-policy phenomenon; this stack can't
  show its win.
- **No training.** Policies and PRMs are frozen — no fine-tuning, RL, or self-improvement
  (an explicit v1 non-goal).
- **The code sandbox is a guardrail, not a jail** (subprocess + timeout + no network + scratch
  dir; opt-in). Use Docker/WSL2 for genuinely untrusted code. See
  [ADR-0003](docs/adr/0003-code-execution-sandbox.md).
- **Single node, one GPU.** No multi-GPU or distributed orchestration.
- **Statistics are modest.** Curves pool 40 problems × 3 seeds; the seeds share problems, so
  CIs are mildly optimistic and close margins are suggestive, not decisive. The compute axis
  counts **tokens, not FLOPs** (which flatters small models). Full list:
  [RESULTS §6](docs/RESULTS.md).

`crucible` and `python -m crucible` are equivalent. Optional extras install per
milestone: `".[datasets]"` (M1), `".[prm]"` (M3), `".[vllm]"`.

---

## How to give feedback

When reporting an issue or a run:

- Run the **Test** steps for the current milestone in [ROADMAP.md](ROADMAP.md).
- Describe what happened in plain language; paste any errors verbatim (the single most
  useful thing); include the printed metrics table for a run.

---

## Project docs

| Doc | What's in it |
|---|---|
| [DESIGN.md](DESIGN.md) | The full design and rationale — the single source of truth. |
| [docs/RESULTS.md](docs/RESULTS.md) | The results report — the lift curve, interpreted honestly. |
| [ROADMAP.md](ROADMAP.md) | The milestone checklist (the plan + what's done). |
| [PROGRESS.md](PROGRESS.md) | Build log: what shipped each milestone and why. |
| [`docs/`](docs/) | Long-form docs and architecture decisions (ADRs). |

## Tech stack

Python 3.11+ · **Typer** CLI over **YAML** config · ports-and-adapters core ·
**math-verify** + SymPy (math equivalence) · **httpx** (Ollama backend) ·
pandas + matplotlib (reporting) · pytest · ruff · mypy(strict). Inference backends are
swappable adapters behind one port — `mock`/`synthetic`/`stepwise` (offline, drive the
tests) and **`ollama`** (the real backend used for every captured result); a vLLM/hosted
adapter slots in the same way. PRM scoring (**transformers** + torch) and the HuggingFace
dataset loaders live behind the `prm` and `datasets` extras, so the base install stays
light and the whole test suite runs with no GPU.

## License

MIT — see [LICENSE](LICENSE).
