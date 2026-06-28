# Crucible — Design

> A verifier-guided reasoning engine: generate many reasoning traces from a small
> open model, score them with a verifier (programmatic checker or a process-reward
> model), and **search** — best-of-N → beam → MCTS over reasoning steps — to turn
> test-time compute into measurable accuracy.

**Status:** Design draft · **Language:** Python · **Stack target:** local-first
(consumer GPU or CPU), cloud-GPU optional · **Kind:** research demonstrator + harness

Name: **Crucible** (reasoning traces tested and refined under search). Alternates:
**Quorum**, **Lodestar**, **Sieve**, **Verdict**.

> **The one assumption to confirm (drives model size + how heavy a search is
> practical):** I've designed for **local-first** execution with a *swappable
> inference backend* — default **Ollama** (native Windows, matches your
> `llm-eval-harness`), with **vLLM** (WSL2 or a rented cloud GPU) for high-throughput
> sampling and an **OpenAI-compatible hosted** adapter as a third option. On CPU-only,
> use 1.5B models + lighter search; a single 8–24 GB NVIDIA GPU makes 7B policy + 7B
> PRM + MCTS comfortable. If your target differs, it's a config change, not a rewrite.

---

## 1. Concept

Reasoning models are the moment, but most people only *consume* them. Crucible builds
the machinery underneath: it treats a frozen small model as a **policy** that proposes
reasoning, and a **verifier** that judges it, and spends inference-time compute
*searching* the space of reasoning for answers the verifier trusts. The headline
deliverable is not a leaderboard number — it's a **curve**: accuracy as a function of
test-time compute, showing that search + verification lifts a small open model well
above its single-shot (pass@1) accuracy on math and code, and *why*.

It implements the full ladder of test-time search, simplest first:

1. **Self-consistency / majority@N** — the verifier-free control (sample N, vote).
2. **Best-of-N** — sample N independent traces, pick the one the verifier likes best
   (outcome verifier, or PRM-weighted).
3. **Beam search over steps** — expand promising partial reasoning, prune with a
   process-reward model (PRM) step by step.
4. **MCTS over reasoning steps** — selection/expansion/rollout/backup with the PRM as
   value, the most expressive (and expensive) search.

This is exactly the recipe the literature has converged on (see §10): for small
(<7B) policies, search against a verifier beats best-of-N on hard problems, and
compute-optimal allocation can be **>4× more efficient** than best-of-N
(Snell et al. 2024); MCTS + a process-preference model lifted Qwen2.5-Math-7B from
**58.8% → 90.0%** on MATH (rStar-Math, 2025). Crucible reproduces the *mechanism* on
consumer hardware and measures it honestly.

**The three engineering pillars** (the parts that make or break it):

1. **Trustworthy verification** — outcome checks that don't reward right-answer /
   wrong-reasoning, a safe code sandbox, and a calibrated PRM. The verifier is the
   ceiling on everything.
2. **A unified step-segmented generator + search controller** — one interface that
   best-of-N, beam, and MCTS all drive, over a clean "reasoning step" abstraction.
3. **Honest measurement** — matched-compute accuracy-vs-compute curves that *count
   the verifier's compute too*, with seeds and confidence intervals. Most hobby
   versions die here.

## 2. Goals / Non-goals

**Goals (v1)**
- A working engine implementing majority@N, best-of-N, PRM-weighted best-of-N,
  PRM-guided beam search, and MCTS over reasoning steps — sharing one
  policy/verifier/search interface.
- Two verifier families: **programmatic** (math answer-equivalence; code execution
  against unit tests) and **learned** (an off-the-shelf open **PRM** for step scores).
- A **measured accuracy lift** on math (GSM8K, MATH-500) and code (HumanEval, MBPP)
  over a small open policy model, reported as accuracy-vs-compute curves with proper
  compute accounting and confidence intervals.
- Reproducible runs: pinned models, fixed problem subsets, seeds, config-driven.

**Non-goals (v1)** — deliberately out, to keep the build focused:
- **No training / fine-tuning / RL self-evolution.** We use *frozen* open policies and
  *off-the-shelf* open PRMs. (rStar-Math's self-evolution loop is inspiration, not
  scope — it's a multi-GPU data-generation project on its own.)
- **Not a production serving system.** It's a research harness; throughput and latency
  matter only insofar as they let experiments finish.
- **Not chasing SOTA.** The artifact is the *lift curve* and the working machinery,
  not a record on AIME.
- **No multimodal, no agentic tool-use, no retrieval.** Pure math + code reasoning.
- **No multi-GPU / distributed orchestration.** Single-node (CPU, one GPU, or one
  rented cloud GPU).

## 3. Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | **Python 3.11+** | The ML ecosystem; matches your `Congruent` / `llm-eval-harness`. |
| Policy inference | **Backend adapter**: Ollama (default) · vLLM (WSL2/cloud GPU) · OpenAI-compatible hosted | The workload is *cheap many-sample generation*; an adapter keeps it swappable per your hardware and mirrors `llm-eval-harness`'s provider-agnostic design. |
| Default policy model | **Qwen2.5-Math-7B-Instruct** (GPU) / **-1.5B-Instruct** (CPU); fallback **Llama-3.2-3B-Instruct** | Small enough to sample many traces; deliberately a *non-reasoning instruct* model so the external-search lift is **visible** (an already-RL'd reasoner like R1-distill bakes the search in and hides the effect). |
| Process reward model | **Qwen2.5-Math-PRM-7B** (GPU) / **Skywork-o1-Open-PRM-Qwen-2.5-1.5B** (CPU), via `transformers` | Strongest open step-level scorers; matched to the Qwen policy family. Run via `transformers`/vLLM because PRM scoring needs per-step logits that generation servers (Ollama) don't expose. |
| Math verifier | **`math-verify`** + **SymPy** | Symbolic answer-equivalence (`1/2` ≡ `0.5`), the correct outcome check — string match silently fails on equivalent forms. |
| Code verifier | **Sandboxed subprocess executor** + HumanEval/MBPP tests | Execution against unit tests is ground truth for code. Isolation is a hazard — see §4.4 / §9. |
| Datasets | **HuggingFace `datasets`** (GSM8K, MATH-500, HumanEval, MBPP) | Standard, versioned, with the canonical TTS subsets. |
| Search core | **Custom** (best-of-N, beam, MCTS) | This is the project's actual IP; no library should own it. |
| Reporting | **pandas + matplotlib**; JSON / CSV / Markdown | The deliverable is the accuracy-vs-compute plot + tables; mirror `llm-eval-harness` report artifacts. |
| Config / CLI | **YAML** + **Typer** | Config-driven reproducibility (cf. `lmeval.config.yaml`); a small CLI front door. |
| Quality | **pytest · ruff · mypy** | Cheap guardrails the build loop runs every milestone. |

*Versions verified 2026-06-27; pin exact versions at scaffold time and re-check.*

## 4. The reasoning-search core (get this exactly right)

Everything hinges on two things being precise: **the "step" abstraction** that lets
all three searches share one generator + verifier, and **compute accounting** that
makes "accuracy lift" honest. Get these right and best-of-N, beam, and MCTS are just
different controllers over the same parts.

### 4.1 The reasoning step

A **step** is a contiguous chunk of a reasoning trace that search can branch on,
score, and prune. Default segmentation: **double-newline (`\n\n`)** boundaries, with a
**max-token budget per step** as a hard cap (so a runaway step can't blow the tree).
Segmentation is configurable and recorded per run — it materially affects beam/MCTS.

### 4.2 Compute accounting (the honesty layer)

Every comparison is at **matched compute**, and the unit counts *both* models:

```python
@dataclass(frozen=True)
class Compute:
    policy_gen_tokens: int      # tokens generated by the policy
    policy_forward_calls: int   # policy invocations (a proxy for prefill/KV cost)
    verifier_forward_calls: int # PRM/judge invocations — counted, never free
    wall_seconds: float
```

> **Why this is non-negotiable:** a PRM-guided search that calls a 7B verifier 50×
> per problem is *not* comparable to best-of-N at the same N. We plot accuracy
> against a single scalar (default: total generated tokens, policy + verifier), so a
> "lift" can never be an artifact of un-counted compute. This is the difference
> between a credible result and a misleading one.

### 4.3 The core interfaces (ports)

```python
class Problem(Protocol):
    id: str
    prompt: str
    answer: str | None        # gold final answer (math) — None for code
    tests: list[str] | None   # unit tests (code) — None for math
    difficulty: str | None    # for compute-optimal per-difficulty analysis

class Step(Protocol):
    text: str
    token_count: int

@dataclass
class Trace:
    steps: list[Step]
    final_answer: str | None
    compute: Compute

class PolicyModel(Protocol):
    """A frozen generator behind the inference adapter (Ollama/vLLM/hosted)."""
    def sample_step(self, prompt: str, prefix: list[Step], *, n: int,
                    temperature: float, max_tokens: int) -> list[Step]: ...
    def sample_full(self, prompt: str, *, n: int, temperature: float) -> list[Trace]: ...

class OutcomeVerifier(Protocol):
    """Ground-truth-ish: math-equivalence or code execution. Returns pass/fail."""
    def verify(self, problem: Problem, trace: Trace) -> bool: ...

class ProcessVerifier(Protocol):
    """A PRM: a scalar score per step (and an aggregate) — no gold answer needed."""
    def score_steps(self, problem: Problem, prefix: list[Step]) -> list[float]: ...

class SearchStrategy(Protocol):
    """best-of-N | beam | mcts — all consume the same policy + verifier."""
    def search(self, problem: Problem, policy: PolicyModel,
               verifier: ProcessVerifier | None, budget: Compute) -> Trace: ...
```

The crucial design fact: **search strategies depend only on these protocols**, so a
new policy backend, a new verifier, or a new dataset is an adapter, never a change to
the search core. (Hexagonal / ports-and-adapters — see §5.)

### 4.4 Verifier semantics (the subtle part)

- **Math outcome:** extract the final answer (`\boxed{}` / "answer is" heuristics),
  then `math-verify` symbolic equivalence vs gold. This is the *evaluation* oracle and
  also a usable *selection* signal (best-of-N can select on it as an upper-bound
  "oracle" line).
- **Code outcome:** run the candidate against the dataset's unit tests in a **locked-
  down sandbox** (subprocess, hard timeout, no network, scratch tempdir; WSL2/Docker
  when available). Never execute model code on the host unguarded (§9).
- **Process (PRM):** step scores guide *search* (beam pruning, MCTS value) but are
  **never** the reported metric — final accuracy is always the outcome verifier on the
  chosen trace. We report both PRM-selected and oracle-selected to expose the PRM's
  selection gap (and any reward-hacking).

## 5. Architecture

Ports-and-adapters (hexagonal): a pure search/verification **core** that knows only
the §4 protocols, ringed by swappable adapters. This is what makes the engine
testable (mock policy/verifier) and portable across your hardware.

```
                 ┌───────────────────────────────────────────┐
   datasets  ──► │                  CORE                      │
 (GSM8K/MATH/    │  SearchStrategy: best-of-N · beam · MCTS   │
  HumanEval/     │  step segmentation · compute accounting    │
  MBPP)          │  experiment runner (problem × method ×     │
                 │  budget × seed → Result)                   │
                 └───▲─────────────▲───────────────▲──────────┘
                     │ policy       │ verify        │ record
        ┌────────────┴───┐ ┌───────┴────────┐ ┌────┴───────────────┐
        │ Inference adapt │ │ Verifier adapt │ │ Reporting           │
        │ Ollama·vLLM·    │ │ math-verify ·  │ │ pandas+matplotlib → │
        │ hosted (gen)    │ │ code-sandbox · │ │ JSON/CSV/MD + the    │
        │ transformers    │ │ PRM (process)  │ │ accuracy-vs-compute  │
        │ (PRM scoring)   │ │                │ │ curve                │
        └─────────────────┘ └────────────────┘ └─────────────────────┘
```

## 6. Core systems

### 6.1 Inference adapter
One interface, three backends. **Ollama** (default, native Windows) and **hosted**
cover generation; **vLLM** adds fast batched sampling + logprobs for heavy search;
**transformers** hosts the **PRM** (step scoring needs logits Ollama won't give). The
adapter normalizes sampling params and returns `Trace`/`Step` with token counts so
compute accounting is automatic. Mirrors your `llm-eval-harness` provider registry.

### 6.2 Verifiers
`OutcomeVerifier` (math-equivalence, code-execution) and `ProcessVerifier` (PRM). A
small **answer-extraction** module feeds the math verifier. The **code sandbox** is
its own module with the safety contract in §4.4/§9. PRMs are loaded by family-matched
config; a `MockVerifier` (returns gold-derived scores) exists for fast core tests.

### 6.3 Search strategies
- **best_of_n** — `sample_full(n)`, select by verifier (PRM-aggregate or oracle) or
  majority vote. The baseline rung.
- **beam** — at each step, expand the top-`k` partial traces by sampling `m`
  continuations, score with the PRM, keep top-`k`; stop at terminal/budget. (HF's
  `dvts` / Diverse Verifier Tree Search is the variant we mirror.)
- **mcts** — UCT/PUCT over the step tree: select by score + exploration bonus, expand
  by sampling continuations, value via PRM (and/or a cheap rollout), backup. The
  headline method; the most compute, the best on hard problems.

### 6.4 Experiment runner & reporting
A grid over `(dataset × method × compute-budget × seed)` → `Result` rows (problem id,
method, compute, pass/fail, PRM/oracle selections). Reporting aggregates into
accuracy with **Wilson confidence intervals** (as in `llm-eval-harness`), emits
JSON/CSV/Markdown, and renders the **accuracy-vs-compute curve** — the headline
artifact — plus per-difficulty breakdowns for the compute-optimal analysis.

## 7. Usage (interface surface)

A small Typer CLI over a YAML config (no GUI — it's a harness):

```bash
crucible run --method best_of_n --n 16 --dataset gsm8k --policy qwen2.5-math-7b-instruct
crucible run --method beam --beam-width 4 --dataset math500 --prm qwen2.5-math-prm-7b
crucible run --method mcts --budget-tokens 8000 --dataset math500
crucible sweep configs/lift-curve.yaml      # the full grid → the headline plot
crucible report runs/2026-06-27/            # tables + accuracy-vs-compute curve
```

Each run writes a self-contained record (config, per-problem traces, verdicts,
compute) so any number in the report is traceable to the trace that produced it.

## 8. Milestones

Built top-down — a thin end-to-end slice first, then each rung of the search ladder,
each independently runnable and *measurable*.

- **M0 — Skeleton & it runs.** CLI + YAML config; the Ollama inference adapter
  generates one CoT; math answer-extraction + `math-verify` outcome check; run on 10
  GSM8K problems and report **pass@1**. Proves the whole spine end-to-end.
- **M1 — Best-of-N + outcome verifier.** Sample N traces; report **majority@N** and
  **oracle best-of-N** vs pass@1 on GSM8K + MATH-500, *with compute accounting*. First
  measurable lift.
- **M2 — PRM integration (learned verifier).** Plug an open PRM as a `ProcessVerifier`;
  **PRM-weighted best-of-N**; compare PRM-selected vs majority vs oracle (exposes the
  PRM selection gap).
- **M3 — Step segmentation + beam search.** Implement the step abstraction and
  PRM-guided beam/DVTS; plot **accuracy-vs-compute** beam vs best-of-N (expect beam to
  win on harder MATH-500, per Snell).
- **M4 — Code track.** Sandboxed code-execution verifier + HumanEval/MBPP; run
  best-of-N/beam with execution feedback. Proves the verifier abstraction generalizes
  from math to code.
- **M5 — MCTS over reasoning steps.** UCT/PUCT with the PRM as value; selection /
  expansion / rollout / backup. Compare MCTS vs beam vs best-of-N at **matched
  compute** — the headline method.
- **M6 — Compute-optimal & the report.** Per-difficulty strategy selection
  (Snell-style); full accuracy-vs-compute curves with seeds + CIs; ablations (verifier
  on/off, PRM vs outcome, segmentation); a written results report with the headline
  plot. **This is the deliverable.**

## 9. Risks / open questions

- **Verifier gaming / false positives.** Outcome match can pass right-answer/wrong-
  reasoning; PRMs are imperfect (ProcessBench F1 ≈ 56 for the best open 7B) and search
  can reward-hack them. → Always report the *outcome* metric on the chosen trace, show
  PRM-selected **and** oracle lines, and inspect a sample of "passed" traces.
- **Compute on consumer hardware.** 7B policy + 7B PRM + MCTS is heavy. → Default to
  1.5B models + MATH-500/subset on CPU; offer vLLM (WSL2/cloud GPU); cache generations
  and PRM scores aggressively; make budgets explicit.
- **Un-counted verifier compute.** The classic way these results mislead. → §4.2
  accounting counts verifier calls; curves are vs total tokens. Treat as load-bearing.
- **Code-execution safety (esp. Windows).** Running model code is dangerous. → Locked-
  down subprocess (timeout, no network, temp dir), WSL2/Docker when available, opt-in
  flag; never unguarded on the host.
- **Step-segmentation ambiguity.** "What is a step?" changes beam/MCTS behavior. →
  Decide a default (`\n\n` + token cap), make it configurable, record it per run, and
  ablate it in M6.
- **Policy/PRM family mismatch.** A PRM scores its own family's style best. → Default
  to a Qwen PRM with a Qwen policy; treat cross-family as an experiment, not the
  baseline.

## 10. References

*(verified 2026-06-27)*

- **Snell et al., "Scaling LLM Test-Time Compute Optimally…"** (arXiv 2408.03314,
  ICLR 2025) — compute-optimal scaling; search > best-of-N for small models on hard
  problems; >4× efficiency. The intellectual backbone.
- **rStar-Math** (arXiv 2501.04519, ICML 2025) — MCTS + process-preference model with
  small models; Qwen2.5-Math-7B 58.8→90.0 on MATH, AIME 53.3%. The MCTS prior art.
- **Lightman et al., "Let's Verify Step by Step"** (PRM800K) — process supervision >
  outcome supervision; the PRM idea.
- **HuggingFace `search-and-learn`** (cookbook: "Scaling Test-Time Compute for Longer
  Thinking") — the reference implementation of best-of-N / beam / DVTS with small Llama
  + a PRM on MATH-500. Closest existing codebase; study it, then build our own.
- **Open PRMs:** Qwen2.5-Math-PRM-7B/72B; Skywork-o1-Open-PRM-Qwen-2.5-1.5B/7B; RLHFlow
  PRM-8B. Catalog: RyanLiu112/Awesome-Process-Reward-Models. **ProcessBench** for PRM
  quality.
- **Small open policies:** Qwen2.5-Math-Instruct (1.5B/7B), Llama-3.2-Instruct (1B/3B);
  DeepSeek-R1-Distill-Qwen (1.5B/7B) as already-reasoning comparators.
- **Tooling:** `math-verify` + SymPy (math equivalence); HuggingFace `datasets`
  (GSM8K, MATH-500, HumanEval, MBPP); vLLM / Ollama (inference).
- **Shared opportunity:** your **`llm-eval-harness`** already does provider-agnostic
  model calling, suite running, Wilson-CI stats, and JSON/CSV/MD reports. Crucible's
  runner + reporting should **reuse or extend it** rather than reinvent — your
  established "extract on reuse" pattern. Candidate for a shared `eval-core` package.
