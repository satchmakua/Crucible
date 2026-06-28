"""The experiment runner: (dataset × method × policy × seed) → results.

This is the core orchestration: load problems, build the policy behind its port, run
the chosen search strategy per problem, score the chosen trace with the outcome
verifier, and accumulate honest compute. It depends only on ports and value types —
no backend or dataset library is imported here directly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

from crucible.config import RunConfig
from crucible.data import CODE_DATASETS, load_dataset, scripts_for
from crucible.domain.ports import OutcomeVerifier, PolicyModel, ProcessVerifier
from crucible.domain.types import Compute, Result
from crucible.inference import OllamaPolicy, ScriptedPolicy, SyntheticPolicy
from crucible.prompts import build_code_prompt, build_cot_prompt
from crucible.search import get_strategy
from crucible.search.selectors import SELECTORS
from crucible.stats import wilson_interval
from crucible.synthetic_stepwise import StepRewardPRM, StepwisePolicy
from crucible.verify import (
    CodeOutcomeVerifier,
    MathOutcomeVerifier,
    MockProcessVerifier,
    PRMVerifier,
    extract_final_answer,
)


@dataclass
class RunSummary:
    """Everything a report needs from a single run."""

    config: RunConfig
    results: list[Result] = field(default_factory=list)
    run_dir: str | None = None

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def correct(self) -> int:
        return sum(1 for r in self.results if r.correct)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def accuracy_ci(self) -> tuple[float, float]:
        return wilson_interval(self.correct, self.total)

    @property
    def total_compute(self) -> Compute:
        out = Compute()
        for r in self.results:
            out = out + r.compute
        return out


def build_policy(config: RunConfig) -> PolicyModel:
    """Construct the inference adapter named by `config.policy.backend`."""
    backend = config.policy.backend
    if backend == "mock":
        return ScriptedPolicy(scripts_for(config.dataset), max_step_tokens=config.max_step_tokens)
    if backend == "synthetic":
        return SyntheticPolicy(
            accuracy=config.synthetic_accuracy,
            seed=config.seed,
            max_step_tokens=config.max_step_tokens,
        )
    if backend == "stepwise":
        return StepwisePolicy(
            step_accuracy=config.step_accuracy, depth=config.step_depth, seed=config.seed
        )
    if backend == "ollama":
        prompt_builder = build_code_prompt if config.dataset in CODE_DATASETS else build_cot_prompt
        return OllamaPolicy(
            config.policy.model,
            max_step_tokens=config.max_step_tokens,
            prompt_builder=prompt_builder,
        )
    if backend == "hosted":
        raise NotImplementedError(
            "the hosted OpenAI-compatible backend lands in a later milestone; "
            "use --policy mock (offline) or --policy ollama."
        )
    raise ValueError(f"unknown policy backend '{backend}' (mock | ollama | hosted).")


def build_outcome_verifier(config: RunConfig) -> OutcomeVerifier:
    """Math equivalence for math datasets; sandboxed execution for code datasets."""
    if config.dataset in CODE_DATASETS:
        if not config.allow_code_execution:
            raise RuntimeError(
                f"dataset '{config.dataset}' runs model-generated code, which is OFF by "
                "default. Pass --allow-code-exec to enable the locked-down sandbox "
                "(isolated subprocess, hard timeout, no network, scratch temp dir)."
            )
        return CodeOutcomeVerifier(allow_execution=True, timeout=config.code_timeout)
    return MathOutcomeVerifier()


def build_process_verifier(config: RunConfig) -> ProcessVerifier | None:
    """The PRM, if requested. `--prm mock` is the seeded simulator; else a real PRM."""
    if not config.prm:
        return None
    if config.prm == "mock":
        return MockProcessVerifier(accuracy=config.prm_accuracy, seed=config.seed)
    if config.prm == "step":
        return StepRewardPRM(accuracy=config.step_prm_accuracy, seed=config.seed)
    return PRMVerifier(config.prm)


def run(config: RunConfig) -> RunSummary:
    """Execute `config` and return a summary (no IO — the caller writes the record)."""
    problems = load_dataset(config.dataset, limit=config.limit)
    policy = build_policy(config)
    outcome = build_outcome_verifier(config)
    process = build_process_verifier(config)
    strategy = get_strategy(config.method)

    is_code = config.dataset in CODE_DATASETS
    summary = RunSummary(config=config)
    for problem in problems:
        t0 = time.perf_counter()
        chosen = strategy.search(problem, policy, outcome, process, config)
        verdict = outcome.verify(problem, chosen)
        elapsed = time.perf_counter() - t0
        compute = chosen.compute + Compute(verifier_forward_calls=1, wall_seconds=elapsed)
        summary.results.append(
            Result(
                problem_id=problem.id,
                method=config.method,
                dataset=config.dataset,
                correct=verdict.correct,
                # For code, "predicted" (a math answer) is meaningless — report None.
                predicted=None if is_code else extract_final_answer(chosen.text),
                gold=problem.answer,
                compute=compute,
                difficulty=problem.difficulty,
            )
        )
    return summary


def run_comparison(config: RunConfig) -> dict[str, RunSummary]:
    """Generate N samples per problem *once*, then score with every selector.

    This is the honest selection-gap experiment: majority, oracle, and (if a PRM is
    configured) prm all see the *same* samples, so their accuracy differences are real,
    not an artifact of independent sampling. Each selector's results carry the shared
    generation compute plus that selector's own selection cost.
    """
    problems = load_dataset(config.dataset, limit=config.limit)
    policy = build_policy(config)
    outcome = build_outcome_verifier(config)
    process = build_process_verifier(config)

    names = ["majority", "oracle"] + (["prm"] if process is not None else [])
    summaries = {
        name: RunSummary(config=replace(config, method="best_of_n", selection=name))
        for name in names
    }

    n = max(1, config.n)
    for problem in problems:
        traces = policy.sample_full(
            problem,
            n=n,
            temperature=config.policy.temperature,
            max_tokens=config.policy.max_tokens,
        )
        if not traces:
            continue
        gen = Compute()
        for trace in traces:
            gen = gen + trace.compute

        for name in names:
            chosen, sel_compute = SELECTORS[name](
                problem, traces, outcome, process, aggregate=config.prm_aggregate
            )
            verdict = outcome.verify(problem, chosen)
            compute = gen + sel_compute + Compute(verifier_forward_calls=1)
            summaries[name].results.append(
                Result(
                    problem_id=problem.id,
                    method="best_of_n",
                    dataset=config.dataset,
                    correct=verdict.correct,
                    predicted=extract_final_answer(chosen.text),
                    gold=problem.answer,
                    compute=compute,
                    difficulty=problem.difficulty,
                )
            )
    return summaries
