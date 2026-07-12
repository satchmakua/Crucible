"""Beam/DVTS search: it needs a PRM, solves the stepwise task, and beats best-of-N."""

from __future__ import annotations

from typing import Any

import pytest

from crucible.config import PolicyConfig, RunConfig
from crucible.domain.types import Problem, Step
from crucible.runner import run
from crucible.search.beam import BeamStrategy
from crucible.verify import MathOutcomeVerifier

_OUTCOME = MathOutcomeVerifier()


def _cfg(method: str, **kw: Any) -> RunConfig:
    base: dict[str, Any] = {
        "method": method,
        "dataset": "sample",
        "seed": 0,
        "prm": "step",
        "step_accuracy": 0.6,
        "step_depth": 5,
        "step_prm_accuracy": 0.99,
        "policy": PolicyConfig(backend="stepwise", model="sim"),
    }
    base.update(kw)
    return RunConfig(**base)


def test_beam_requires_a_process_verifier() -> None:
    cfg = _cfg("beam")
    cfg.prm = None  # no PRM → beam can't prune
    with pytest.raises(ValueError, match="needs a process verifier"):
        run(cfg)


def test_beam_solves_the_stepwise_task() -> None:
    summary = run(_cfg("beam", beam_width=6, beam_expansions=6, max_steps=8))
    assert summary.accuracy >= 0.8


def test_beam_beats_best_of_n() -> None:
    beam = run(_cfg("beam", beam_width=4, beam_expansions=4, max_steps=8))
    bon = run(_cfg("best_of_n", n=8, selection="prm"))
    assert beam.correct >= bon.correct


def test_beam_counts_policy_and_verifier_compute() -> None:
    c = run(_cfg("beam", beam_width=4, beam_expansions=4)).total_compute
    assert c.policy_forward_calls > 0
    assert c.verifier_forward_calls > 0
    assert c.verifier_gen_tokens > 0  # PRM forward-pass tokens land on the compute axis


class _TerminalTrapPolicy:
    """First expansion offers a completed correct trace plus a tempting non-terminal one;
    every later step is non-terminal. Beam must not discard the completed answer."""

    name = "trap"

    def sample_step(self, problem: Problem, prefix: list[Step], *, n: int, temperature: float, max_tokens: int) -> list[Step]:
        if not prefix:
            opts = [Step("It is done. \\boxed{42}", 4), Step("Step 1: keep going 7", 5)]
            return [opts[i % 2] for i in range(n)]
        return [Step(f"Step {len(prefix) + 1}: keep going 7", 5) for _ in range(n)]

    def sample_full(self, *a: Any, **k: Any) -> list[Any]:  # pragma: no cover
        raise NotImplementedError


class _PreferNonTerminalPRM:
    """Scores non-terminal (still-reasoning) partials ABOVE completed ones."""

    name = "trap-prm"

    def score_steps(self, problem: Problem, prefix: list[Step]) -> list[float]:
        text = "\n\n".join(s.text for s in prefix)
        return [0.1 if "boxed" in text else 0.9]


def test_beam_returns_completed_trace_not_top_prm_partial() -> None:
    problem = Problem(id="p", prompt="q", answer="42")
    cfg = RunConfig(
        method="beam", beam_width=2, beam_expansions=2, max_steps=3,
        policy=PolicyConfig(backend="mock"),
    )
    chosen = BeamStrategy().search(
        problem, _TerminalTrapPolicy(), _OUTCOME, _PreferNonTerminalPRM(), cfg
    )
    # Even though the PRM prefers the non-terminal partials, the completed \boxed{42}
    # trace must be returned (standard beam returns the best COMPLETED hypothesis).
    assert _OUTCOME.verify(problem, chosen).correct
