"""H4 — adversarial/degenerate stress tests for the search strategies.

Covers the corners the happy-path tests skip: width-1 (greedy) beams, single-candidate
expansion, MCTS budget exhaustion, and tied / empty PRM scores. These must not crash and
must always return a valid `Trace`.
"""

from __future__ import annotations

from typing import Any

from crucible.config import PolicyConfig, RunConfig
from crucible.domain.types import Problem, Step, Trace
from crucible.runner import run
from crucible.search.beam import BeamStrategy
from crucible.search.mcts import MCTSStrategy
from crucible.verify import MathOutcomeVerifier

_OUTCOME = MathOutcomeVerifier()


def _stepcfg(method: str, **kw: Any) -> RunConfig:
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


def test_beam_width_1_greedy_runs() -> None:
    summary = run(_stepcfg("beam", beam_width=1, beam_expansions=4, max_steps=8))
    assert summary.total == 6  # greedy beam still completes every problem


def test_beam_single_candidate_expansion() -> None:
    summary = run(_stepcfg("beam", beam_width=2, beam_expansions=1, max_steps=8))
    assert summary.total == 6


def test_mcts_budget_exhaustion_does_not_crash() -> None:
    # A budget too small to reach any terminal must still return a Trace, not raise.
    summary = run(_stepcfg("mcts", budget_tokens=1, mcts_max_sims=200))
    assert summary.total == 6


class _TiedPRM:
    """Constant score — every candidate ties."""

    name = "tied"

    def score_steps(self, problem: Problem, prefix: list[Step]) -> list[float]:
        return [0.5]


class _EmptyPRM:
    """Returns no scores — aggregate must fall back to 0.0, not crash."""

    name = "empty"

    def score_steps(self, problem: Problem, prefix: list[Step]) -> list[float]:
        return []


class _TerminatingPolicy:
    """Non-terminal for two steps, then emits a boxed answer."""

    name = "term"

    def sample_step(self, problem: Problem, prefix: list[Step], *, n: int, temperature: float, max_tokens: int) -> list[Step]:
        if len(prefix) >= 2:
            return [Step("\\boxed{1}", 2) for _ in range(n)]
        return [Step(f"step {len(prefix) + 1}", 2) for _ in range(n)]

    def sample_full(self, *a: Any, **k: Any) -> list[Any]:  # pragma: no cover
        raise NotImplementedError


def test_beam_survives_tied_and_empty_prm_scores() -> None:
    problem = Problem(id="p", prompt="q", answer="1")
    cfg = RunConfig(
        method="beam", beam_width=2, beam_expansions=2, max_steps=5,
        policy=PolicyConfig(backend="mock"),
    )
    for prm in (_TiedPRM(), _EmptyPRM()):
        chosen = BeamStrategy().search(problem, _TerminatingPolicy(), _OUTCOME, prm, cfg)
        assert isinstance(chosen, Trace)


def test_mcts_survives_tied_and_empty_prm_scores() -> None:
    problem = Problem(id="p", prompt="q", answer="1")
    cfg = RunConfig(
        method="mcts", beam_expansions=2, budget_tokens=500, max_steps=5,
        policy=PolicyConfig(backend="mock"),
    )
    for prm in (_TiedPRM(), _EmptyPRM()):
        chosen = MCTSStrategy().search(problem, _TerminatingPolicy(), _OUTCOME, prm, cfg)
        assert isinstance(chosen, Trace)
