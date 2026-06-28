"""Process verifier (PRM): the mock simulator's signal and score aggregation."""

from __future__ import annotations

import pytest

from crucible.domain.types import Problem, Step
from crucible.verify import MockProcessVerifier, aggregate_scores

_PROBLEM = Problem(id="x", prompt="?", answer="42")


def test_aggregate_modes() -> None:
    assert aggregate_scores([1.0, 2.0, 3.0], "mean") == 2.0
    assert aggregate_scores([1.0, 2.0, 3.0], "min") == 1.0
    assert aggregate_scores([1.0, 2.0, 3.0], "last") == 3.0
    assert aggregate_scores([], "mean") == 0.0


def test_aggregate_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown aggregate"):
        aggregate_scores([1.0], "bogus")


def test_mock_prm_scores_correct_above_wrong() -> None:
    prm = MockProcessVerifier(accuracy=0.9, seed=0)
    correct = aggregate_scores(prm.score_steps(_PROBLEM, [Step("answer \\boxed{42}", 3)]))
    wrong = aggregate_scores(prm.score_steps(_PROBLEM, [Step("answer \\boxed{0}", 3)]))
    assert correct > wrong


def test_mock_prm_is_deterministic() -> None:
    a = MockProcessVerifier(seed=1).score_steps(_PROBLEM, [Step("x \\boxed{42}", 2)])
    b = MockProcessVerifier(seed=1).score_steps(_PROBLEM, [Step("x \\boxed{42}", 2)])
    assert a == b
