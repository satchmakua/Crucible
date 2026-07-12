"""SyntheticPolicy: a seeded, reproducible simulator of a known accuracy."""

from __future__ import annotations

from crucible.domain.types import Problem
from crucible.inference.synthetic import SyntheticPolicy
from crucible.verify import MathOutcomeVerifier

_PROBLEM = Problem(id="x", prompt="?", answer="42")
_OUTCOME = MathOutcomeVerifier()


def _rate(accuracy: float, *, n: int, seed: int = 0) -> float:
    traces = SyntheticPolicy(accuracy=accuracy, seed=seed).sample_full(
        _PROBLEM, n=n, temperature=0.0, max_tokens=8
    )
    return sum(_OUTCOME.verify(_PROBLEM, t).correct for t in traces) / len(traces)


def test_same_seed_is_deterministic() -> None:
    a = SyntheticPolicy(accuracy=0.5, seed=7).sample_full(_PROBLEM, n=10, temperature=0, max_tokens=8)
    b = SyntheticPolicy(accuracy=0.5, seed=7).sample_full(_PROBLEM, n=10, temperature=0, max_tokens=8)
    assert [t.text for t in a] == [t.text for t in b]


def test_empirical_accuracy_tracks_parameter() -> None:
    assert 0.6 < _rate(0.7, n=500) < 0.8


def test_extremes() -> None:
    assert _rate(0.0, n=50) == 0.0
    assert _rate(1.0, n=50) == 1.0


def test_distractor_never_math_equals_gold() -> None:
    from crucible.inference.synthetic import _distractor
    from crucible.verify import math_equal

    for gold in ("0.0", "-0", "0/5", "72", "1", "0"):
        assert not math_equal(_distractor(gold), gold), gold


def test_zero_equivalent_gold_scores_honestly() -> None:
    # gold '0.0' with accuracy 0 → every trace is wrong; the distractor must not be a
    # zero-equivalent '0' that the symbolic verifier would score correct (false 100%).
    problem = Problem(id="z", prompt="?", answer="0.0")
    traces = SyntheticPolicy(accuracy=0.0, seed=0).sample_full(
        problem, n=30, temperature=0.0, max_tokens=8
    )
    assert sum(_OUTCOME.verify(problem, t).correct for t in traces) == 0
