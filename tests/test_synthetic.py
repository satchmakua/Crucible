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


# --- sample_step: the synthetic backend can drive beam/MCTS (not empty steps). -------


def _walk_chain(policy: SyntheticPolicy, depth: int) -> list:
    steps: list = []
    for _ in range(depth):
        steps.append(policy.sample_step(_PROBLEM, steps, n=1, temperature=0.7, max_tokens=64)[0])
    return steps


def test_sample_step_emits_marked_nonempty_steps() -> None:
    policy = SyntheticPolicy(accuracy=0.5, seed=0, depth=3)
    steps = policy.sample_step(_PROBLEM, [], n=4, temperature=0.7, max_tokens=64)
    assert len(steps) == 4
    from crucible.synthetic_stepwise import BAD_MARKER, GOOD_MARKER

    for step in steps:
        assert step.text and step.token_count > 0
        assert GOOD_MARKER in step.text or BAD_MARKER in step.text


def test_sample_step_is_deterministic_given_seed_and_prefix() -> None:
    a = SyntheticPolicy(accuracy=0.5, seed=3, depth=3)
    b = SyntheticPolicy(accuracy=0.5, seed=3, depth=3)
    sa = a.sample_step(_PROBLEM, [], n=5, temperature=0.7, max_tokens=64)
    sb = b.sample_step(_PROBLEM, [], n=5, temperature=0.7, max_tokens=64)
    assert [s.text for s in sa] == [s.text for s in sb]


def test_all_good_chain_ends_in_gold_and_flawed_chain_does_not() -> None:
    from crucible.domain.types import Trace

    perfect = _walk_chain(SyntheticPolicy(accuracy=1.0, seed=0, depth=3), depth=3)
    assert "\\boxed{42}" in perfect[-1].text
    assert _OUTCOME.verify(_PROBLEM, Trace(steps=perfect)).correct

    flawed = _walk_chain(SyntheticPolicy(accuracy=0.0, seed=0, depth=3), depth=3)
    assert "\\boxed{" in flawed[-1].text  # still terminates with an explicit answer
    assert not _OUTCOME.verify(_PROBLEM, Trace(steps=flawed)).correct


def test_beam_and_mcts_run_on_the_synthetic_backend() -> None:
    # The regression this guards: sample_step used to return empty Step("") objects,
    # so the main synthetic backend silently couldn't drive the top of the ladder.
    from crucible.config import PolicyConfig, RunConfig
    from crucible.runner import run

    for method in ("beam", "mcts"):
        cfg = RunConfig(
            method=method,
            dataset="sample",
            synthetic_accuracy=1.0,
            prm="step",
            step_depth=3,
            beam_width=2,
            beam_expansions=2,
            max_steps=6,
            budget_tokens=400,
            mcts_max_sims=20,
            policy=PolicyConfig(backend="synthetic", model="sim"),
        )
        summary = run(cfg)
        assert summary.total == 6
        assert summary.accuracy == 1.0, method  # perfect steps → search must find gold
