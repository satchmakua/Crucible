"""The synthetic stepwise task: policy correctness model + step PRM signal."""

from __future__ import annotations

from crucible.domain.types import Problem, Step
from crucible.synthetic_stepwise import (
    BAD_MARKER,
    GOOD_MARKER,
    StepRewardPRM,
    StepwisePolicy,
    _distractor,
)
from crucible.verify import MathOutcomeVerifier, aggregate_scores, extract_final_answer

_PROBLEM = Problem(id="p", prompt="?", answer="42")
_OUTCOME = MathOutcomeVerifier()


def test_perfect_policy_emits_gold_all_good() -> None:
    pol = StepwisePolicy(step_accuracy=1.0, depth=4, seed=0)
    trace = pol.sample_full(_PROBLEM, n=1, temperature=0.0, max_tokens=8)[0]
    assert extract_final_answer(trace.text) == "42"
    assert _OUTCOME.verify(_PROBLEM, trace).correct
    assert len(trace.steps) == 4


def test_failing_policy_never_emits_gold() -> None:
    pol = StepwisePolicy(step_accuracy=0.0, depth=4, seed=0)
    trace = pol.sample_full(_PROBLEM, n=1, temperature=0.0, max_tokens=8)[0]
    assert not _OUTCOME.verify(_PROBLEM, trace).correct


def test_sample_step_marks_final_with_boxed() -> None:
    pol = StepwisePolicy(step_accuracy=1.0, depth=2, seed=1)
    first = pol.sample_step(_PROBLEM, [], n=1, temperature=0.0, max_tokens=8)[0]
    assert "\\boxed" not in first.text  # not terminal yet
    second = pol.sample_step(_PROBLEM, [first], n=1, temperature=0.0, max_tokens=8)[0]
    assert "\\boxed" in second.text  # final step carries the answer


def test_step_prm_scores_good_above_bad() -> None:
    prm = StepRewardPRM(accuracy=0.95, seed=0)
    good = aggregate_scores(prm.score_steps(_PROBLEM, [Step(f"ok {GOOD_MARKER}", 3)]))
    bad = aggregate_scores(prm.score_steps(_PROBLEM, [Step(f"no {BAD_MARKER}", 3)]))
    assert good > bad


def test_all_flawed_trace_is_not_scored_correct_for_zero_gold() -> None:
    # A zero-equivalent gold must not collide with the '0' distractor, or an all-BAD
    # chain (which should be wrong) would score correct, breaking 'correct iff all good'.
    from crucible.verify import math_equal

    problem = Problem(id="z", prompt="?", answer="0.0")
    trace = StepwisePolicy(step_accuracy=0.0, depth=4, seed=0).sample_full(
        problem, n=1, temperature=0.0, max_tokens=8
    )[0]
    assert not _OUTCOME.verify(problem, trace).correct
    assert not math_equal(_distractor("0.0"), "0.0")
