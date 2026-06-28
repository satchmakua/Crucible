"""Selectors: PRM-weighted selection picks a good trace and counts its compute."""

from __future__ import annotations

import pytest

from crucible.domain.types import Compute, Problem, Step, Trace
from crucible.search.selectors import select_prm
from crucible.verify import MathOutcomeVerifier, MockProcessVerifier

_PROBLEM = Problem(id="x", prompt="?", answer="42")
_OUTCOME = MathOutcomeVerifier()


def _trace(text: str) -> Trace:
    return Trace(
        steps=[Step(text=text, token_count=2)],
        final_answer=None,
        compute=Compute(policy_gen_tokens=2, policy_forward_calls=1),
    )


def test_prm_selects_a_correct_trace_with_skilled_prm() -> None:
    traces = [_trace("\\boxed{0}"), _trace("\\boxed{42}"), _trace("\\boxed{0}")]
    chosen, compute = select_prm(
        _PROBLEM, traces, _OUTCOME, MockProcessVerifier(accuracy=1.0, seed=0)
    )
    assert _OUTCOME.verify(_PROBLEM, chosen).correct
    # Scored every candidate; counted a forward pass + its tokens per candidate.
    assert compute.verifier_forward_calls == 3
    assert compute.verifier_gen_tokens == 6


def test_prm_requires_a_process_verifier() -> None:
    with pytest.raises(ValueError, match="needs a process verifier"):
        select_prm(_PROBLEM, [_trace("\\boxed{1}")], _OUTCOME, None)
