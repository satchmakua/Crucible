"""Compute accounting — the honesty layer must add up and count both models."""

from __future__ import annotations

from crucible.domain.types import Compute


def test_addition_sums_every_field() -> None:
    a = Compute(policy_gen_tokens=10, policy_forward_calls=1, verifier_forward_calls=2, wall_seconds=0.5)
    b = Compute(policy_gen_tokens=5, policy_forward_calls=1, verifier_forward_calls=3, wall_seconds=1.0)
    total = a + b
    assert total.policy_gen_tokens == 15
    assert total.policy_forward_calls == 2
    assert total.verifier_forward_calls == 5
    assert total.wall_seconds == 1.5


def test_total_tokens_counts_policy_and_verifier() -> None:
    c = Compute(policy_gen_tokens=100, verifier_gen_tokens=40)
    assert c.total_tokens == 140


def test_empty_compute_is_zero() -> None:
    c = Compute()
    assert c.total_tokens == 0
    assert (c + c).total_tokens == 0


def test_wall_seconds_replaced_not_summed() -> None:
    # The runner sets wall_seconds to the true outer elapsed rather than adding it to the
    # per-trace generation time the strategy already summed (which double-counted real
    # Ollama runs). Token/call counts are preserved.
    from dataclasses import replace

    trace_compute = Compute(policy_gen_tokens=50, policy_forward_calls=4, wall_seconds=100.0)
    result = replace(trace_compute + Compute(verifier_forward_calls=1), wall_seconds=5.0)
    assert result.wall_seconds == 5.0  # true elapsed, not 100 + 5
    assert result.policy_gen_tokens == 50
    assert result.policy_forward_calls == 4
    assert result.verifier_forward_calls == 1
