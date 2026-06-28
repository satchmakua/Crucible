"""Same-samples selection-gap comparison (majority / prm / oracle)."""

from __future__ import annotations

from crucible.config import PolicyConfig, RunConfig
from crucible.runner import run_comparison


def _cfg(prm_accuracy: float) -> RunConfig:
    return RunConfig(
        method="best_of_n",
        dataset="sample",
        n=16,
        seed=0,
        prm="mock",
        prm_accuracy=prm_accuracy,
        synthetic_accuracy=0.5,
        policy=PolicyConfig(backend="synthetic", model="sim"),
    )


def test_three_selectors_over_all_problems() -> None:
    summaries = run_comparison(_cfg(0.9))
    assert set(summaries) == {"majority", "oracle", "prm"}
    assert all(s.total == 6 for s in summaries.values())


def test_selectors_share_identical_generation() -> None:
    # The whole point: all selectors score the *same* samples, so generation compute
    # is identical across them (only selection cost differs).
    summaries = run_comparison(_cfg(0.9))
    gen_tokens = {s.total_compute.policy_gen_tokens for s in summaries.values()}
    assert len(gen_tokens) == 1


def test_oracle_upper_bounds_the_others() -> None:
    summaries = run_comparison(_cfg(0.6))
    assert summaries["oracle"].accuracy >= summaries["prm"].accuracy
    assert summaries["oracle"].accuracy >= summaries["majority"].accuracy


def test_perfect_prm_matches_oracle_and_beats_majority() -> None:
    summaries = run_comparison(_cfg(1.0))
    assert summaries["prm"].accuracy == summaries["oracle"].accuracy
    assert summaries["prm"].accuracy >= summaries["majority"].accuracy


def test_prm_compute_counted_majority_free() -> None:
    summaries = run_comparison(_cfg(0.9))
    assert summaries["prm"].total_compute.verifier_gen_tokens > 0
    assert summaries["majority"].total_compute.verifier_gen_tokens == 0
