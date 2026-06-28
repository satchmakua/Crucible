"""Selectors: given N candidate traces, pick one and report its selection compute.

Factored out so best-of-N (one selector per run) and the selection-gap *comparison*
(all selectors on the *same* N samples) share identical logic. Each returns
`(chosen_trace, selection_compute)` — the compute is the selector's own cost only; the
caller adds the shared generation compute.

- **majority** — vote on extracted answers (verifier-free; 0 selection cost).
- **oracle** — first trace the outcome verifier passes (upper bound; counts the checks).
- **prm** — highest aggregate process-reward score (counts a PRM forward pass per
  candidate, whose tokens land on the honest compute axis).
"""

from __future__ import annotations

from collections import Counter

from crucible.domain.ports import OutcomeVerifier, ProcessVerifier
from crucible.domain.types import Compute, Problem, Trace
from crucible.verify import aggregate_scores, extract_final_answer


def select_majority(
    problem: Problem,
    traces: list[Trace],
    outcome: OutcomeVerifier,
    process: ProcessVerifier | None,
    *,
    aggregate: str = "mean",
) -> tuple[Trace, Compute]:
    answers = [extract_final_answer(t.text) for t in traces]
    counts = Counter(a for a in answers if a is not None)
    if not counts:
        return traces[0], Compute()
    top = counts.most_common(1)[0][0]
    for trace, answer in zip(traces, answers, strict=True):
        if answer == top:
            return trace, Compute()
    return traces[0], Compute()


def select_oracle(
    problem: Problem,
    traces: list[Trace],
    outcome: OutcomeVerifier,
    process: ProcessVerifier | None,
    *,
    aggregate: str = "mean",
) -> tuple[Trace, Compute]:
    checks = 0
    for trace in traces:
        checks += 1
        if outcome.verify(problem, trace).correct:
            return trace, Compute(verifier_forward_calls=checks)
    return traces[0], Compute(verifier_forward_calls=checks)


def select_prm(
    problem: Problem,
    traces: list[Trace],
    outcome: OutcomeVerifier,
    process: ProcessVerifier | None,
    *,
    aggregate: str = "mean",
) -> tuple[Trace, Compute]:
    if process is None:
        raise ValueError("selection 'prm' needs a process verifier — pass --prm.")
    best_idx, best_score = 0, float("-inf")
    calls, tokens = 0, 0
    for idx, trace in enumerate(traces):
        scores = process.score_steps(problem, trace.steps)
        calls += 1
        tokens += sum(s.token_count for s in trace.steps)
        score = aggregate_scores(scores, aggregate)
        if score > best_score:
            best_score, best_idx = score, idx
    return traces[best_idx], Compute(verifier_forward_calls=calls, verifier_gen_tokens=tokens)


SELECTORS = {
    "majority": select_majority,
    "oracle": select_oracle,
    "prm": select_prm,
}
