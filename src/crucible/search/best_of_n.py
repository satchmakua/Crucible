"""best_of_n — sample N traces, keep the best one. The first measurable lift.

Two verifier-light selectors ship here (PRM-weighted selection arrives in M3):

- **majority** — vote on the extracted final answers and return a representative
  trace. Verifier-free (0 verifier calls for selection).
- **oracle** — return the first trace the *outcome* verifier passes. This "cheats"
  with the gold answer, so it's an **upper bound**, not a deployable selector — it
  bounds how much lift the samples even contain. Counts one verifier call per check.

Either way the returned trace's `.compute` accounts for **all N samples** (plus any
selection-time verifier calls), so the accuracy-vs-compute curve stays honest.
"""

from __future__ import annotations

from collections import Counter

from crucible.config import RunConfig
from crucible.domain.ports import OutcomeVerifier, PolicyModel, ProcessVerifier
from crucible.domain.types import Compute, Problem, Trace
from crucible.verify import extract_final_answer


class BestOfNStrategy:
    """Sample `config.n` traces and select one by `config.selection`."""

    name = "best_of_n"

    def search(
        self,
        problem: Problem,
        policy: PolicyModel,
        outcome: OutcomeVerifier,
        process: ProcessVerifier | None,
        config: RunConfig,
    ) -> Trace:
        n = max(1, config.n)
        traces = policy.sample_full(
            problem,
            n=n,
            temperature=config.policy.temperature,
            max_tokens=config.policy.max_tokens,
        )
        if not traces:
            return Trace(steps=[], final_answer=None, compute=Compute())

        gen = Compute()
        for t in traces:
            gen = gen + t.compute

        if config.selection == "majority":
            chosen, sel_compute = self._majority(traces), Compute()
        elif config.selection == "oracle":
            chosen, sel_compute = self._oracle(problem, traces, outcome)
        else:
            raise ValueError(
                f"unknown selection '{config.selection}' (majority | oracle)."
            )

        return Trace(steps=chosen.steps, final_answer=chosen.final_answer, compute=gen + sel_compute)

    @staticmethod
    def _majority(traces: list[Trace]) -> Trace:
        answers = [extract_final_answer(t.text) for t in traces]
        counts = Counter(a for a in answers if a is not None)
        if not counts:
            return traces[0]
        top = counts.most_common(1)[0][0]
        for trace, answer in zip(traces, answers, strict=True):
            if answer == top:
                return trace
        return traces[0]

    @staticmethod
    def _oracle(
        problem: Problem, traces: list[Trace], outcome: OutcomeVerifier
    ) -> tuple[Trace, Compute]:
        checks = 0
        for trace in traces:
            checks += 1
            if outcome.verify(problem, trace).correct:
                return trace, Compute(verifier_forward_calls=checks)
        return traces[0], Compute(verifier_forward_calls=checks)
