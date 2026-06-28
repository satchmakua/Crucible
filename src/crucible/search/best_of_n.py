"""best_of_n — sample N traces, keep the best one. The first measurable lift.

The selection logic lives in `search.selectors` (shared with the comparison path).
Three selectors: **majority** (verifier-free vote), **oracle** (first verifier-passing
trace — an upper bound, not deployable), and **prm** (highest process-reward score,
needs `--prm`). The returned trace's `.compute` accounts for **all N samples** plus the
selector's own cost, so the accuracy-vs-compute curve stays honest.
"""

from __future__ import annotations

from crucible.config import RunConfig
from crucible.domain.ports import OutcomeVerifier, PolicyModel, ProcessVerifier
from crucible.domain.types import Compute, Problem, Trace
from crucible.search.selectors import SELECTORS


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
        for trace in traces:
            gen = gen + trace.compute

        selector = SELECTORS.get(config.selection)
        if selector is None:
            raise ValueError(
                f"unknown selection '{config.selection}' ({' | '.join(SELECTORS)})."
            )
        chosen, sel_compute = selector(
            problem, traces, outcome, process, aggregate=config.prm_aggregate
        )
        return Trace(
            steps=chosen.steps, final_answer=chosen.final_answer, compute=gen + sel_compute
        )
