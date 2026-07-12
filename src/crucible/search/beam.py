"""beam — PRM-guided beam search over reasoning steps (DVTS-style).

Each round: expand every kept partial by sampling `m` continuations (`sample_step`),
score each resulting partial with the process verifier, and keep the top `k`. Terminal
partials (those that have emitted a final answer) are carried forward, not re-expanded.
Search stops when the whole beam is terminal or the step budget runs out.

Unlike best-of-N, the PRM prunes *partial* reasoning, so compute concentrates on
promising branches — the lift the literature reports on harder problems (DESIGN.md
§6.3). The returned trace's `.compute` accounts for every policy step sampled **and**
every PRM forward pass (its tokens land on the honest compute axis).
"""

from __future__ import annotations

from crucible.config import RunConfig
from crucible.domain.ports import OutcomeVerifier, PolicyModel, ProcessVerifier
from crucible.domain.types import Compute, Problem, Trace
from crucible.verify import aggregate_scores, has_explicit_answer


def _is_terminal(trace: Trace) -> bool:
    # A partial is terminal only once it emits an explicit final answer — not when a
    # bare number (e.g. "Step 2") appears mid-reasoning.
    return has_explicit_answer(trace.text)


class BeamStrategy:
    """Keep the top-k partial traces each round, scored by the PRM."""

    name = "beam"

    def search(
        self,
        problem: Problem,
        policy: PolicyModel,
        outcome: OutcomeVerifier,
        process: ProcessVerifier | None,
        config: RunConfig,
    ) -> Trace:
        if process is None:
            raise ValueError("beam search needs a process verifier — pass --prm.")
        k = max(1, config.beam_width)
        m = max(1, config.beam_expansions)

        compute = Compute()
        beams: list[Trace] = [Trace(steps=[], final_answer=None, compute=Compute())]
        # Track the best *completed* trace across all rounds. Standard beam search
        # returns the best COMPLETED hypothesis; without this, a completed (possibly
        # correct) trace can be dropped for a higher-PRM non-terminal partial, or the
        # loop can exhaust its budget with only non-terminal beams and return a partial
        # whose "answer" is a stray mid-reasoning number.
        best_terminal: Trace | None = None
        best_terminal_score = float("-inf")

        for _ in range(max(1, config.max_steps)):
            candidates: list[Trace] = []
            for partial in beams:
                if _is_terminal(partial):
                    candidates.append(partial)
                    continue
                next_steps = policy.sample_step(
                    problem,
                    partial.steps,
                    n=m,
                    temperature=config.policy.temperature,
                    max_tokens=config.policy.max_tokens,
                )
                gen_tokens = sum(s.token_count for s in next_steps)
                compute = compute + Compute(
                    policy_gen_tokens=gen_tokens, policy_forward_calls=len(next_steps)
                )
                for step in next_steps:
                    candidates.append(
                        Trace(steps=[*partial.steps, step], final_answer=None, compute=Compute())
                    )

            scored: list[tuple[float, Trace]] = []
            for cand in candidates:
                scores = process.score_steps(problem, cand.steps)
                compute = compute + Compute(
                    verifier_forward_calls=1,
                    verifier_gen_tokens=sum(s.token_count for s in cand.steps),
                )
                score = aggregate_scores(scores, config.prm_aggregate)
                scored.append((score, cand))
                if _is_terminal(cand) and score > best_terminal_score:
                    best_terminal_score, best_terminal = score, cand

            scored.sort(key=lambda pair: pair[0], reverse=True)
            beams = [trace for _score, trace in scored[:k]]
            if all(_is_terminal(b) for b in beams):
                break

        # Prefer the best completed trace; only fall back to a partial if none finished.
        if best_terminal is not None:
            best = best_terminal
        else:
            best = beams[0] if beams else Trace(steps=[], final_answer=None, compute=Compute())
        return Trace(steps=best.steps, final_answer=best.final_answer, compute=compute)
