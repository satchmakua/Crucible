r"""SyntheticPolicy — a seeded simulator of a policy with a known accuracy.

Like the mock, this is a **test/demo backend, not a real model**: given a problem's
gold answer it fabricates a "correct" trace (`\boxed{gold}`) with probability
`accuracy` and a "wrong" trace otherwise, using a seed derived from the run seed and
the problem id. That makes the test-time-scaling behaviour deterministic and
analysable — pass@1 ≈ `accuracy`, oracle@N ≈ 1-(1-accuracy)^N — so best-of-N's lift
curve can be generated and unit-tested without a GPU or network.
"""

from __future__ import annotations

import random
from functools import cache

from crucible.domain.types import Compute, Problem, Step, Trace
from crucible.segment import approx_tokens, segment
from crucible.verify.math_outcome import math_equal


@cache
def _distractor(gold: str | None) -> str:
    """A wrong answer that is NOT math-equivalent to gold.

    A literal "0" collides with a zero-equivalent gold ("0.0", "-0", "0/5") under the
    symbolic verifier, which would score the "wrong" trace as correct. Pick the first
    candidate the verifier does *not* equate with gold instead.
    """
    if gold is None:
        return "0"
    for candidate in ("0", "1", "2", "-1", "7"):
        if not math_equal(candidate, gold):
            return candidate
    return "999999999"

_CORRECT = (
    "Let me work through this carefully.\n\n"
    "Reasoning step by step about the quantities involved.\n\n"
    "After checking the arithmetic, the final answer is \\boxed{{{ans}}}."
)
_WRONG = (
    "Let me work through this carefully.\n\n"
    "Reasoning step by step, though I may have slipped somewhere.\n\n"
    "I'll go with the final answer \\boxed{{{ans}}}."
)


class SyntheticPolicy:
    """Emits correct/incorrect traces at a configured rate (seeded, reproducible)."""

    name = "synthetic"

    def __init__(self, *, accuracy: float = 0.5, seed: int = 0, max_step_tokens: int = 512) -> None:
        self.accuracy = accuracy
        self.seed = seed
        self._max_step_tokens = max_step_tokens

    def _trace(self, text: str) -> Trace:
        steps = segment(text, max_step_tokens=self._max_step_tokens)
        compute = Compute(policy_gen_tokens=approx_tokens(text), policy_forward_calls=1)
        return Trace(steps=steps, final_answer=None, compute=compute)

    def sample_full(
        self, problem: Problem, *, n: int, temperature: float, max_tokens: int
    ) -> list[Trace]:
        gold = problem.answer
        distractor = _distractor(gold)
        rng = random.Random(f"{self.seed}:{problem.id}")
        traces: list[Trace] = []
        for i in range(n):
            # A per-sample nonce makes each attempt's text distinct (the boxed answer
            # still drives extraction), so a downstream PRM scores candidates
            # individually rather than seeing N identical strings.
            nonce = f"Attempt {i + 1}.\n\n"
            if gold is not None and rng.random() < self.accuracy:
                text = nonce + _CORRECT.format(ans=gold)
            else:
                text = nonce + _WRONG.format(ans=distractor)
            traces.append(self._trace(text))
        return traces

    def sample_step(
        self,
        problem: Problem,
        prefix: list[Step],
        *,
        n: int,
        temperature: float,
        max_tokens: int,
    ) -> list[Step]:
        # Step-wise sampling isn't meaningful for this simulator; beam/MCTS (M4+) use
        # real backends. Return empty steps so the contract is satisfied.
        return [Step(text="", token_count=0) for _ in range(n)]
