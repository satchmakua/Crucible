r"""Prompt construction for the policy.

Kept separate from the inference adapters so the prompt is a tunable, recorded part of
a run rather than baked into a backend or a dataset. The math CoT prompt deliberately
asks for a ``\boxed{}`` final answer, which is exactly what `verify.answer_extract`
keys on — prompt and extractor are designed together.
"""

from __future__ import annotations

from crucible.domain.types import Problem

MATH_COT_INSTRUCTION = (
    "Solve the following math problem. Reason step by step, then give the final "
    "answer on its own line inside \\boxed{}.\n\n"
)


def build_cot_prompt(problem: Problem) -> str:
    """A zero-shot chain-of-thought prompt for a math problem."""
    return f"{MATH_COT_INSTRUCTION}Problem: {problem.prompt}\n\nSolution:"


CODE_INSTRUCTION = (
    "Write a correct, self-contained Python solution to the problem below. Put the "
    "complete solution in a single ```python code block, defining the required "
    "function(s) by name.\n\n"
)


def build_code_prompt(problem: Problem) -> str:
    """A zero-shot prompt for a code problem (extractor keys on the ```python block)."""
    return f"{CODE_INSTRUCTION}{problem.prompt}\n"
