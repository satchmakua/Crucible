"""The math CoT prompt elicits the \\boxed{} answer the extractor keys on."""

from __future__ import annotations

from crucible.domain.types import Problem
from crucible.prompts import build_cot_prompt


def test_prompt_includes_problem_and_boxed_instruction() -> None:
    problem = Problem(id="x", prompt="What is 2 + 2?", answer="4")
    prompt = build_cot_prompt(problem)
    assert "What is 2 + 2?" in prompt
    assert "\\boxed{}" in prompt
    assert "step by step" in prompt.lower()
