"""Answer extraction: the small heuristic that feeds the math verifier."""

from __future__ import annotations

import pytest

from crucible.verify import extract_final_answer
from crucible.verify.answer_extract import has_explicit_answer


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Therefore the answer is \\boxed{72}.", "72"),
        ("\\boxed{1/4}", "1/4"),
        ("So we get \\boxed{40} mph.", "40"),
        ("The final answer is 144.", "144"),
        ("first \\boxed{1} then \\boxed{2}", "2"),  # takes the last boxed
        ("after simplifying we get 12 cupcakes left", "12"),  # number fallback
        ("\\boxed{\\frac{1}{2}}", "\\frac{1}{2}"),  # one level of nested braces
    ],
)
def test_extracts_expected(text: str, expected: str) -> None:
    assert extract_final_answer(text) == expected


@pytest.mark.parametrize("text", ["", "no number or boxed answer here"])
def test_returns_none_when_nothing_found(text: str) -> None:
    assert extract_final_answer(text) is None


# Regression (stress test): real models often write "The answer is N" prose and drop
# \boxed. The phrase branch must match that (and stay terminal), while a bare "answer"
# mid-sentence must NOT count as a final answer.
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The final answer is 42", "42"),
        ("So the answer is 100 apples", "100"),
        ("Answer: 7", "7"),
        ("the answer is -3/4", "-3/4"),
    ],
)
def test_extracts_prose_answers(text: str, expected: str) -> None:
    assert extract_final_answer(text) == expected


def test_has_explicit_answer_on_prose_and_boxed() -> None:
    assert has_explicit_answer("The final answer is 42")
    assert has_explicit_answer("Answer: 7")
    assert has_explicit_answer("we conclude \\boxed{5}")


def test_has_explicit_answer_ignores_bare_answer_word() -> None:
    # No connector (is/:/=) and no \boxed — must not be mistaken for a final answer,
    # or beam/MCTS would treat a still-reasoning trace as terminal.
    assert not has_explicit_answer("consider the answer to this problem")
    assert not has_explicit_answer("what is the answer to life")
    assert not has_explicit_answer("Step 2: intermediate value 48")
