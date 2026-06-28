"""GSM8K / MATH-500 row mapping and gold extraction (pure, no download)."""

from __future__ import annotations

import pytest

from crucible.data.hf import (
    extract_gsm8k_gold,
    gsm8k_to_problem,
    math500_to_problem,
)


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("Janet has 3 eggs.\nShe sells them.\n#### 18", "18"),
        ("reasoning\n#### 1,000", "1000"),
        ("#### $42", "42"),
        ("no marker here 7", "no marker here 7"),  # falls back to whole field, stripped
    ],
)
def test_extract_gsm8k_gold(field: str, expected: str) -> None:
    assert extract_gsm8k_gold(field) == expected


def test_gsm8k_to_problem() -> None:
    row = {"question": "  How many?  ", "answer": "steps\n#### 42"}
    p = gsm8k_to_problem(row, 3)
    assert p.id == "gsm8k-3"
    assert p.prompt == "How many?"
    assert p.answer == "42"
    assert p.difficulty == "grade-school"


def test_math500_to_problem() -> None:
    row = {"problem": "Find x.", "answer": "\\frac{1}{2}", "level": 5}
    p = math500_to_problem(row, 0)
    assert p.id == "math500-0"
    assert p.prompt == "Find x."
    assert p.answer == "\\frac{1}{2}"
    assert p.difficulty == "level-5"
