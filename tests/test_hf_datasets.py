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


def test_math500_hard_keeps_levels_4_and_5_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    from crucible import data
    from crucible.data import hf
    from crucible.domain.types import Problem

    fake = [
        Problem(id=f"math500-{i}", prompt="p", answer="1", difficulty=f"level-{lvl}")
        for i, lvl in enumerate([1, 4, 5, 2, 5])
    ]
    monkeypatch.setattr(hf, "load_math500", lambda limit=None: fake if limit is None else fake[:limit])

    hard = hf.load_math500_hard(None)
    assert [p.id for p in hard] == ["math500-1", "math500-2", "math500-4"]
    assert hf.load_math500_hard(2) == hard[:2]
    # And it is registered as a dataset (the beam/MCTS real-cell target).
    assert data.load_dataset("math500-hard", limit=2) == hard[:2]
