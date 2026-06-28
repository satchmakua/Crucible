"""Code extraction from model responses."""

from __future__ import annotations

from crucible.verify import extract_code


def test_extracts_python_fence() -> None:
    text = "Sure!\n\n```python\ndef f():\n    return 1\n```\nDone."
    assert extract_code(text) == "def f():\n    return 1"


def test_extracts_bare_fence() -> None:
    assert extract_code("```\nx = 1\n```") == "x = 1"


def test_takes_last_block() -> None:
    text = "```python\nold = 1\n```\nthen\n```python\nnew = 2\n```"
    assert extract_code(text) == "new = 2"


def test_falls_back_to_whole_text() -> None:
    assert extract_code("def f():\n    return 2") == "def f():\n    return 2"
