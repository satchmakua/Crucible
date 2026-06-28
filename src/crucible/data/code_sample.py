"""A tiny bundled code dataset + scripted solutions for the mock backend.

Three simple problems with unit tests, paired with canned "model" code — two correct,
one wrong — so a `pass1` run on the mock backend through the **real sandbox** yields a
deterministic 2/3, exercising extraction + execution + the opt-in gate end-to-end.
"""

from __future__ import annotations

from crucible.domain.types import Problem

CODE_SAMPLE_PROBLEMS: tuple[Problem, ...] = (
    Problem(
        id="c1",
        prompt="Write a function `add(a, b)` that returns the sum of two numbers.",
        tests=("assert add(2, 3) == 5", "assert add(-1, 1) == 0", "assert add(0, 0) == 0"),
        difficulty="code",
    ),
    Problem(
        id="c2",
        prompt="Write a function `is_even(n)` that returns True iff n is even.",
        tests=("assert is_even(4)", "assert not is_even(3)", "assert is_even(0)"),
        difficulty="code",
    ),
    Problem(
        id="c3",
        prompt="Write a function `reverse_string(s)` that returns s reversed.",
        tests=("assert reverse_string('abc') == 'cba'", "assert reverse_string('') == ''"),
        difficulty="code",
    ),
)

# Canned completions keyed by id. Correct: c1, c2. Wrong: c3 (returns s unchanged).
CODE_SAMPLE_SCRIPTS: dict[str, list[str]] = {
    "c1": ["Here is the solution:\n\n```python\ndef add(a, b):\n    return a + b\n```"],
    "c2": ["```python\ndef is_even(n):\n    return n % 2 == 0\n```"],
    "c3": ["```python\ndef reverse_string(s):\n    return s  # bug: not reversed\n```"],
}
