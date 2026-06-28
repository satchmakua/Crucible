"""The code-execution sandbox: it passes good code and contains bad code."""

from __future__ import annotations

from crucible.verify import run_in_sandbox

_ADD = "def add(a, b):\n    return a + b"


def test_correct_code_passes() -> None:
    result = run_in_sandbox(_ADD, ["assert add(2, 3) == 5"], timeout=10)
    assert result.passed


def test_wrong_code_fails() -> None:
    result = run_in_sandbox(_ADD, ["assert add(2, 3) == 6"], timeout=10)
    assert not result.passed


def test_raising_code_fails() -> None:
    result = run_in_sandbox("def f():\n    raise ValueError('boom')", ["f()"], timeout=10)
    assert not result.passed


def test_syntax_error_fails() -> None:
    result = run_in_sandbox("def f(:\n    pass", ["f()"], timeout=10)
    assert not result.passed


def test_infinite_loop_times_out() -> None:
    result = run_in_sandbox("x = 0", ["while True:\n    pass"], timeout=2)
    assert not result.passed
    assert "tim" in result.detail.lower()  # "timed out"


def test_network_access_is_blocked() -> None:
    code = "import socket"
    tests = ["socket.create_connection(('example.com', 80))"]
    result = run_in_sandbox(code, tests, timeout=10)
    assert not result.passed  # the preamble neuters socket before any connection
