"""Code outcome verifier: execution against unit tests is ground truth for code.

Mirrors the math outcome verifier behind the same `OutcomeVerifier` port — the search
core doesn't care whether "correct" means symbolic equivalence or passing tests. Refuses
to run unless execution was explicitly enabled (defense in depth; the runner also gates
this before the loop starts).
"""

from __future__ import annotations

from crucible.domain.types import Problem, Trace, Verdict
from crucible.verify.code_extract import extract_code
from crucible.verify.code_sandbox import run_in_sandbox


class CodeOutcomeVerifier:
    """`OutcomeVerifier` for code: extract the solution, run it against the tests."""

    name = "code"

    def __init__(self, *, allow_execution: bool = False, timeout: float = 10.0) -> None:
        self.allow_execution = allow_execution
        self.timeout = timeout

    def verify(self, problem: Problem, trace: Trace) -> Verdict:
        if not self.allow_execution:
            raise RuntimeError(
                "code execution is disabled; enable it explicitly (--allow-code-exec) "
                "to run model code in the locked-down sandbox."
            )
        if not problem.tests:
            return Verdict(correct=False, detail="problem has no tests")
        code = extract_code(trace.text)
        if not code.strip():
            return Verdict(correct=False, detail="no code extracted from trace")
        result = run_in_sandbox(code, list(problem.tests), timeout=self.timeout)
        return Verdict(correct=result.passed, detail=result.detail)
