"""Verifiers: outcome (ground-truth-ish) and process (PRM) checks.

M0 ships the math outcome verifier (answer extraction + symbolic equivalence). The
code-execution outcome verifier (M5) and PRM process verifier (M3) slot in behind the
same ports.
"""

from __future__ import annotations

from crucible.verify.answer_extract import extract_final_answer, has_explicit_answer
from crucible.verify.code_extract import extract_code
from crucible.verify.code_outcome import CodeOutcomeVerifier
from crucible.verify.code_sandbox import SandboxResult, run_in_sandbox
from crucible.verify.math_outcome import MathOutcomeVerifier, math_equal
from crucible.verify.process import MockProcessVerifier, PRMVerifier, aggregate_scores

__all__ = [
    "CodeOutcomeVerifier",
    "MathOutcomeVerifier",
    "MockProcessVerifier",
    "PRMVerifier",
    "SandboxResult",
    "aggregate_scores",
    "extract_code",
    "extract_final_answer",
    "has_explicit_answer",
    "math_equal",
    "run_in_sandbox",
]
