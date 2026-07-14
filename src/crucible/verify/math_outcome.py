"""Math outcome verifier: symbolic answer-equivalence against the gold answer.

String matching silently fails on equivalent forms (``1/2`` vs ``0.5``), so we use
``math-verify`` (SymPy under the hood) for the actual check (DESIGN.md §4.4). This is
both the evaluation oracle *and* a usable selection signal for best-of-N.

Implementation notes:
- ``math-verify``'s built-in timeout uses signals/subprocesses, which are unavailable
  or fragile on Windows; we disable it (inputs here are short) and silence the
  resulting library warning.
- If ``math-verify`` ever errors on an input, we fall back to a normalized string /
  SymPy comparison so a single bad parse never crashes a run.
"""

from __future__ import annotations

import logging

from crucible.domain.types import Problem, Trace, Verdict
from crucible.verify.answer_extract import extract_final_answer

# Quiet the "timeout is disabled" warnings — a deliberate, documented choice here.
for _name in ("math_verify", "math_verify.utils", "math_verify.grader", "math_verify.parser"):
    logging.getLogger(_name).setLevel(logging.ERROR)


def _candidate_forms(s: str) -> list[str]:
    """The delimited form first, then the raw string.

    math-verify's LaTeX extractor is built for *delimited* math. MATH-500 golds and
    extracted ``\\boxed`` contents are bare, and bare compound answers parse wrong:
    ``\\left( 3, \\frac{\\pi}{2} \\right)`` extracts as just ``3`` and the matching
    tuple ``(3, \\frac{\\pi}{2})`` as nothing — a systematic false negative on
    tuple/interval/coordinate answers (and a lurking ``3 == 3`` false positive).
    ``$``-wrapping restores full-fidelity parses.
    """
    if "$" in s:
        return [s]
    return [f"${s}$", s]


def _math_verify_equal(pred: str, gold: str) -> bool:
    import itertools

    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

    cfg = [LatexExtractionConfig(), ExprExtractionConfig()]
    # The first (gold form, pred form) pair where both sides parse is definitive —
    # fidelity order, so a parsed-as-tuple never loses to a parsed-as-first-number.
    for g_form, p_form in itertools.product(_candidate_forms(gold), _candidate_forms(pred)):
        g = parse(g_form, extraction_config=cfg, parsing_timeout=None)
        p = parse(p_form, extraction_config=cfg, parsing_timeout=None)
        if g and p:
            return bool(verify(g, p, timeout_seconds=None))
    raise ValueError("empty parse")


def _fallback_equal(pred: str, gold: str) -> bool:
    """Used only if math-verify raises: normalized string, then SymPy equivalence."""
    a, b = pred.strip(), gold.strip()
    if a == b:
        return True
    try:
        from sympy import simplify
        from sympy.parsing.sympy_parser import parse_expr

        return bool(simplify(parse_expr(a) - parse_expr(b)) == 0)
    except Exception:
        return False


def math_equal(pred: str, gold: str) -> bool:
    """True if `pred` is mathematically equivalent to `gold`."""
    try:
        return _math_verify_equal(pred, gold)
    except Exception:
        return _fallback_equal(pred, gold)


class MathOutcomeVerifier:
    """`OutcomeVerifier` for math: extract the final answer, check equivalence."""

    name = "math"

    def verify(self, problem: Problem, trace: Trace) -> Verdict:
        if problem.answer is None:
            return Verdict(correct=False, detail="problem has no gold answer")
        predicted = extract_final_answer(trace.text)
        if predicted is None:
            return Verdict(correct=False, detail="no answer extracted from trace")
        ok = math_equal(predicted, problem.answer)
        return Verdict(correct=ok, detail=f"predicted={predicted!r} gold={problem.answer!r}")
