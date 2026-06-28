"""End-to-end code track: the opt-in gate and a real sandboxed run."""

from __future__ import annotations

import pytest

from crucible.config import PolicyConfig, RunConfig
from crucible.data.hf import humaneval_to_problem, mbpp_to_problem
from crucible.runner import run


def _cfg(*, allow: bool, method: str = "pass1", **kw: object) -> RunConfig:
    return RunConfig(
        method=method,
        dataset="code-sample",
        allow_code_execution=allow,
        policy=PolicyConfig(backend="mock", model="scripted"),
        **kw,  # type: ignore[arg-type]
    )


def test_code_dataset_blocked_without_optin() -> None:
    with pytest.raises(RuntimeError, match="OFF by default"):
        run(_cfg(allow=False))


def test_code_sample_pass_at_1_with_execution() -> None:
    summary = run(_cfg(allow=True))
    assert summary.total == 3
    assert summary.correct == 2  # c1, c2 correct; c3 (reverse) is buggy
    verdicts = {r.problem_id: r.correct for r in summary.results}
    assert verdicts == {"c1": True, "c2": True, "c3": False}


def test_best_of_n_oracle_on_code() -> None:
    summary = run(_cfg(allow=True, method="best_of_n", n=2, selection="oracle"))
    assert summary.correct == 2


def test_humaneval_mapper() -> None:
    row = {
        "task_id": "HumanEval/0",
        "prompt": "def f(x):\n",
        "test": "def check(c):\n    assert c(1) == 1",
        "entry_point": "f",
    }
    p = humaneval_to_problem(row, 0)
    assert p.id == "HumanEval/0"
    assert p.tests is not None and "check(f)" in p.tests[0]
    assert p.difficulty == "code"


def test_mbpp_mapper() -> None:
    row = {"task_id": 7, "text": "Write foo.", "test_list": ["assert foo(1) == 1"]}
    p = mbpp_to_problem(row, 0)
    assert p.id == "mbpp-7"
    assert p.tests == ("assert foo(1) == 1",)
