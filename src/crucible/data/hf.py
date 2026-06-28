"""HuggingFace dataset loaders (GSM8K, MATH-500) behind the `datasets` extra.

The row→`Problem` mappings and GSM8K gold extraction are pure functions so they can be
unit-tested without a download; only `_load_split` touches the network. `datasets` is
imported lazily, so the base install (sample-only) never needs it.
"""

from __future__ import annotations

from typing import Any

from crucible.domain.types import Problem


def _require_datasets() -> Any:
    try:
        import datasets
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise NotImplementedError(
            "loading this dataset needs the `datasets` extra: "
            'pip install -e ".[datasets]"'
        ) from exc
    return datasets


def _load_split(path: str, name: str | None, split: str, limit: int | None) -> list[dict[str, Any]]:
    datasets = _require_datasets()
    ds = datasets.load_dataset(path, name, split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return [dict(row) for row in ds]


def extract_gsm8k_gold(answer_field: str) -> str:
    """GSM8K answers end with ``#### <final>``; return the bare final value."""
    tail = answer_field.split("####")[-1] if "####" in answer_field else answer_field
    return tail.strip().replace(",", "").replace("$", "").strip()


def gsm8k_to_problem(row: dict[str, Any], idx: int) -> Problem:
    return Problem(
        id=f"gsm8k-{idx}",
        prompt=str(row["question"]).strip(),
        answer=extract_gsm8k_gold(str(row["answer"])),
        difficulty="grade-school",
    )


def math500_to_problem(row: dict[str, Any], idx: int) -> Problem:
    level = row.get("level", "?")
    return Problem(
        id=f"math500-{idx}",
        prompt=str(row["problem"]).strip(),
        answer=str(row["answer"]).strip(),
        difficulty=f"level-{level}",
    )


def load_gsm8k(limit: int | None = None) -> list[Problem]:
    rows = _load_split("openai/gsm8k", "main", "test", limit)
    return [gsm8k_to_problem(row, i) for i, row in enumerate(rows)]


def load_math500(limit: int | None = None) -> list[Problem]:
    rows = _load_split("HuggingFaceH4/MATH-500", None, "test", limit)
    return [math500_to_problem(row, i) for i, row in enumerate(rows)]


def humaneval_to_problem(row: dict[str, Any], idx: int) -> Problem:
    # The HumanEval `test` defines check(candidate); we append the call so the test tuple
    # is self-contained (candidate code + this block must exit cleanly to pass).
    test = str(row["test"])
    entry = str(row["entry_point"])
    return Problem(
        id=str(row.get("task_id", f"humaneval-{idx}")),
        prompt=str(row["prompt"]),
        tests=(f"{test}\n\ncheck({entry})",),
        difficulty="code",
    )


def mbpp_to_problem(row: dict[str, Any], idx: int) -> Problem:
    tests = tuple(str(t) for t in row.get("test_list", []))
    return Problem(
        id=f"mbpp-{row.get('task_id', idx)}",
        prompt=str(row["text"]),
        tests=tests,
        difficulty="code",
    )


def load_humaneval(limit: int | None = None) -> list[Problem]:
    rows = _load_split("openai_humaneval", None, "test", limit)
    return [humaneval_to_problem(row, i) for i, row in enumerate(rows)]


def load_mbpp(limit: int | None = None) -> list[Problem]:
    rows = _load_split("mbpp", None, "test", limit)
    return [mbpp_to_problem(row, i) for i, row in enumerate(rows)]
