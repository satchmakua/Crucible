"""Dataset name → problems (and, for the mock backend, scripted outputs)."""

from __future__ import annotations

from collections.abc import Callable

from crucible.data.code_sample import CODE_SAMPLE_PROBLEMS, CODE_SAMPLE_SCRIPTS
from crucible.data.hf import load_gsm8k, load_humaneval, load_math500, load_mbpp
from crucible.data.sample import SAMPLE_PROBLEMS, SAMPLE_SCRIPTS
from crucible.domain.types import Problem

# Bundled datasets (no download needed).
_BUNDLED: dict[str, tuple[Problem, ...]] = {
    "sample": SAMPLE_PROBLEMS,
    "code-sample": CODE_SAMPLE_PROBLEMS,
}

# HuggingFace-backed loaders (need the `datasets` extra; raise a clear message if absent).
_LOADERS: dict[str, Callable[[int | None], list[Problem]]] = {
    "gsm8k": load_gsm8k,
    "math500": load_math500,
    "humaneval": load_humaneval,
    "mbpp": load_mbpp,
}

# Datasets whose answers are code run against unit tests (the execution sandbox, M5).
CODE_DATASETS = frozenset({"code-sample", "humaneval", "mbpp"})

_SCRIPTS = {
    "sample": SAMPLE_SCRIPTS,
    "code-sample": CODE_SAMPLE_SCRIPTS,
}


def available_datasets() -> list[str]:
    return sorted({*_BUNDLED, *_LOADERS})


def load_dataset(name: str, *, limit: int | None = None) -> list[Problem]:
    if name in _BUNDLED:
        problems = list(_BUNDLED[name])
        return problems[:limit] if limit is not None else problems
    if name in _LOADERS:
        return _LOADERS[name](limit)
    raise ValueError(
        f"unknown dataset '{name}'. Available now: {', '.join(available_datasets())}."
    )


def scripts_for(name: str) -> dict[str, list[str]]:
    """Canned mock-backend outputs for a dataset (only bundled sets have them)."""
    return dict(_SCRIPTS.get(name, {}))
