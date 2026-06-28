"""Dataset name → problems (and, for the mock backend, scripted outputs)."""

from __future__ import annotations

from collections.abc import Callable

from crucible.data.hf import load_gsm8k, load_math500
from crucible.data.sample import SAMPLE_PROBLEMS, SAMPLE_SCRIPTS
from crucible.domain.types import Problem

# Datasets loadable now. GSM8K + MATH-500 need the `datasets` extra installed; the
# loaders raise a clear message if it's missing.
_LOADERS: dict[str, Callable[[int | None], list[Problem]]] = {
    "gsm8k": load_gsm8k,
    "math500": load_math500,
}

# Named but not yet wired, so the CLI can give a milestone-aware message.
_PLANNED = {
    "humaneval": "M5",
    "mbpp": "M5",
}


def available_datasets() -> list[str]:
    return ["sample", *sorted(_LOADERS)]


def load_dataset(name: str, *, limit: int | None = None) -> list[Problem]:
    if name == "sample":
        problems = list(SAMPLE_PROBLEMS)
        return problems[:limit] if limit is not None else problems
    if name in _LOADERS:
        return _LOADERS[name](limit)
    if name in _PLANNED:
        raise NotImplementedError(
            f"dataset '{name}' is planned for milestone {_PLANNED[name]} (see ROADMAP.md). "
            f"Available now: {', '.join(available_datasets())}."
        )
    raise ValueError(
        f"unknown dataset '{name}'. Available now: {', '.join(available_datasets())}."
    )


def scripts_for(name: str) -> dict[str, list[str]]:
    """Canned mock-backend outputs for a dataset (only the bundled `sample` set has them)."""
    return dict(SAMPLE_SCRIPTS) if name == "sample" else {}
