"""Datasets behind one loader interface.

M0 ships a tiny bundled `sample` set (math, with gold answers) so the engine runs
cold. The real datasets — GSM8K, MATH-500, HumanEval, MBPP — load via HuggingFace
`datasets` from M1 (install the `datasets` extra).
"""

from __future__ import annotations

from crucible.data.registry import (
    CODE_DATASETS,
    available_datasets,
    load_dataset,
    scripts_for,
)

__all__ = ["CODE_DATASETS", "available_datasets", "load_dataset", "scripts_for"]
