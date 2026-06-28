"""Run configuration — config-driven reproducibility (DESIGN.md §3, §7).

A `RunConfig` fully determines a run: dataset, method, policy backend, sampling, and
budget. It round-trips to/from a plain dict so it can be loaded from YAML and written
verbatim into each run record.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PolicyConfig:
    """Which generator to use and how to sample from it."""

    backend: str = "mock"  # mock | ollama | hosted
    model: str = "scripted"
    temperature: float = 0.7
    max_tokens: int = 1024


@dataclass
class RunConfig:
    """Everything needed to reproduce one run."""

    method: str = "pass1"  # pass1 | best_of_n | beam | mcts (later milestones)
    dataset: str = "sample"  # sample | gsm8k | math500 | humaneval | mbpp
    limit: int | None = None  # cap number of problems (None = all)
    n: int = 1  # samples per problem (best_of_n etc.)
    selection: str = "majority"  # best_of_n selector: majority | oracle (PRM in M3)
    seed: int = 0
    synthetic_accuracy: float = 0.5  # per-problem correctness for the synthetic backend
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    prm: str | None = None  # process-reward model id ("mock" for the seeded simulator)
    prm_accuracy: float = 0.8  # skill of the mock PRM (1.0 ≈ oracle, 0.5 ≈ random)
    prm_aggregate: str = "mean"  # how to reduce per-step PRM scores: mean | min | last | prod
    segmentation: str = "double_newline"
    max_step_tokens: int = 512
    budget_tokens: int | None = None
    output_dir: str = "runs"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunConfig:
        data = dict(data)
        policy_raw = data.pop("policy", None)
        cfg = cls(**data)
        if policy_raw is not None:
            cfg.policy = PolicyConfig(**policy_raw)
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config at {path} must be a YAML mapping, got {type(data).__name__}")
        return cls.from_dict(data)
