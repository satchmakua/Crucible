"""Sweeps: run a grid of experiments → the accuracy-vs-compute curve.

A sweep YAML is the base run config plus a ``grid:`` list. Each grid cell overrides the
base; any list-valued field in a cell (e.g. ``n: [4, 8, 16]``) is expanded into the
cartesian product of runs. An optional ``seeds: [...]`` list runs every cell once per
seed and **pools** the results (problems × seeds), so accuracy comes with a proper
Wilson CI over more samples. Every run writes its own record under the sweep directory,
and the pooled cells are aggregated into ``sweep.json`` + ``curve.png`` (with the
compute-optimal frontier overlaid) — the headline artifact (DESIGN.md §1, §6.4).
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from crucible.config import RunConfig
from crucible.report import render_curve, write_run_record
from crucible.runner import RunSummary, run
from crucible.stats import wilson_interval


def expand_grid(base: dict[str, Any], grid: list[dict[str, Any]]) -> list[RunConfig]:
    """Expand a base config + grid cells into concrete `RunConfig`s."""
    configs: list[RunConfig] = []
    for cell in grid:
        merged: dict[str, Any] = {**base, **cell}
        list_keys = [k for k, v in merged.items() if isinstance(v, list)]
        if not list_keys:
            configs.append(RunConfig.from_dict(merged))
            continue
        for combo in itertools.product(*(merged[k] for k in list_keys)):
            expanded = dict(merged)
            for key, value in zip(list_keys, combo, strict=True):
                expanded[key] = value
            configs.append(RunConfig.from_dict(expanded))
    return configs


def _knob(cfg: RunConfig) -> int:
    """The value that varies along a method's line (for the table + grouping)."""
    if cfg.method == "beam":
        return cfg.beam_width
    if cfg.method == "mcts":
        return cfg.budget_tokens or 0
    return cfg.n


def _cell_key(cfg: RunConfig) -> str:
    # Pool two runs into one cell only if they are the SAME experiment point except for
    # the seed. Keying on just (method, selection, knob) would silently merge distinct
    # operating points when a user sweeps a compute-affecting field that `_knob` doesn't
    # surface (beam_expansions, max_steps, temperature, …) — mislabeling several curve
    # points as one. A full-config signature (minus seed/output_dir) is the honest key.
    signature = cfg.to_dict()
    signature.pop("seed", None)
    signature.pop("output_dir", None)
    return json.dumps(signature, sort_keys=True, default=str)


def aggregate_cell(summaries: list[RunSummary]) -> dict[str, Any]:
    """Pool a cell's runs (across seeds) into one row with a Wilson CI."""
    cfg = summaries[0].config
    results = [r for s in summaries for r in s.results]
    total = len(results)
    correct = sum(1 for r in results if r.correct)
    low, high = wilson_interval(correct, total)
    tokens = sum(r.compute.total_tokens for r in results)
    return {
        "method": cfg.method,
        "selection": cfg.selection,
        "n": _knob(cfg),
        "seeds": len(summaries),
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "accuracy_ci_low": low,
        "accuracy_ci_high": high,
        "mean_tokens": tokens / total if total else 0.0,
    }


@dataclass
class SweepResult:
    sweep_dir: Path
    cells: list[dict[str, Any]]
    curve_path: Path


def run_sweep(config_path: str | Path) -> SweepResult:
    """Run every cell (× every seed) and write records + sweep.json + curve.png."""
    with open(config_path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError("sweep config must be a YAML mapping")

    grid = data.get("grid") or []
    if not grid:
        raise ValueError("sweep config needs a non-empty 'grid:' list")
    base = {str(k): v for k, v in data.items() if k not in ("grid", "seeds")}
    seeds = list(data.get("seeds") or [base.get("seed", 0)])
    output_dir = str(base.get("output_dir", "runs"))
    configs = expand_grid(base, list(grid))

    sweep_dir = Path(output_dir) / datetime.now().strftime("sweep-%Y-%m-%dT%H-%M-%S")
    sweep_dir.mkdir(parents=True, exist_ok=True)

    groups: dict[str, list[RunSummary]] = {}
    for i, cfg in enumerate(configs):
        for seed in seeds:
            scfg = replace(cfg, seed=seed)
            summary = run(scfg)
            name = f"{i:03d}-{scfg.method}-{scfg.selection}-n{_knob(scfg)}-seed{seed}"
            write_run_record(summary, base_dir=sweep_dir, name=name)
            groups.setdefault(_cell_key(scfg), []).append(summary)

    cells = [aggregate_cell(summaries) for summaries in groups.values()]
    cells.sort(key=lambda c: (c["method"], c["mean_tokens"]))

    (sweep_dir / "sweep.json").write_text(json.dumps(cells, indent=2), encoding="utf-8")
    curve_path = render_curve(cells, sweep_dir / "curve.png")
    return SweepResult(sweep_dir=sweep_dir, cells=cells, curve_path=curve_path)
