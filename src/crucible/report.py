"""Reporting: write a self-contained run record and print a human summary.

Every run writes its config, per-problem results, and aggregate metrics to a
timestamped directory, so any number in a report is traceable to the trace that
produced it (DESIGN.md §7). M0 emits JSON + CSV; the accuracy-vs-compute *curve*
(pandas + matplotlib) is the headline artifact built in M2+.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from crucible.runner import RunSummary

_CSV_FIELDS = [
    "problem_id",
    "method",
    "dataset",
    "correct",
    "predicted",
    "gold",
    "difficulty",
    "policy_gen_tokens",
    "policy_forward_calls",
    "verifier_forward_calls",
    "total_tokens",
    "wall_seconds",
]


def _summary_dict(summary: RunSummary) -> dict[str, Any]:
    c = summary.total_compute
    low, high = summary.accuracy_ci
    return {
        "method": summary.config.method,
        "dataset": summary.config.dataset,
        "policy_backend": summary.config.policy.backend,
        "policy_model": summary.config.policy.model,
        "total": summary.total,
        "correct": summary.correct,
        "accuracy": summary.accuracy,
        "accuracy_ci_low": low,
        "accuracy_ci_high": high,
        "compute": {
            "policy_gen_tokens": c.policy_gen_tokens,
            "policy_forward_calls": c.policy_forward_calls,
            "verifier_forward_calls": c.verifier_forward_calls,
            "verifier_gen_tokens": c.verifier_gen_tokens,
            "total_tokens": c.total_tokens,
            "wall_seconds": round(c.wall_seconds, 4),
        },
    }


def write_run_record(
    summary: RunSummary, base_dir: str | Path | None = None, name: str | None = None
) -> Path:
    """Persist the run to `<output_dir>/<name|timestamp>/` and return that directory."""
    root = Path(base_dir) if base_dir is not None else Path(summary.config.output_dir)
    run_dir = root / (name or datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f"))
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.json").write_text(
        json.dumps(summary.config.to_dict(), indent=2), encoding="utf-8"
    )
    (run_dir / "summary.json").write_text(
        json.dumps(_summary_dict(summary), indent=2), encoding="utf-8"
    )

    with (run_dir / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for r in summary.results:
            writer.writerow(
                {
                    "problem_id": r.problem_id,
                    "method": r.method,
                    "dataset": r.dataset,
                    "correct": int(r.correct),
                    "predicted": r.predicted,
                    "gold": r.gold,
                    "difficulty": r.difficulty,
                    "policy_gen_tokens": r.compute.policy_gen_tokens,
                    "policy_forward_calls": r.compute.policy_forward_calls,
                    "verifier_forward_calls": r.compute.verifier_forward_calls,
                    "total_tokens": r.compute.total_tokens,
                    "wall_seconds": round(r.compute.wall_seconds, 4),
                }
            )

    summary.run_dir = str(run_dir)
    return run_dir


def print_summary(summary: RunSummary, console: Console | None = None) -> None:
    """Pretty-print the per-problem results and the headline metrics."""
    console = console or Console()

    per = Table(title=f"{summary.config.method} / {summary.config.dataset}", title_style="bold")
    per.add_column("id")
    per.add_column("correct", justify="center")
    per.add_column("predicted")
    per.add_column("gold")
    per.add_column("tokens", justify="right")
    for r in summary.results:
        mark = "[green]yes[/green]" if r.correct else "[red]no[/red]"
        per.add_row(
            r.problem_id, mark, str(r.predicted), str(r.gold), str(r.compute.total_tokens)
        )
    console.print(per)

    c = summary.total_compute
    low, high = summary.accuracy_ci
    head = Table(show_header=False, box=None)
    head.add_column(style="bold")
    head.add_column()
    head.add_row("accuracy", f"{summary.accuracy:.1%}  ({summary.correct}/{summary.total})")
    head.add_row("95% CI (Wilson)", f"[{low:.1%}, {high:.1%}]")
    head.add_row("total tokens", f"{c.total_tokens:,}")
    head.add_row("verifier calls", f"{c.verifier_forward_calls:,}")
    head.add_row("wall seconds", f"{c.wall_seconds:.3f}")
    if summary.run_dir:
        head.add_row("record", summary.run_dir)
    console.print(head)


def read_summary(run_dir: str | Path) -> dict[str, Any]:
    """Load a previously written `summary.json`."""
    path = Path(run_dir) / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"no summary.json in {run_dir}")
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def print_record(data: dict[str, Any], console: Console | None = None) -> None:
    """Print a summary dict loaded from disk (used by the `report` command)."""
    console = console or Console()
    table = Table(show_header=False, box=None, title=f"{data['method']} / {data['dataset']}")
    table.add_column(style="bold")
    table.add_column()
    acc = data["accuracy"]
    table.add_row("accuracy", f"{acc:.1%}  ({data['correct']}/{data['total']})")
    table.add_row(
        "95% CI (Wilson)",
        f"[{data['accuracy_ci_low']:.1%}, {data['accuracy_ci_high']:.1%}]",
    )
    table.add_row("total tokens", f"{data['compute']['total_tokens']:,}")
    table.add_row("verifier calls", f"{data['compute']['verifier_forward_calls']:,}")
    console.print(table)
