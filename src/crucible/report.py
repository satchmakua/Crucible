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


def _curve_label(cell: dict[str, Any]) -> str:
    return "pass1" if cell["method"] == "pass1" else f"{cell['method']} ({cell['selection']})"


def print_sweep(cells: list[dict[str, Any]], console: Console | None = None) -> None:
    """Print a sweep's cells as an accuracy-vs-compute table."""
    console = console or Console()
    table = Table(title="sweep: accuracy vs compute", title_style="bold")
    table.add_column("method")
    table.add_column("N", justify="right")
    table.add_column("accuracy", justify="right")
    table.add_column("95% CI", justify="center")
    table.add_column("tokens/problem", justify="right")
    for cell in sorted(cells, key=lambda c: (_curve_label(c), c["mean_tokens"])):
        table.add_row(
            _curve_label(cell),
            str(cell["n"]),
            f"{cell['accuracy']:.1%}",
            f"[{cell['accuracy_ci_low']:.0%}, {cell['accuracy_ci_high']:.0%}]",
            f"{cell['mean_tokens']:.0f}",
        )
    console.print(table)


def render_curve(cells: list[dict[str, Any]], out_path: str | Path) -> Path:
    """Render the accuracy-vs-compute curve (one line per method/selector) to a PNG."""
    import matplotlib

    matplotlib.use("Agg")  # headless: no display needed
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    groups: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        groups.setdefault(_curve_label(cell), []).append(cell)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for label, items in sorted(groups.items()):
        pts = sorted(items, key=lambda c: c["mean_tokens"])
        xs = [c["mean_tokens"] for c in pts]
        ys = [c["accuracy"] for c in pts]
        # max(0, …) guards floating-point cases where a clamped Wilson bound lands a
        # hair past an accuracy of exactly 0 or 1 (matplotlib rejects negative yerr).
        yerr_lo = [max(0.0, c["accuracy"] - c["accuracy_ci_low"]) for c in pts]
        yerr_hi = [max(0.0, c["accuracy_ci_high"] - c["accuracy"]) for c in pts]
        ax.errorbar(xs, ys, yerr=[yerr_lo, yerr_hi], marker="o", capsize=3, label=label)

    ax.set_xscale("log")
    ax.set_xlabel("total tokens per problem (policy + verifier)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Accuracy vs test-time compute")
    ax.grid(visible=True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# --- Selection-gap comparison (M3) -------------------------------------------

_SELECTOR_ORDER = ("majority", "prm", "oracle")


def _ordered(summaries: dict[str, RunSummary]) -> list[str]:
    return [name for name in _SELECTOR_ORDER if name in summaries]


def print_comparison(summaries: dict[str, RunSummary], console: Console | None = None) -> None:
    """Print majority / prm / oracle accuracy on the same samples + gap to oracle."""
    console = console or Console()
    oracle_acc = summaries["oracle"].accuracy if "oracle" in summaries else 0.0
    table = Table(title="selection gap: best-of-N (same samples)", title_style="bold")
    table.add_column("selector")
    table.add_column("accuracy", justify="right")
    table.add_column("95% CI", justify="center")
    table.add_column("gap to oracle", justify="right")
    table.add_column("tokens/problem", justify="right")
    for name in _ordered(summaries):
        s = summaries[name]
        low, high = s.accuracy_ci
        denom = s.total or 1
        table.add_row(
            name,
            f"{s.accuracy:.1%}  ({s.correct}/{s.total})",
            f"[{low:.0%}, {high:.0%}]",
            f"{oracle_acc - s.accuracy:+.1%}",
            f"{s.total_compute.total_tokens / denom:.0f}",
        )
    console.print(table)


def render_comparison(summaries: dict[str, RunSummary], out_path: str | Path) -> Path:
    """Bar chart of accuracy per selector (same samples), with Wilson error bars."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    names = _ordered(summaries)
    accs = [summaries[n].accuracy for n in names]
    cis = [summaries[n].accuracy_ci for n in names]
    yerr_lo = [max(0.0, a - lo) for a, (lo, _hi) in zip(accs, cis, strict=True)]
    yerr_hi = [max(0.0, hi - a) for a, (_lo, hi) in zip(accs, cis, strict=True)]
    colors = ["#4C78A8", "#F58518", "#54A24B"][: len(names)]

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.bar(names, accs, yerr=[yerr_lo, yerr_hi], capsize=4, color=colors)
    for i, a in enumerate(accs):
        ax.text(i, min(a + 0.03, 0.99), f"{a:.0%}", ha="center")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("accuracy")
    ax.set_title("Selection gap (best-of-N, same samples)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def write_comparison_record(
    summaries: dict[str, RunSummary], base_dir: str | Path | None = None
) -> Path:
    """Write per-selector records + comparison.json + comparison.png; return the dir."""
    any_cfg = next(iter(summaries.values())).config
    root = Path(base_dir) if base_dir is not None else Path(any_cfg.output_dir)
    comp_dir = root / datetime.now().strftime("compare-%Y-%m-%dT%H-%M-%S")
    comp_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, Any] = {}
    for name, summary in summaries.items():
        write_run_record(summary, base_dir=comp_dir, name=name)
        low, high = summary.accuracy_ci
        denom = summary.total or 1
        metrics[name] = {
            "accuracy": summary.accuracy,
            "correct": summary.correct,
            "total": summary.total,
            "accuracy_ci_low": low,
            "accuracy_ci_high": high,
            "mean_tokens": summary.total_compute.total_tokens / denom,
        }
    (comp_dir / "comparison.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    render_comparison(summaries, comp_dir / "comparison.png")
    return comp_dir
