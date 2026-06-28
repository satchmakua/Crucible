"""The `crucible` command-line front door (Typer over a YAML/flag config).

It's a harness, not a server: `run` executes one experiment and writes a record;
`report` prints a past run's metrics; `sweep` (the grid → headline curve) arrives in
M2. Run the offline M0 demo with:

    crucible run --method pass1 --dataset sample --policy mock
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Annotated

import httpx
import typer
from rich.console import Console

from crucible import __version__
from crucible.config import PolicyConfig, RunConfig
from crucible.report import print_record, print_summary, read_summary, write_run_record
from crucible.runner import run as run_experiment

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Crucible — verifier-guided reasoning search over test-time compute.",
)
console = Console()


@app.command()
def run(
    method: Annotated[str, typer.Option(help="search method (pass1; more in later milestones)")] = "pass1",
    dataset: Annotated[str, typer.Option(help="dataset (sample bundled; gsm8k/math500 in M1)")] = "sample",
    policy: Annotated[str, typer.Option(help="inference backend: mock | ollama | hosted")] = "mock",
    model: Annotated[str, typer.Option(help="policy model id for the chosen backend")] = "scripted",
    n: Annotated[int, typer.Option(help="samples per problem")] = 1,
    temperature: Annotated[float, typer.Option(help="sampling temperature")] = 0.7,
    max_tokens: Annotated[int, typer.Option(help="max tokens per generation")] = 1024,
    limit: Annotated[int | None, typer.Option(help="cap the number of problems")] = None,
    seed: Annotated[int, typer.Option(help="random seed")] = 0,
    config: Annotated[
        Path | None, typer.Option("--config", help="YAML config; if given, other flags are ignored")
    ] = None,
    output_dir: Annotated[Path, typer.Option(help="where run records are written")] = Path("runs"),
    save: Annotated[bool, typer.Option(help="write a run record to --output-dir")] = True,
) -> None:
    """Execute one experiment and report its accuracy and compute."""
    if config is not None:
        cfg = RunConfig.from_yaml(config)
    else:
        cfg = RunConfig(
            method=method,
            dataset=dataset,
            n=n,
            seed=seed,
            limit=limit,
            policy=PolicyConfig(
                backend=policy, model=model, temperature=temperature, max_tokens=max_tokens
            ),
            output_dir=str(output_dir),
        )

    try:
        summary = run_experiment(cfg)
    except (NotImplementedError, ValueError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except httpx.HTTPError as exc:
        console.print(
            f"[red]error:[/red] could not reach the '{cfg.policy.backend}' backend "
            f"({type(exc).__name__}). Is the server running and the model pulled? {exc}"
        )
        raise typer.Exit(code=1) from exc

    if save:
        write_run_record(summary)
    print_summary(summary, console)


@app.command()
def report(
    run_dir: Annotated[Path, typer.Argument(help="a run directory written by `run`")],
) -> None:
    """Print the metrics from a past run directory."""
    try:
        data = read_summary(run_dir)
    except FileNotFoundError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    print_record(data, console)


@app.command()
def sweep(
    config: Annotated[Path, typer.Argument(help="a sweep YAML (grid of runs)")],
) -> None:
    """Run a grid of experiments → the accuracy-vs-compute curve. (Lands in M2.)"""
    console.print(
        "[yellow]sweep[/yellow] is planned for milestone M2 (the headline "
        "accuracy-vs-compute curve). See ROADMAP.md. Use `crucible run` for now."
    )
    raise typer.Exit(code=1)


@app.command()
def version() -> None:
    """Print the installed Crucible version."""
    console.print(f"crucible {__version__}")


def main() -> None:
    # Make output robust on consoles whose code page can't encode UTF-8 (common on
    # Windows when piped). Console text is ASCII-safe regardless; this just prevents a
    # crash if a future code path prints richer content.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(ValueError, OSError):
                reconfigure(encoding="utf-8", errors="replace")
    app()


if __name__ == "__main__":
    main()
