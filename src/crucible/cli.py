"""The `crucible` command-line front door (Typer over a YAML/flag config).

It's a harness, not a server: `run` executes one experiment and writes a record;
`report` prints a past run's metrics; `sweep` (the grid → headline curve) arrives in
M2. Run the offline M0 demo with:

    crucible run --method pass1 --dataset sample --policy mock
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from typing import Annotated

import httpx
import typer
from rich.console import Console

from crucible import __version__
from crucible.config import PolicyConfig, RunConfig
from crucible.report import (
    print_comparison,
    print_record,
    print_summary,
    print_sweep,
    read_summary,
    render_curve,
    write_comparison_record,
    write_run_record,
)
from crucible.runner import run as run_experiment
from crucible.runner import run_comparison

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
    n: Annotated[int, typer.Option(help="samples per problem (best_of_n)")] = 1,
    selection: Annotated[
        str, typer.Option(help="best_of_n selector: majority | oracle")
    ] = "majority",
    temperature: Annotated[float, typer.Option(help="sampling temperature")] = 0.7,
    max_tokens: Annotated[int, typer.Option(help="max tokens per generation")] = 1024,
    limit: Annotated[int | None, typer.Option(help="cap the number of problems")] = None,
    seed: Annotated[int, typer.Option(help="random seed")] = 0,
    synthetic_accuracy: Annotated[
        float, typer.Option(help="per-problem correctness for --policy synthetic")
    ] = 0.5,
    prm: Annotated[
        str | None, typer.Option(help="PRM id for selection=prm ('mock' = simulator)")
    ] = None,
    prm_accuracy: Annotated[float, typer.Option(help="skill of the mock PRM")] = 0.8,
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
            selection=selection,
            seed=seed,
            limit=limit,
            synthetic_accuracy=synthetic_accuracy,
            prm=prm,
            prm_accuracy=prm_accuracy,
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
    run_dir: Annotated[Path, typer.Argument(help="a run or sweep directory")],
) -> None:
    """Print metrics from a past run, or re-render a sweep's curve."""
    sweep_json = run_dir / "sweep.json"
    if sweep_json.exists():
        cells = json.loads(sweep_json.read_text(encoding="utf-8"))
        print_sweep(cells, console)
        curve = render_curve(cells, run_dir / "curve.png")
        console.print(f"[green]curve:[/green] {curve}")
        return
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
    """Run a grid of experiments → the accuracy-vs-compute curve."""
    from crucible.sweep import run_sweep

    try:
        result = run_sweep(config)
    except (NotImplementedError, ValueError, FileNotFoundError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except httpx.HTTPError as exc:
        console.print(
            f"[red]error:[/red] inference backend unreachable ({type(exc).__name__}). {exc}"
        )
        raise typer.Exit(code=1) from exc

    print_sweep(result.cells, console)
    console.print(f"[green]curve:[/green] {result.curve_path}")
    console.print(f"[green]sweep:[/green] {result.sweep_dir}")


@app.command()
def compare(
    dataset: Annotated[str, typer.Option(help="dataset name")] = "sample",
    policy: Annotated[str, typer.Option(help="inference backend: mock | synthetic | ollama")] = "synthetic",
    model: Annotated[str, typer.Option(help="policy model id")] = "sim",
    n: Annotated[int, typer.Option(help="samples per problem")] = 8,
    prm: Annotated[str, typer.Option(help="PRM id ('mock' = simulator)")] = "mock",
    prm_accuracy: Annotated[float, typer.Option(help="skill of the mock PRM (illustrative)")] = 0.3,
    synthetic_accuracy: Annotated[float, typer.Option(help="accuracy for --policy synthetic")] = 0.3,
    temperature: Annotated[float, typer.Option(help="sampling temperature")] = 0.7,
    max_tokens: Annotated[int, typer.Option(help="max tokens per generation")] = 1024,
    limit: Annotated[int | None, typer.Option(help="cap the number of problems")] = None,
    seed: Annotated[int, typer.Option(help="random seed")] = 0,
    output_dir: Annotated[Path, typer.Option(help="where records are written")] = Path("runs"),
    save: Annotated[bool, typer.Option(help="write records + comparison.png")] = True,
) -> None:
    """Compare majority / PRM / oracle selection on the SAME best-of-N samples.

    Exposes the PRM's selection gap: with a real PRM, oracle >= prm >= majority.
    """
    cfg = RunConfig(
        method="best_of_n",
        dataset=dataset,
        n=n,
        seed=seed,
        limit=limit,
        prm=prm,
        prm_accuracy=prm_accuracy,
        synthetic_accuracy=synthetic_accuracy,
        policy=PolicyConfig(
            backend=policy, model=model, temperature=temperature, max_tokens=max_tokens
        ),
        output_dir=str(output_dir),
    )
    try:
        summaries = run_comparison(cfg)
    except (NotImplementedError, ValueError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except httpx.HTTPError as exc:
        console.print(
            f"[red]error:[/red] inference backend unreachable ({type(exc).__name__}). {exc}"
        )
        raise typer.Exit(code=1) from exc

    print_comparison(summaries, console)
    if save:
        comp_dir = write_comparison_record(summaries)
        console.print(f"[green]comparison:[/green] {comp_dir}")


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
