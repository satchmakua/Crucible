"""The `crucible` command-line front door (Typer over a YAML/flag config).

It's a harness, not a server: `run` executes one experiment and writes a record;
`report` prints a past run's metrics; `sweep` runs a grid; `compare` exposes the PRM's
selection gap; `bench` captures a real lift curve once and regenerates it offline.

Run the offline demo (no GPU, no network):

    crucible run --method pass1 --dataset sample --policy mock

Reproduce the real headline curve from the committed cassettes:

    crucible bench curve tests/fixtures/math500-bestofn-seed*.json
"""

from __future__ import annotations

import contextlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import httpx
import typer
from rich.console import Console

from crucible import __version__
from crucible.config import PolicyConfig, RunConfig
from crucible.report import (
    print_comparison,
    print_frontier,
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
bench_app = typer.Typer(
    no_args_is_help=True,
    help="Capture real accuracy-vs-compute curves once; regenerate them offline (H1/H3).",
)
app.add_typer(bench_app, name="bench")
console = Console()


@app.command()
def run(
    method: Annotated[str, typer.Option(help="search method: pass1 | best_of_n | beam | mcts")] = "pass1",
    dataset: Annotated[
        str,
        typer.Option(help="dataset: sample | gsm8k | math500 | math500-hard | code-sample | humaneval | mbpp"),
    ] = "sample",
    policy: Annotated[
        str, typer.Option(help="inference backend: mock | synthetic | stepwise | ollama")
    ] = "mock",
    model: Annotated[str, typer.Option(help="policy model id for the chosen backend")] = "scripted",
    n: Annotated[int, typer.Option(help="samples per problem (best_of_n)")] = 1,
    selection: Annotated[
        str, typer.Option(help="best_of_n selector: majority | oracle | prm")
    ] = "majority",
    temperature: Annotated[float, typer.Option(help="sampling temperature")] = 0.7,
    max_tokens: Annotated[int, typer.Option(help="max tokens per generation")] = 1024,
    limit: Annotated[int | None, typer.Option(help="cap the number of problems")] = None,
    seed: Annotated[int, typer.Option(help="random seed")] = 0,
    beam_width: Annotated[int, typer.Option(min=1, help="beam: partial traces kept per round (k)")] = 4,
    beam_expansions: Annotated[
        int, typer.Option(min=1, help="beam/mcts: continuations sampled per partial (m)")
    ] = 4,
    max_steps: Annotated[int, typer.Option(min=1, help="beam/mcts: hard cap on search depth")] = 8,
    budget_tokens: Annotated[
        int | None, typer.Option(help="mcts: total token budget per problem (policy + verifier)")
    ] = None,
    synthetic_accuracy: Annotated[
        float, typer.Option(help="per-problem correctness for --policy synthetic")
    ] = 0.5,
    prm: Annotated[
        str | None, typer.Option(help="PRM id for selection=prm ('mock' = simulator)")
    ] = None,
    prm_accuracy: Annotated[float, typer.Option(help="skill of the mock PRM")] = 0.8,
    allow_code_exec: Annotated[
        bool, typer.Option(help="enable the code-execution sandbox (off by default)")
    ] = False,
    record: Annotated[
        Path | None, typer.Option(help="record this run's generations to a JSON cassette (H3)")
    ] = None,
    record_prm: Annotated[
        Path | None, typer.Option(help="record this run's PRM scores to a JSON cassette (H3)")
    ] = None,
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
            beam_width=beam_width,
            beam_expansions=beam_expansions,
            max_steps=max_steps,
            budget_tokens=budget_tokens,
            synthetic_accuracy=synthetic_accuracy,
            prm=prm,
            prm_accuracy=prm_accuracy,
            allow_code_execution=allow_code_exec,
            record=str(record) if record is not None else None,
            record_prm=str(record_prm) if record_prm is not None else None,
            policy=PolicyConfig(
                backend=policy, model=model, temperature=temperature, max_tokens=max_tokens
            ),
            output_dir=str(output_dir),
        )

    try:
        summary = run_experiment(cfg)
    except (NotImplementedError, ValueError, RuntimeError) as exc:
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
        print_frontier(cells, console)
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
    except (NotImplementedError, ValueError, FileNotFoundError, RuntimeError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except httpx.HTTPError as exc:
        console.print(
            f"[red]error:[/red] inference backend unreachable ({type(exc).__name__}). {exc}"
        )
        raise typer.Exit(code=1) from exc

    print_sweep(result.cells, console)
    print_frontier(result.cells, console)
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
    except (NotImplementedError, ValueError, RuntimeError) as exc:
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


@bench_app.command("record")
def bench_record(
    out: Annotated[Path, typer.Option(help="cassette output path (JSON)")],
    dataset: Annotated[str, typer.Option(help="dataset name (math500 / math500-hard / gsm8k / ...)")] = "math500",
    policy: Annotated[str, typer.Option(help="inference backend: ollama | synthetic | mock")] = "ollama",
    model: Annotated[str, typer.Option(help="policy model id for the chosen backend")] = "qwen2.5:1.5b-instruct",
    prm: Annotated[
        str | None, typer.Option(help="PRM id to score every trace ('mock' = simulator)")
    ] = None,
    max_n: Annotated[int, typer.Option(min=1, help="samples generated per problem (the curve's max N)")] = 8,
    limit: Annotated[int | None, typer.Option(help="number of problems in this chunk")] = None,
    offset: Annotated[int, typer.Option(min=0, help="skip this many problems first (chunked captures)")] = 0,
    seed: Annotated[int, typer.Option(help="run seed (drives per-sample generation seeds)")] = 0,
    temperature: Annotated[float, typer.Option(help="sampling temperature")] = 0.7,
    max_tokens: Annotated[int, typer.Option(help="max tokens per generation")] = 1024,
    prm_aggregate: Annotated[str, typer.Option(help="per-step score reduction: mean | min | last | prod")] = "mean",
    synthetic_accuracy: Annotated[
        float, typer.Option(help="per-problem correctness for --policy synthetic")
    ] = 0.5,
    prm_accuracy: Annotated[float, typer.Option(help="skill of the mock PRM")] = 0.8,
) -> None:
    """Sample max-N traces per problem ONCE (scored by outcome + PRM) into a cassette.

    The expensive, live step of the lift curve — run it in short --offset chunks, then
    `crucible bench merge` the chunks and `crucible bench curve` the result offline.
    """
    from crucible.bench import (
        SampleRecord,
        curve_cells,
        default_n_values,
        record_samples,
        save_samples,
    )

    cfg = RunConfig(
        method="best_of_n",
        dataset=dataset,
        n=max_n,
        limit=limit,
        seed=seed,
        prm=prm,
        prm_accuracy=prm_accuracy,
        prm_aggregate=prm_aggregate,
        synthetic_accuracy=synthetic_accuracy,
        policy=PolicyConfig(
            backend=policy, model=model, temperature=temperature, max_tokens=max_tokens
        ),
    )

    def meta_for(collected: list[SampleRecord], *, partial: bool) -> dict[str, Any]:
        return {
            "dataset": dataset,
            "limit": limit,
            "offset": offset,
            "seed": seed,
            "max_n": max_n,
            "backend": policy,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prm": prm,
            "prm_aggregate": prm_aggregate,
            "partial": partial,
            "recorded": len(collected),
            "resume_offset": offset + len(collected),  # --offset for the next chunk
            "captured": datetime.now().isoformat(timespec="seconds"),
        }

    # Persist after every completed problem: a CUDA/TDR crash or timeout at problem 99
    # of a 100-problem chunk then costs one problem, not the whole GPU-hours chunk. The
    # partial cassette records the resume offset so the next run picks up where it died.
    collected: list[SampleRecord] = []

    def report_progress(done: int, total: int, rec: SampleRecord) -> None:
        collected.append(rec)
        save_samples(collected, out, meta_for(collected, partial=True))
        hits = sum(1 for t in rec.traces if t.correct)
        console.print(f"[{done}/{total}] {rec.problem_id}: {hits}/{len(rec.traces)} correct")

    try:
        records = record_samples(cfg, max_n, offset=offset, progress=report_progress)
    except (NotImplementedError, ValueError, RuntimeError) as exc:
        if collected:
            save_samples(collected, out, meta_for(collected, partial=True))
            console.print(
                f"[yellow]partial:[/yellow] saved {len(collected)} problems to {out}; "
                f"resume with --offset {offset + len(collected)}."
            )
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except (httpx.HTTPError, KeyboardInterrupt) as exc:
        if collected:
            save_samples(collected, out, meta_for(collected, partial=True))
            console.print(
                f"[yellow]partial:[/yellow] saved {len(collected)} problems to {out}; "
                f"resume with --offset {offset + len(collected)}."
            )
        console.print(
            f"[red]error:[/red] capture interrupted ({type(exc).__name__}). "
            "Is the backend up and the model pulled?"
        )
        raise typer.Exit(code=1) from exc

    path = save_samples(records, out, meta_for(records, partial=False))
    cells = curve_cells(records, default_n_values(max_n), has_prm=prm is not None)
    print_sweep(cells, console)
    print_frontier(cells, console)
    console.print(f"[green]cassette:[/green] {path} ({len(records)} problems x {max_n} traces)")


@bench_app.command("merge")
def bench_merge(
    cassettes: Annotated[list[Path], typer.Argument(help="chunk cassettes, in order")],
    out: Annotated[Path, typer.Option(help="merged cassette output path")],
) -> None:
    """Concatenate chunked capture cassettes of ONE seed into one per-seed cassette.

    Guards against the chunk-bookkeeping slips that would silently corrupt a committed
    fixture: a duplicate problem id (a re-recorded or double-listed chunk) is an error,
    and chunks that disagree on the experiment identity (dataset / model / seed / max_n)
    are refused — pooling *across* seeds is `bench curve`'s job, not merge's.
    """
    from crucible.bench import SampleRecord, load_samples_with_meta, save_samples

    def fail(message: str) -> None:
        console.print(f"[red]error:[/red] {message}")
        raise typer.Exit(code=1)

    all_records: list[SampleRecord] = []
    parts: list[dict[str, Any]] = []
    seen: dict[str, str] = {}  # problem_id -> cassette it first appeared in
    identity_keys = ("dataset", "backend", "model", "seed", "max_n", "temperature", "prm")
    identity: dict[str, Any] | None = None
    for path in cassettes:
        try:
            records, meta = load_samples_with_meta(path)
        except FileNotFoundError:
            fail(f"cassette not found: {path}")
        if meta.get("partial"):
            fail(f"{path.name} is a PARTIAL capture (an interrupted chunk); finish or re-record it before merging.")
        this_identity = {k: meta.get(k) for k in identity_keys}
        if identity is None:
            identity = this_identity
        elif this_identity != identity:
            diff = {k: (identity[k], this_identity[k]) for k in identity_keys if identity[k] != this_identity[k]}
            fail(f"{path.name} disagrees with earlier chunks on {diff}; merge is per-seed (use `bench curve` to pool seeds).")
        for rec in records:
            if rec.problem_id in seen:
                fail(f"duplicate problem '{rec.problem_id}' in {path.name} (already in {seen[rec.problem_id]}); check --offset bookkeeping.")
            seen[rec.problem_id] = path.name
        all_records.extend(records)
        parts.append(meta)

    merged_meta: dict[str, Any] = dict(identity or {})
    merged_meta.update({"merged_from": [p.name for p in cassettes], "parts": parts})
    merged = save_samples(all_records, out, merged_meta)
    console.print(f"[green]merged:[/green] {merged} ({len(all_records)} records)")


@bench_app.command("curve")
def bench_curve(
    cassettes: Annotated[list[Path], typer.Argument(help="bench cassettes to pool (e.g. one per seed)")],
    run: Annotated[
        list[Path] | None,
        typer.Option(help="overlay a live run dir (beam/MCTS) as a point on the curve"),
    ] = None,
    difficulty: Annotated[
        str | None, typer.Option(help="keep only these difficulties (comma-separated, e.g. 'level-4,level-5')")
    ] = None,
    restrict_to_runs: Annotated[
        bool, typer.Option(help="restrict cassette problems to those the --run dirs cover (matched-problem comparison)")
    ] = False,
    n_values: Annotated[
        str | None, typer.Option(help="comma-separated N values (default: powers of 2 up to the recorded max)")
    ] = None,
    out_dir: Annotated[
        Path | None, typer.Option(help="where cells.json + curve.png go (default runs/bench-<timestamp>)")
    ] = None,
) -> None:
    """The offline half: pool cassettes into the accuracy-vs-compute curve + frontier.

    Pure computation over recorded samples — no GPU, no network. Multiple cassettes
    pool (problems x seeds, like `sweep`); `--run` overlays live beam/MCTS records on
    the same honest compute axis.
    """
    from crucible.bench import (
        SampleRecord,
        cell_from_run,
        curve_cells,
        default_n_values,
        filter_records,
        load_samples,
        records_have_consistent_prm,
        run_problem_ids,
    )

    def fail(message: str) -> None:
        console.print(f"[red]error:[/red] {message}")
        raise typer.Exit(code=1)

    records: list[SampleRecord] = []
    for path in cassettes:
        try:
            records.extend(load_samples(path))
        except FileNotFoundError:
            fail(f"cassette not found: {path}")
    if not records:
        fail("the cassettes contain no records.")

    run_dirs = run or []
    for run_dir in run_dirs:
        if not (run_dir / "results.csv").exists():
            fail(f"run dir missing results.csv: {run_dir}")

    difficulties = {d.strip() for d in difficulty.split(",")} if difficulty else None
    problem_ids: set[str] | None = None
    if restrict_to_runs:
        if not run_dirs:
            fail("--restrict-to-runs needs at least one --run.")
        id_sets = [run_problem_ids(d) for d in run_dirs]
        problem_ids = set.intersection(*id_sets)
        if any(ids != problem_ids for ids in id_sets):
            console.print(
                "[yellow]note:[/yellow] --run dirs cover different problems; "
                "restricting to their intersection so every line spans the same set."
            )

    records = filter_records(records, difficulties=difficulties, problem_ids=problem_ids)
    if not records:
        fail("no records left after filtering.")

    # A curve pooled from PRM-scored and unscored records would draw a 'prm' line that
    # is a silent blend of PRM selection and pass@1 (unscored records fall back to the
    # first sample) while still charging PRM compute — refuse it (DESIGN §2, §4.4).
    any_prm, all_prm = records_have_consistent_prm(records)
    if any_prm and not all_prm:
        fail(
            "pooled cassettes mix PRM-scored and unscored records; the 'prm' line would "
            "silently blend in pass@1. Re-capture the unscored chunk with --prm, or pool "
            "only scored cassettes."
        )

    max_n = min(len(r.traces) for r in records)
    if n_values:
        try:
            values = [int(v) for v in n_values.split(",")]
        except ValueError:
            fail(f"--n-values must be comma-separated integers, got {n_values!r}.")
        bad = [v for v in values if v < 1 or v > max_n]
        if bad:
            fail(f"--n-values {bad} out of range; recorded traces per problem = {max_n} (min N=1).")
    else:
        values = default_n_values(max_n)

    cells = curve_cells(records, values, has_prm=all_prm)
    for cell in cells:
        cell["source"] = "cassette"
    # Overlay each run on exactly the problem/difficulty subset the cassette cells cover,
    # so the comparison is genuinely matched. If a run doesn't cover the whole set, say
    # so — an overlay silently computed on a different set is the trap this guards.
    final_ids = {r.problem_id for r in records}
    for run_dir in run_dirs:
        covered = run_problem_ids(run_dir) & final_ids
        if covered != final_ids:
            console.print(
                f"[yellow]note:[/yellow] {run_dir.name} covers {len(covered)}/{len(final_ids)} "
                "of the curve's problems; its overlay cell is computed on that overlap only."
            )
        cells.append(cell_from_run(run_dir, keep_ids=final_ids, difficulties=difficulties))

    target = out_dir or Path("runs") / datetime.now().strftime("bench-%Y-%m-%dT%H-%M-%S")
    target.mkdir(parents=True, exist_ok=True)
    (target / "cells.json").write_text(json.dumps(cells, indent=2), encoding="utf-8")
    curve = render_curve(cells, target / "curve.png")
    print_sweep(cells, console)
    print_frontier(cells, console)
    console.print(f"[green]cells:[/green] {target / 'cells.json'}")
    console.print(f"[green]curve:[/green] {curve}")


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
