"""The `crucible bench` verb: record → merge → curve, cold (synthetic policy + mock PRM).

The GPU path uses the same code with `--policy ollama` and a real PRM id — these tests
prove the capture/replay plumbing so a live capture is one command, not an interactive
Python session.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from crucible.bench import (
    SampleRecord,
    TraceRecord,
    cell_from_run,
    default_n_values,
    filter_records,
    load_samples,
    load_samples_with_meta,
    run_problem_ids,
)
from crucible.cli import app
from crucible.config import PolicyConfig, RunConfig
from crucible.data import load_dataset
from crucible.report import write_run_record
from crucible.runner import run

runner = CliRunner()


def _record(
    tmp_path: Path, name: str, *, seed: int, prm: str | None = "mock", extra: list[str] | None = None
) -> Path:
    out = tmp_path / name
    args = [
        "bench", "record",
        "--policy", "synthetic", "--model", "sim",
        "--dataset", "sample", "--max-n", "4", "--seed", str(seed),
        "--out", str(out),
    ] + (["--prm", prm] if prm else []) + (extra or [])
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    return out


def test_bench_record_writes_cassette_with_meta_and_prm_scores(tmp_path: Path) -> None:
    out = _record(tmp_path, "s1.json", seed=1)
    records, meta = load_samples_with_meta(out)
    assert len(records) == 6  # the bundled sample set
    assert all(len(r.traces) == 4 for r in records)
    assert all(t.prm_score is not None for r in records for t in r.traces)
    assert meta["seed"] == 1 and meta["max_n"] == 4 and meta["backend"] == "synthetic"


def test_bench_record_offset_selects_a_chunk(tmp_path: Path) -> None:
    out = _record(tmp_path, "chunk.json", seed=0, extra=["--limit", "2", "--offset", "2"])
    expected = [p.id for p in load_dataset("sample")][2:4]
    assert [r.problem_id for r in load_samples(out)] == expected


def test_bench_merge_concatenates_chunks(tmp_path: Path) -> None:
    a = _record(tmp_path, "a.json", seed=0, extra=["--limit", "2"])
    b = _record(tmp_path, "b.json", seed=0, extra=["--limit", "2", "--offset", "2"])
    merged = tmp_path / "merged.json"
    result = runner.invoke(app, ["bench", "merge", str(a), str(b), "--out", str(merged)])
    assert result.exit_code == 0, result.output
    records, meta = load_samples_with_meta(merged)
    assert len(records) == 4
    assert [r.problem_id for r in records] == [p.id for p in load_dataset("sample")][:4]
    assert meta["merged_from"] == ["a.json", "b.json"] and len(meta["parts"]) == 2


def test_bench_curve_pools_seeds_and_renders(tmp_path: Path) -> None:
    c1 = _record(tmp_path, "seed1.json", seed=1)
    c2 = _record(tmp_path, "seed2.json", seed=2)
    out_dir = tmp_path / "curve"
    result = runner.invoke(app, ["bench", "curve", str(c1), str(c2), "--out-dir", str(out_dir)])
    assert result.exit_code == 0, result.output
    assert (out_dir / "curve.png").exists()
    import json

    cells = json.loads((out_dir / "cells.json").read_text(encoding="utf-8"))
    pass1 = next(c for c in cells if c["method"] == "pass1")
    assert pass1["total"] == 12  # 6 problems x 2 seeds, pooled like `sweep`
    assert {c["selection"] for c in cells if c["method"] == "best_of_n"} == {
        "majority", "oracle", "prm",
    }


def test_bench_curve_overlays_and_restricts_to_a_run(tmp_path: Path) -> None:
    cassette = _record(tmp_path, "full.json", seed=1)
    cfg = RunConfig(
        method="best_of_n", dataset="sample", n=2, selection="oracle", limit=3,
        policy=PolicyConfig(backend="synthetic", model="sim"),
    )
    run_dir = write_run_record(run(cfg), base_dir=tmp_path, name="live-run")
    out_dir = tmp_path / "curve"
    result = runner.invoke(
        app,
        ["bench", "curve", str(cassette), "--run", str(run_dir), "--restrict-to-runs",
         "--out-dir", str(out_dir)],
    )
    assert result.exit_code == 0, result.output
    import json

    cells = json.loads((out_dir / "cells.json").read_text(encoding="utf-8"))
    # Cassette cells restricted to the run's 3 problems; the run overlaid as a cell.
    assert next(c for c in cells if c["method"] == "pass1")["total"] == 3
    overlay = [c for c in cells if c["method"] == "best_of_n" and c["total"] == 3 and c["n"] == 2]
    assert any(c["selection"] == "oracle" for c in overlay)


def test_default_n_values_ladder() -> None:
    assert default_n_values(1) == [1]
    assert default_n_values(8) == [1, 2, 4, 8]
    assert default_n_values(6) == [1, 2, 4, 6]


def test_filter_records_by_difficulty_and_ids() -> None:
    def rec(pid: str, difficulty: str) -> SampleRecord:
        return SampleRecord(pid, "1", difficulty, [TraceRecord("\\boxed{1}", 3, True, None)])

    records = [rec("a", "level-1"), rec("b", "level-4"), rec("c", "level-5")]
    hard = filter_records(records, difficulties={"level-4", "level-5"})
    assert [r.problem_id for r in hard] == ["b", "c"]
    only_c = filter_records(records, difficulties={"level-4", "level-5"}, problem_ids={"c"})
    assert [r.problem_id for r in only_c] == ["c"]


def test_cell_from_run_reads_beam_knob_and_compute(tmp_path: Path) -> None:
    cfg = RunConfig(
        method="beam", dataset="sample", beam_width=2, beam_expansions=2, max_steps=6,
        prm="step", step_depth=3,
        policy=PolicyConfig(backend="stepwise", model="sim"),
    )
    run_dir = write_run_record(run(cfg), base_dir=tmp_path, name="beam-run")
    cell = cell_from_run(run_dir)
    assert cell["method"] == "beam" and cell["n"] == 2 and cell["total"] == 6
    assert cell["mean_tokens"] > 0
    assert cell["source"] == "run:beam-run"
    assert run_problem_ids(run_dir) == {p.id for p in load_dataset("sample")}


def test_cell_from_run_honors_keep_ids(tmp_path: Path) -> None:
    # The matched-comparison guarantee: an overlay restricted to a subset reports that
    # subset's accuracy + a fresh CI, not the full run's summary.
    cfg = RunConfig(
        method="best_of_n", dataset="sample", n=2, selection="oracle",
        policy=PolicyConfig(backend="synthetic", model="sim"),
    )
    run_dir = write_run_record(run(cfg), base_dir=tmp_path, name="run-a")
    keep = {p.id for p in load_dataset("sample")[:2]}
    cell = cell_from_run(run_dir, keep_ids=keep)
    assert cell["total"] == 2


# --- Guards against silent corruption of a committed fixture (review findings). ------


def test_bench_curve_rejects_mixed_prm_pool(tmp_path: Path) -> None:
    scored = _record(tmp_path, "scored.json", seed=1, prm="mock")
    unscored = _record(tmp_path, "unscored.json", seed=2, prm=None)
    result = runner.invoke(app, ["bench", "curve", str(scored), str(unscored)])
    assert result.exit_code == 1
    assert "mix" in result.output.lower() and "prm" in result.output.lower()


@pytest.mark.parametrize("spec", ["16", "0", "-1", "4,banana"])
def test_bench_curve_rejects_bad_n_values(tmp_path: Path, spec: str) -> None:
    cassette = _record(tmp_path, "c.json", seed=1)  # max_n = 4
    result = runner.invoke(app, ["bench", "curve", str(cassette), "--n-values", spec])
    assert result.exit_code == 1, result.output
    assert "n-values" in result.output.lower()


def test_bench_curve_missing_cassette_exits_cleanly(tmp_path: Path) -> None:
    result = runner.invoke(app, ["bench", "curve", str(tmp_path / "nope.json")])
    assert result.exit_code == 1
    assert "not found" in result.output.lower()
    assert "Traceback" not in result.output


def test_bench_merge_rejects_duplicate_problem(tmp_path: Path) -> None:
    cassette = _record(tmp_path, "c.json", seed=0)
    result = runner.invoke(app, ["bench", "merge", str(cassette), str(cassette), "--out", str(tmp_path / "m.json")])
    assert result.exit_code == 1
    assert "duplicate" in result.output.lower()


def test_bench_merge_rejects_seed_mismatch(tmp_path: Path) -> None:
    a = _record(tmp_path, "a.json", seed=0, extra=["--limit", "2"])
    b = _record(tmp_path, "b.json", seed=1, extra=["--limit", "2", "--offset", "2"])
    result = runner.invoke(app, ["bench", "merge", str(a), str(b), "--out", str(tmp_path / "m.json")])
    assert result.exit_code == 1
    assert "seed" in result.output.lower() or "disagree" in result.output.lower()


def test_bench_merge_carries_identity_meta(tmp_path: Path) -> None:
    a = _record(tmp_path, "a.json", seed=0, extra=["--limit", "2"])
    b = _record(tmp_path, "b.json", seed=0, extra=["--limit", "2", "--offset", "2"])
    merged = tmp_path / "m.json"
    result = runner.invoke(app, ["bench", "merge", str(a), str(b), "--out", str(merged)])
    assert result.exit_code == 0, result.output
    _records, meta = load_samples_with_meta(merged)
    assert meta["seed"] == 0 and meta["dataset"] == "sample" and meta["model"] == "sim"


def test_bench_record_marks_complete_and_resume_offset(tmp_path: Path) -> None:
    out = _record(tmp_path, "chunk.json", seed=0, extra=["--limit", "3", "--offset", "2"])
    _records, meta = load_samples_with_meta(out)
    assert meta["partial"] is False
    assert meta["recorded"] == 3 and meta["resume_offset"] == 5  # offset 2 + 3 recorded


def test_bench_record_rejects_max_n_zero(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["bench", "record", "--policy", "synthetic", "--model", "sim", "--dataset", "sample",
         "--max-n", "0", "--out", str(tmp_path / "z.json")],
    )
    assert result.exit_code != 0  # typer rejects min=1 before any capture
