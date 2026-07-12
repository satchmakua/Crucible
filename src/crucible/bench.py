"""Real accuracy-vs-compute curves without a 3-hour GPU marathon (H1/H3).

Naively sweeping best-of-N over N re-generates samples for every cell. Instead: generate
``max_n`` samples per problem **once**, score each with the outcome verifier and (if set)
the PRM, and record everything to a **cassette**. The curve at every N and for every
selector (majority / oracle / prm) is then a pure computation over that cassette — so a
real run is captured once and its lift curve regenerates **offline in CI** (the standing
"run live once, commit a fixture" pattern, now covering the PRM side).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from crucible.config import RunConfig
from crucible.data import load_dataset
from crucible.runner import build_outcome_verifier, build_policy, build_process_verifier
from crucible.stats import wilson_interval
from crucible.verify import aggregate_scores, extract_final_answer


@dataclass
class TraceRecord:
    text: str
    tokens: int  # total generated tokens for this trace (policy side)
    correct: bool  # outcome verifier on this trace
    prm_score: float | None  # aggregate PRM score, or None if no PRM


@dataclass
class SampleRecord:
    problem_id: str
    gold: str | None
    difficulty: str | None
    traces: list[TraceRecord] = field(default_factory=list)


def record_samples(config: RunConfig, max_n: int) -> list[SampleRecord]:  # pragma: no cover - GPU
    """Generate `max_n` traces per problem once; score each with outcome + PRM."""
    problems = load_dataset(config.dataset, limit=config.limit)
    policy = build_policy(config)
    outcome = build_outcome_verifier(config)
    process = build_process_verifier(config)

    records: list[SampleRecord] = []
    for problem in problems:
        traces = policy.sample_full(
            problem,
            n=max_n,
            temperature=config.policy.temperature,
            max_tokens=config.policy.max_tokens,
        )
        trs: list[TraceRecord] = []
        for t in traces:
            prm_score = None
            if process is not None:
                prm_score = aggregate_scores(process.score_steps(problem, t.steps), config.prm_aggregate)
            trs.append(
                TraceRecord(
                    text=t.text,
                    tokens=t.compute.total_tokens,
                    correct=outcome.verify(problem, t).correct,
                    prm_score=prm_score,
                )
            )
        records.append(SampleRecord(problem.id, problem.answer, problem.difficulty, trs))
    return records


def save_samples(records: list[SampleRecord], path: str | Path, meta: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"meta": meta, "records": [asdict(r) for r in records]}, indent=2),
        encoding="utf-8",
    )
    return out


def load_samples(path: str | Path) -> list[SampleRecord]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        SampleRecord(
            problem_id=r["problem_id"],
            gold=r.get("gold"),
            difficulty=r.get("difficulty"),
            traces=[TraceRecord(**t) for t in r["traces"]],
        )
        for r in data["records"]
    ]


def _select(selector: str, traces: list[TraceRecord]) -> TraceRecord:
    if selector == "oracle":
        return next((t for t in traces if t.correct), traces[0])
    if selector == "prm":
        return max(traces, key=lambda t: (t.prm_score if t.prm_score is not None else -1.0))
    # majority: vote on extracted answers, return a representative of the modal answer
    answers = [extract_final_answer(t.text) for t in traces]
    counts = Counter(a for a in answers if a is not None)
    if not counts:
        return traces[0]
    top = counts.most_common(1)[0][0]
    for trace, answer in zip(traces, answers, strict=True):
        if answer == top:
            return trace
    return traces[0]


def curve_cells(
    records: list[SampleRecord], n_values: list[int], *, has_prm: bool
) -> list[dict[str, Any]]:
    """Compute accuracy-vs-compute cells (pass1 + best_of_n selectors) from the cassette."""
    cells: list[dict[str, Any]] = []

    # pass@1 — a single sample.
    tot = sum(1 for r in records if r.traces)
    cor = sum(1 for r in records if r.traces and r.traces[0].correct)
    tok = sum(r.traces[0].tokens for r in records if r.traces)
    low, high = wilson_interval(cor, tot)
    cells.append(_cell("pass1", "none", 1, cor, tot, low, high, tok / max(tot, 1)))

    selectors = ["majority", "oracle"] + (["prm"] if has_prm else [])
    for n in n_values:
        for selector in selectors:
            correct = total = gen_tokens = prm_tokens = 0
            for rec in records:
                trs = rec.traces[:n]
                if not trs:
                    continue
                total += 1
                gen_tokens += sum(t.tokens for t in trs)
                # The PRM reads every candidate → its forward-pass tokens are counted too.
                if selector == "prm":
                    prm_tokens += sum(t.tokens for t in trs)
                correct += 1 if _select(selector, trs).correct else 0
            low, high = wilson_interval(correct, total)
            mean_tokens = (gen_tokens + prm_tokens) / max(total, 1)
            cells.append(_cell("best_of_n", selector, n, correct, total, low, high, mean_tokens))
    return cells


def _cell(
    method: str, selection: str, n: int, correct: int, total: int, low: float, high: float, tokens: float
) -> dict[str, Any]:
    return {
        "method": method,
        "selection": selection,
        "n": n,
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "accuracy_ci_low": low,
        "accuracy_ci_high": high,
        "mean_tokens": tokens,
    }
