"""Cassettes: record real model calls once, replay them offline forever (ROADMAP H3).

The standing pattern (CLAUDE.md / DESIGN §9): **run live once, then commit a recorded
fixture so CI reproduces the numbers without a GPU.** Two recorder/replayer pairs live
here because they share the cassette-key convention:

- `RecordingPolicy` / `CassettePolicy` — generation. Full traces (`sample_full`) are
  keyed by problem id; stepwise continuations (`sample_step`) are keyed by
  ``step_key(problem_id, prefix)`` with FIFO batches per key, so best-of-N *and*
  beam/MCTS runs replay call-for-call.
- `RecordingProcessVerifier` / `CassetteProcessVerifier` — the PRM. Every
  `score_steps` call is keyed the same way, so a PRM-guided search replays with no
  GPU. A missing key on replay raises (search drift is a real failure, not a zero).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, NamedTuple

from crucible.domain.ports import PolicyModel, ProcessVerifier
from crucible.domain.types import Compute, Problem, Step, Trace


def step_key(problem_id: str, prefix: list[Step]) -> str:
    """Stable identity of a (problem, partial-trace) call across record and replay."""
    digest = hashlib.sha256("\x1f".join(s.text for s in prefix).encode("utf-8")).hexdigest()
    return f"{problem_id}:{digest[:16]}"


def _trace_to_dict(trace: Trace) -> dict[str, Any]:
    c = trace.compute
    return {
        "steps": [{"text": s.text, "token_count": s.token_count} for s in trace.steps],
        "compute": {
            "policy_gen_tokens": c.policy_gen_tokens,
            "policy_forward_calls": c.policy_forward_calls,
            "wall_seconds": c.wall_seconds,
        },
    }


def _trace_from_dict(data: dict[str, Any]) -> Trace:
    steps = [Step(text=s["text"], token_count=int(s["token_count"])) for s in data["steps"]]
    c = data.get("compute", {})
    compute = Compute(
        policy_gen_tokens=int(c.get("policy_gen_tokens", 0)),
        policy_forward_calls=int(c.get("policy_forward_calls", 0)),
        wall_seconds=float(c.get("wall_seconds", 0.0)),
    )
    return Trace(steps=steps, final_answer=None, compute=compute)


def _problem_to_dict(p: Problem) -> dict[str, Any]:
    return {"id": p.id, "prompt": p.prompt, "answer": p.answer, "difficulty": p.difficulty}


def _problem_from_dict(data: dict[str, Any]) -> Problem:
    return Problem(
        id=str(data["id"]),
        prompt=str(data["prompt"]),
        answer=data.get("answer"),
        difficulty=data.get("difficulty"),
    )


class RecordingPolicy:
    """Wraps a real `PolicyModel`, forwarding calls and recording every response."""

    name = "recording"

    def __init__(self, inner: PolicyModel, path: str | Path) -> None:
        self._inner = inner
        self._path = Path(path)
        self._records: dict[str, tuple[Problem, list[Trace]]] = {}
        self._problems: dict[str, Problem] = {}
        self._step_records: dict[str, list[list[Step]]] = {}
        self._sampling: dict[str, Any] = {}

    def _note_sampling(self, temperature: float, max_tokens: int) -> None:
        # Provenance: a committed fixture should say what produced it, not just "ollama".
        self._sampling.setdefault("temperature", temperature)
        self._sampling.setdefault("max_tokens", max_tokens)

    def sample_full(
        self, problem: Problem, *, n: int, temperature: float, max_tokens: int
    ) -> list[Trace]:
        traces = self._inner.sample_full(
            problem, n=n, temperature=temperature, max_tokens=max_tokens
        )
        self._records[problem.id] = (problem, traces)
        self._problems[problem.id] = problem
        self._note_sampling(temperature, max_tokens)
        self._sampling.setdefault("n", n)
        return traces

    def sample_step(
        self, problem: Problem, prefix: list[Step], *, n: int, temperature: float, max_tokens: int
    ) -> list[Step]:
        steps = self._inner.sample_step(
            problem, prefix, n=n, temperature=temperature, max_tokens=max_tokens
        )
        self._note_sampling(temperature, max_tokens)
        # FIFO batches per key: two identical prefixes in one search (rare but legal)
        # record two batches and replay in the same order.
        self._step_records.setdefault(step_key(problem.id, prefix), []).append(steps)
        self._problems[problem.id] = problem
        return steps

    def save(self) -> Path:
        """Write the captured problems + traces + step batches to a JSON cassette."""
        data = {
            "backend": getattr(self._inner, "name", "unknown"),
            # Self-describing provenance: which model, sampled how. Without this a
            # committed fixture can't be traced back to the run that produced it.
            "model": getattr(self._inner, "model", None),
            "seed": getattr(self._inner, "_seed", None),
            "sampling": dict(self._sampling),
            "records": [
                {"problem": _problem_to_dict(p), "traces": [_trace_to_dict(t) for t in ts]}
                for p, ts in self._records.values()
            ],
            # Beam/MCTS runs never call sample_full, so the problems they touched are
            # saved separately — a step-only cassette is still self-contained.
            "problems": [_problem_to_dict(p) for p in self._problems.values()],
            "step_records": [
                {
                    "key": key,
                    "batches": [
                        [{"text": s.text, "token_count": s.token_count} for s in batch]
                        for batch in batches
                    ],
                }
                for key, batches in self._step_records.items()
            ],
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return self._path


class CassettePolicy:
    """Replays a recorded cassette — no network, no model, fully deterministic."""

    name = "cassette"

    def __init__(
        self,
        records: dict[str, list[Trace]],
        steps: dict[str, list[list[Step]]] | None = None,
    ) -> None:
        self._records = records
        self._steps = steps or {}
        self._cursors: dict[str, int] = {}

    def sample_full(
        self, problem: Problem, *, n: int, temperature: float, max_tokens: int
    ) -> list[Trace]:
        traces = self._records.get(problem.id)
        if not traces:
            return [Trace(steps=[], final_answer=None, compute=Compute()) for _ in range(n)]
        return [traces[i % len(traces)] for i in range(n)]

    def sample_step(
        self, problem: Problem, prefix: list[Step], *, n: int, temperature: float, max_tokens: int
    ) -> list[Step]:
        # Symmetric with CassetteProcessVerifier: a miss (or asking more often than the
        # recorded run did) means the replayed search diverged from what was captured —
        # a real failure, not a zero. Fabricating empty steps here would let a diverged
        # beam/MCTS replay report confident wrong numbers.
        key = step_key(problem.id, prefix)
        batches = self._steps.get(key)
        cursor = self._cursors.get(key, 0)
        if not batches or cursor >= len(batches):
            raise KeyError(
                f"step cassette exhausted for {key} — the replayed search diverged "
                "from the recorded run (config or code drift)."
            )
        self._cursors[key] = cursor + 1
        return list(batches[cursor])


class CassetteBundle(NamedTuple):
    """Everything a cassette holds: problems, full traces, and step batches."""

    problems: list[Problem]
    traces: dict[str, list[Trace]]
    steps: dict[str, list[list[Step]]]


def load_cassette(path: str | Path) -> tuple[list[Problem], dict[str, list[Trace]]]:
    """Load a cassette into (problems, id → traces) for offline replay."""
    bundle = load_bundle(path)
    return bundle.problems, bundle.traces


def load_bundle(path: str | Path) -> CassetteBundle:
    """Load a full cassette (traces + step batches); older trace-only files still work."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    problems: list[Problem] = []
    seen: set[str] = set()
    records: dict[str, list[Trace]] = {}
    for rec in data["records"]:
        problem = _problem_from_dict(rec["problem"])
        problems.append(problem)
        seen.add(problem.id)
        records[problem.id] = [_trace_from_dict(t) for t in rec["traces"]]
    for raw in data.get("problems", []):
        problem = _problem_from_dict(raw)
        if problem.id not in seen:
            problems.append(problem)
            seen.add(problem.id)
    steps: dict[str, list[list[Step]]] = {
        rec["key"]: [
            [Step(text=s["text"], token_count=int(s["token_count"])) for s in batch]
            for batch in rec["batches"]
        ]
        for rec in data.get("step_records", [])
    }
    return CassetteBundle(problems=problems, traces=records, steps=steps)


# --- The PRM side (H3 remainder): record/replay `score_steps` calls. -----------------


class RecordingProcessVerifier:
    """Wraps a real `ProcessVerifier`, recording every score_steps call by step key."""

    name = "recording-prm"

    def __init__(self, inner: ProcessVerifier, path: str | Path) -> None:
        self._inner = inner
        self._path = Path(path)
        self._records: dict[str, list[float]] = {}

    def score_steps(self, problem: Problem, prefix: list[Step]) -> list[float]:
        scores = self._inner.score_steps(problem, prefix)
        # A deterministic PRM returns the same scores for the same prefix, so a repeat
        # key (e.g. a terminal beam rescored each round) just overwrites with equals.
        self._records[step_key(problem.id, prefix)] = list(scores)
        return list(scores)

    def save(self) -> Path:
        data = {
            "prm": getattr(self._inner, "name", "unknown"),
            "model_id": getattr(self._inner, "model_id", None),
            "records": self._records,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return self._path


class CassetteProcessVerifier:
    """Replays recorded PRM scores. A missing key means the search diverged — raise."""

    name = "cassette-prm"

    def __init__(self, records: dict[str, list[float]]) -> None:
        self._records = records

    def score_steps(self, problem: Problem, prefix: list[Step]) -> list[float]:
        key = step_key(problem.id, prefix)
        if key not in self._records:
            raise KeyError(
                f"PRM cassette has no scores for {key} — the replayed search diverged "
                "from the recorded run (config or code drift)."
            )
        return list(self._records[key])


def load_prm_cassette(path: str | Path) -> dict[str, list[float]]:
    """Load a PRM cassette into the key → scores mapping `CassetteProcessVerifier` takes."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(k): [float(v) for v in vs] for k, vs in data["records"].items()}
