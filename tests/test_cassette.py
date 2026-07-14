"""Cassette record/replay (H3): a run's generations reproduce offline in CI."""

from __future__ import annotations

from pathlib import Path

import pytest

from crucible.config import PolicyConfig, RunConfig
from crucible.domain.types import Problem, Step
from crucible.inference import (
    CassettePolicy,
    CassetteProcessVerifier,
    load_bundle,
    load_cassette,
    load_prm_cassette,
)
from crucible.runner import run
from crucible.search.beam import BeamStrategy
from crucible.search.mcts import MCTSStrategy
from crucible.verify import MathOutcomeVerifier

_OUTCOME = MathOutcomeVerifier()


def _replay_pass_at_1(problems: list, records: dict) -> int:
    policy = CassettePolicy(records)
    return sum(
        _OUTCOME.verify(p, policy.sample_full(p, n=1, temperature=0.0, max_tokens=1)[0]).correct
        for p in problems
    )


def test_record_then_replay_reproduces_results(tmp_path: Path) -> None:
    cassette = tmp_path / "sample.json"
    # Record a run (mock backend stands in for a real one), then replay it offline.
    cfg = RunConfig(
        method="pass1",
        dataset="sample",
        record=str(cassette),
        policy=PolicyConfig(backend="mock", model="scripted"),
    )
    recorded = run(cfg)
    assert cassette.exists()

    problems, records = load_cassette(cassette)
    assert len(problems) == recorded.total
    # The offline replay reproduces the recorded run's pass@1 exactly — no model needed.
    assert _replay_pass_at_1(problems, records) == recorded.correct == 4


# --- Real captured fixture (H3): a live GSM8K run, replayed with no GPU/network. -----

_FIXTURE = Path(__file__).parent / "fixtures" / "gsm8k-m1.json"


def test_real_gsm8k_fixture_reproduces_pass_at_1() -> None:
    if not _FIXTURE.exists():  # pragma: no cover - present once a real run is recorded
        import pytest

        pytest.skip("no recorded GSM8K fixture yet — run `crucible run ... --record`")
    problems, records = load_cassette(_FIXTURE)
    assert len(problems) == 3
    # Reproduces the real Ollama run's numbers offline, no GPU/network (see PROGRESS):
    # qwen2.5:7b-instruct, greedy, 3/3 on the first 3 GSM8K test problems.
    assert _replay_pass_at_1(problems, records) == 3


# --- The real GSM8K lift curve (§0 of RESULTS.md), reproduced offline from its cassette.
_CURVE_FIXTURE = Path(__file__).parent / "fixtures" / "gsm8k-bestofn.json"


def test_real_gsm8k_lift_curve_reproduces_offline() -> None:
    if not _CURVE_FIXTURE.exists():  # pragma: no cover
        import pytest

        pytest.skip("no best-of-N cassette recorded yet")
    from crucible.bench import curve_cells, load_samples

    records = load_samples(_CURVE_FIXTURE)
    assert len(records) == 20
    cells = curve_cells(records, [1, 2, 4, 8], has_prm=True)

    def hits(method: str, selection: str, n: int) -> int:
        cell = next(
            c for c in cells if c["method"] == method and c["selection"] == selection and c["n"] == n
        )
        assert cell["total"] == 20
        return int(cell["correct"])

    # The real numbers behind docs/gsm8k-lift-curve.png (Ollama 1.5B + Skywork 1.5B PRM):
    assert hits("pass1", "none", 1) == 8  # pass@1 = 40%
    assert hits("best_of_n", "oracle", 8) == 18  # oracle@8 = 90% — search doubles accuracy
    assert hits("best_of_n", "prm", 8) == 12  # PRM@8 = 60%
    assert hits("best_of_n", "majority", 8) == 10  # majority@8 = 50%
    # The headline: the learned PRM beats verifier-free majority.
    assert hits("best_of_n", "prm", 8) > hits("best_of_n", "majority", 8)


# --- Step + PRM cassettes (the H3 remainder): beam/MCTS runs replay offline. ---------


def _search_config(method: str) -> RunConfig:
    return RunConfig(
        method=method,
        dataset="sample",
        prm="step",
        step_depth=3,
        beam_width=2,
        beam_expansions=2,
        max_steps=6,
        budget_tokens=400,
        mcts_max_sims=20,
        policy=PolicyConfig(backend="stepwise", model="sim"),
    )


def _replay_search(
    cfg: RunConfig, cassette: Path, prm_cassette: Path, strategy: BeamStrategy | MCTSStrategy
) -> dict[str, tuple[bool, str]]:
    bundle = load_bundle(cassette)
    policy = CassettePolicy(bundle.traces, bundle.steps)
    process = CassetteProcessVerifier(load_prm_cassette(prm_cassette))
    out: dict[str, tuple[bool, str]] = {}
    for problem in bundle.problems:
        chosen = strategy.search(problem, policy, _OUTCOME, process, cfg)
        out[problem.id] = (_OUTCOME.verify(problem, chosen).correct, chosen.text)
    return out


@pytest.mark.parametrize(
    ("method", "strategy_cls"), [("beam", BeamStrategy), ("mcts", MCTSStrategy)]
)
def test_search_run_records_and_replays_offline(
    tmp_path: Path, method: str, strategy_cls: type[BeamStrategy] | type[MCTSStrategy]
) -> None:
    cassette = tmp_path / f"{method}.json"
    prm_cassette = tmp_path / f"{method}-prm.json"
    cfg = _search_config(method)
    cfg.record = str(cassette)
    cfg.record_prm = str(prm_cassette)

    live = run(cfg)
    assert cassette.exists() and prm_cassette.exists()

    # The replay walks the same search over recorded steps + recorded PRM scores —
    # no backend, no live PRM — and must land on the same chosen traces.
    replayed = _replay_search(cfg, cassette, prm_cassette, strategy_cls())
    assert len(replayed) == live.total
    for result in live.results:
        correct, text = replayed[result.problem_id]
        assert correct == result.correct
    assert sum(1 for correct, _ in replayed.values() if correct) == live.correct


def test_prm_cassette_raises_on_unrecorded_prefix(tmp_path: Path) -> None:
    verifier = CassetteProcessVerifier({})
    with pytest.raises(KeyError, match="diverged"):
        verifier.score_steps(
            Problem(id="p", prompt="?", answer="1"), [Step(text="s", token_count=1)]
        )
