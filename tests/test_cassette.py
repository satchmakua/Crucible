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


# --- The real 3-seed MATH-500 lift curve (§0), reproduced offline from its cassettes.
_MATH500_FIXTURES = [
    Path(__file__).parent / "fixtures" / f"math500-bestofn-seed{s}.json" for s in (0, 1, 2)
]


_7B_FIXTURE = Path(__file__).parent / "fixtures" / "math500-7b-pass1.json"


def test_real_7b_baseline_reproduces_offline() -> None:
    if not _7B_FIXTURE.exists():  # pragma: no cover
        import pytest

        pytest.skip("no 7B baseline cassette recorded yet")
    problems, records = load_cassette(_7B_FIXTURE)
    assert len(problems) == 40
    # H2 baseline: qwen2.5:7b-instruct pass@1 = 27/40 = 67.5% on MATH-500 (problems
    # 0-39) at ~524 tokens/problem. This BEATS the 1.5B + search at matched compute —
    # small does not beat big on this stack (the honest negative; see docs/RESULTS.md §0).
    correct = _replay_pass_at_1(problems, records)
    assert correct == 27


def test_real_math500_lift_curve_reproduces_offline() -> None:
    if not all(f.exists() for f in _MATH500_FIXTURES):  # pragma: no cover
        import pytest

        pytest.skip("no MATH-500 bench cassettes recorded yet")
    from crucible.bench import curve_cells, load_samples

    records = [r for f in _MATH500_FIXTURES for r in load_samples(f)]
    assert len(records) == 120  # 40 problems x 3 seeds, pooled

    cells = curve_cells(records, [1, 2, 4, 8], has_prm=True)

    def hits(method: str, selection: str, n: int) -> int:
        cell = next(
            c for c in cells if c["method"] == method and c["selection"] == selection and c["n"] == n
        )
        assert cell["total"] == 120
        return int(cell["correct"])

    # The real numbers behind docs/math500-lift-curve.png (Ollama 1.5B + Skywork 1.5B PRM,
    # 3 seeds, MATH-500 problems 0-39). pass@1 46/120 = 38.3%; oracle@8 84/120 = 70%.
    assert hits("pass1", "none", 1) == 46
    assert hits("best_of_n", "oracle", 8) == 84  # search nearly doubles pass@1
    assert hits("best_of_n", "prm", 8) == 66  # PRM@8 = 55%
    assert hits("best_of_n", "majority", 8) == 63  # majority@8 = 52.5%
    # The headline, on identical samples: the learned PRM beats verifier-free majority
    # at every N>=2 (the effect a small/easy pilot hides; here on 3-seed MATH-500).
    for n in (2, 4, 8):
        assert hits("best_of_n", "prm", n) > hits("best_of_n", "majority", n), n


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


# --- The real beam/MCTS cells (§0), a live PRM-guided search reproduced with no GPU. ---
_FIXDIR = Path(__file__).parent / "fixtures"


def _replay_real_search(
    strategy: BeamStrategy | MCTSStrategy, cfg: RunConfig, name: str
) -> tuple[int, int, int]:
    steps = _FIXDIR / f"math500-{name}-hard-steps.json"
    prm = _FIXDIR / f"math500-{name}-hard-prm.json"
    if not (steps.exists() and prm.exists()):  # pragma: no cover
        import pytest

        pytest.skip(f"no real {name} cassette recorded yet")
    bundle = load_bundle(steps)
    policy = CassettePolicy(bundle.traces, bundle.steps)
    process = CassetteProcessVerifier(load_prm_cassette(prm))
    correct = tokens = 0
    for problem in bundle.problems:
        chosen = strategy.search(problem, policy, _OUTCOME, process, cfg)
        correct += _OUTCOME.verify(problem, chosen).correct
        tokens += chosen.compute.total_tokens
    return len(bundle.problems), correct, tokens


def test_real_beam_cell_reproduces_offline() -> None:
    cfg = RunConfig(
        method="beam", dataset="math500-hard", beam_width=2, beam_expansions=2,
        max_steps=5, prm="cassette", prm_aggregate="mean",
        policy=PolicyConfig(backend="cassette", model="x", max_tokens=512),
    )
    total, correct, tokens = _replay_real_search(BeamStrategy(), cfg, "beam")
    # The honest real-model beam cell: on the 8 hardest MATH-500 problems (levels 4-5,
    # where the 1.5B policy's pass@1 is 0%), PRM-guided beam solves none at ~37k
    # tokens/problem — stepwise search can't help a non-reasoning policy that restarts
    # rather than continues a partial trace. Reproduced from the recorded run, no GPU.
    assert total == 8
    assert correct == 0
    assert tokens == 294026


def test_real_mcts_cell_reproduces_offline() -> None:
    cfg = RunConfig(
        method="mcts", dataset="math500-hard", beam_expansions=2, max_steps=5,
        budget_tokens=10000, mcts_max_sims=30, mcts_c_puct=1.4, prm="cassette",
        prm_aggregate="mean", policy=PolicyConfig(backend="cassette", model="x", max_tokens=512),
    )
    total, correct, tokens = _replay_real_search(MCTSStrategy(), cfg, "mcts")
    # Budget-capped MCTS on the same 8 hardest problems: 1/8 at ~12k tokens/problem —
    # cheaper than beam (the budget bites) and it solves one, but still below best-of-N.
    assert total == 8
    assert correct == 1
    assert tokens == 94556
