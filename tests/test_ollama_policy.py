"""OllamaPolicy against a mocked HTTP transport — the real adapter, no live server."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx

from crucible.domain.types import Problem
from crucible.inference.ollama import OllamaPolicy
from crucible.verify import MathOutcomeVerifier


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))


def test_builds_payload_and_parses_response() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"response": "Reason.\n\nThe answer is \\boxed{42}.", "eval_count": 9}
        )

    policy = OllamaPolicy("test-model", client=_client(handler))
    problem = Problem(id="x", prompt="What is 6 times 7?", answer="42")
    traces = policy.sample_full(problem, n=2, temperature=0.5, max_tokens=128)

    assert len(traces) == 2
    assert captured["path"] == "/api/generate"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["model"] == "test-model"
    assert "\\boxed{}" in body["prompt"]
    assert "What is 6 times 7?" in body["prompt"]
    assert body["options"]["temperature"] == 0.5
    assert body["options"]["num_predict"] == 128

    t = traces[0]
    assert t.compute.policy_gen_tokens == 9
    assert t.compute.policy_forward_calls == 1
    assert "\\boxed{42}" in t.text


def test_generated_trace_verifies_against_gold() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"response": "6*7 = 42, so \\boxed{42}", "eval_count": 6})

    policy = OllamaPolicy("m", client=_client(handler))
    problem = Problem(id="x", prompt="6*7?", answer="42")
    trace = policy.sample_full(problem, n=1, temperature=0.0, max_tokens=64)[0]
    assert MathOutcomeVerifier().verify(problem, trace).correct


def test_token_count_falls_back_without_eval_count() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"response": "one two three"})

    policy = OllamaPolicy("m", client=_client(handler))
    trace = policy.sample_full(
        Problem(id="x", prompt="q", answer="1"), n=1, temperature=0.0, max_tokens=8
    )[0]
    assert trace.compute.policy_gen_tokens == 3  # approx_tokens fallback


def _seed_capturing_handler(seeds: list[object]) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seeds.append(body["options"].get("seed"))
        return httpx.Response(200, json={"response": "\\boxed{1}", "eval_count": 2})

    return handler


def test_run_seed_derives_distinct_reproducible_request_seeds() -> None:
    problem = Problem(id="p1", prompt="q", answer="1")

    first: list[object] = []
    OllamaPolicy("m", client=_client(_seed_capturing_handler(first)), seed=0).sample_full(
        problem, n=3, temperature=0.7, max_tokens=8
    )
    # Distinct per sample (best-of-N stays diverse) but valid Ollama seeds.
    assert len(set(first)) == 3
    assert all(isinstance(s, int) and s >= 0 for s in first)

    again: list[object] = []
    OllamaPolicy("m", client=_client(_seed_capturing_handler(again)), seed=0).sample_full(
        problem, n=3, temperature=0.7, max_tokens=8
    )
    assert again == first  # same run seed → the same requests

    other: list[object] = []
    OllamaPolicy("m", client=_client(_seed_capturing_handler(other)), seed=1).sample_full(
        problem, n=3, temperature=0.7, max_tokens=8
    )
    assert set(other).isdisjoint(first)  # a different seed is a different experiment


def test_no_run_seed_leaves_requests_unseeded() -> None:
    seeds: list[object] = []
    OllamaPolicy("m", client=_client(_seed_capturing_handler(seeds))).sample_full(
        Problem(id="p1", prompt="q", answer="1"), n=2, temperature=0.7, max_tokens=8
    )
    assert seeds == [None, None]


def test_step_request_seeds_vary_with_prefix_and_index() -> None:
    from crucible.domain.types import Step

    seeds: list[object] = []
    policy = OllamaPolicy("m", client=_client(_seed_capturing_handler(seeds)), seed=0)
    problem = Problem(id="p1", prompt="q", answer="1")
    policy.sample_step(problem, [], n=2, temperature=0.7, max_tokens=64)
    policy.sample_step(
        problem, [Step(text="partial", token_count=1)], n=2, temperature=0.7, max_tokens=64
    )
    assert len(set(seeds)) == 4  # every (prefix, index) pair gets its own seed
