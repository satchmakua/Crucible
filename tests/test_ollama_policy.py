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
