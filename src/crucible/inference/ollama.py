"""OllamaPolicy — the default real backend (native Windows).

Talks to a running Ollama server's HTTP API. Token counts come straight from Ollama's
`eval_count`, so compute accounting stays honest. The prompt is built by
`prompts.build_cot_prompt` (overridable) rather than sending the raw problem, so the
model is actually asked to reason and to emit a ``\boxed{}`` answer.

An `httpx.Client` can be injected for testing against a mock transport; otherwise a
short-lived client is opened per request.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable

import httpx

from crucible.domain.types import Compute, Problem, Step, Trace
from crucible.prompts import build_cot_prompt
from crucible.segment import approx_tokens, segment

DEFAULT_HOST = "http://localhost:11434"


class OllamaPolicy:
    """`PolicyModel` backed by an Ollama server (`/api/generate`)."""

    name = "ollama"

    def __init__(
        self,
        model: str,
        *,
        host: str | None = None,
        max_step_tokens: int = 512,
        timeout: float = 120.0,
        prompt_builder: Callable[[Problem], str] = build_cot_prompt,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST).rstrip("/")
        self._max_step_tokens = max_step_tokens
        self._timeout = timeout
        self._prompt_builder = prompt_builder
        self._client = client

    def _post(self, payload: dict[str, object]) -> dict[str, object]:
        if self._client is not None:
            resp = self._client.post("/api/generate", json=payload)
        else:
            with httpx.Client(base_url=self.host, timeout=self._timeout) as client:
                resp = client.post("/api/generate", json=payload)
        resp.raise_for_status()
        data: dict[str, object] = resp.json()
        return data

    def _generate(self, prompt: str, *, temperature: float, max_tokens: int) -> tuple[str, int]:
        payload: dict[str, object] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        data = self._post(payload)
        text = str(data.get("response", ""))
        eval_count = data.get("eval_count")
        tokens = int(eval_count) if isinstance(eval_count, int) else approx_tokens(text)
        return text, tokens

    def _trace(self, text: str, tokens: int, elapsed: float) -> Trace:
        steps = segment(text, max_step_tokens=self._max_step_tokens)
        compute = Compute(policy_gen_tokens=tokens, policy_forward_calls=1, wall_seconds=elapsed)
        return Trace(steps=steps, final_answer=None, compute=compute)

    def sample_full(
        self, problem: Problem, *, n: int, temperature: float, max_tokens: int
    ) -> list[Trace]:
        prompt = self._prompt_builder(problem)
        traces: list[Trace] = []
        for _ in range(n):
            t0 = time.perf_counter()
            text, tokens = self._generate(prompt, temperature=temperature, max_tokens=max_tokens)
            traces.append(self._trace(text, tokens, time.perf_counter() - t0))
        return traces

    def sample_step(
        self,
        problem: Problem,
        prefix: list[Step],
        *,
        n: int,
        temperature: float,
        max_tokens: int,
    ) -> list[Step]:
        base = self._prompt_builder(problem)
        prefix_text = "\n\n".join(s.text for s in prefix)
        prompt = f"{base}\n\n{prefix_text}".strip()
        steps: list[Step] = []
        for _ in range(n):
            text, tokens = self._generate(
                prompt, temperature=temperature, max_tokens=min(max_tokens, self._max_step_tokens)
            )
            first = segment(text, max_step_tokens=self._max_step_tokens)
            steps.append(first[0] if first else Step(text=text, token_count=tokens))
        return steps
