"""OllamaPolicy — the default real backend (native Windows).

Talks to a running Ollama server's HTTP API. Token counts come straight from Ollama's
`eval_count`, so compute accounting stays honest. The prompt is built by
`prompts.build_cot_prompt` (overridable) rather than sending the raw problem, so the
model is actually asked to reason and to emit a ``\boxed{}`` answer.

An `httpx.Client` can be injected for testing against a mock transport; otherwise a
short-lived client is opened per request.
"""

from __future__ import annotations

import hashlib
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
        seed: int | None = None,
    ) -> None:
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST).rstrip("/")
        self._max_step_tokens = max_step_tokens
        self._timeout = timeout
        self._prompt_builder = prompt_builder
        self._client = client
        self._seed = seed

    def _request_seed(self, *parts: object) -> int | None:
        """A stable per-request Ollama seed derived from the run seed + call identity.

        Every sample index (and every step prefix) gets its own seed, so best-of-N
        stays diverse while a whole run is reproducible for a fixed run seed — which
        is what makes "3 seeds" a real, re-runnable experiment rather than three
        arbitrary reruns. None (no run seed) leaves Ollama unseeded.
        """
        if self._seed is None:
            return None
        key = ":".join(str(p) for p in (self._seed, *parts))
        return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:4], "big") & 0x7FFFFFFF

    def _post(self, payload: dict[str, object]) -> dict[str, object]:
        if self._client is not None:
            resp = self._client.post("/api/generate", json=payload)
        else:
            with httpx.Client(base_url=self.host, timeout=self._timeout) as client:
                resp = client.post("/api/generate", json=payload)
        resp.raise_for_status()
        data: dict[str, object] = resp.json()
        return data

    def _generate(
        self, prompt: str, *, temperature: float, max_tokens: int, seed: int | None = None
    ) -> tuple[str, int]:
        options: dict[str, object] = {"temperature": temperature, "num_predict": max_tokens}
        if seed is not None:
            options["seed"] = seed
        payload: dict[str, object] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": options,
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
        for i in range(n):
            t0 = time.perf_counter()
            text, tokens = self._generate(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=self._request_seed(problem.id, i),
            )
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
        for i in range(n):
            text, tokens = self._generate(
                prompt,
                temperature=temperature,
                max_tokens=min(max_tokens, self._max_step_tokens),
                seed=self._request_seed("step", problem.id, prefix_text, i),
            )
            # Keep the first segment as the step (what beam/MCTS branch on), but charge
            # the FULL generation's real eval_count to it: the whole continuation was
            # paid on the GPU even though only the first segment is retained. Using
            # segment()'s word count instead would land beam/MCTS on a deflated token
            # axis vs the eval_count-based best-of-N cassette cells — a matched-compute
            # violation (DESIGN §2). eval_count is the same real-tokenizer unit as
            # sample_full, so search and best-of-N compare honestly.
            first = segment(text, max_step_tokens=self._max_step_tokens)
            step_text = first[0].text if first else text
            steps.append(Step(text=step_text, token_count=tokens))
        return steps
