"""Process verifiers (PRMs): a scalar score per reasoning step.

A PRM guides *search* (best-of-N selection now; beam/MCTS value later) but is **never**
the reported metric — final accuracy is always the outcome verifier on the chosen trace
(DESIGN.md §4.4). Two adapters behind the `ProcessVerifier` port:

- `MockProcessVerifier` — a seeded simulator of an imperfect PRM, so PRM-weighted
  selection and the majority/PRM/oracle gap can be tested cold. Like the synthetic
  policy, it's a demo backend that peeks at the gold answer to fabricate a noisy signal.
- `PRMVerifier` — a real open PRM via `transformers` (the `prm` extra). Lazy-imported;
  exercised from a real run, not the cold test suite.

`aggregate_scores` reduces per-step scores to one number for selection.
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

from crucible.domain.types import Problem, Step
from crucible.verify.answer_extract import extract_final_answer
from crucible.verify.math_outcome import math_equal

if TYPE_CHECKING:
    from typing import Any


def aggregate_scores(scores: list[float], how: str = "mean") -> float:
    """Reduce per-step PRM scores to a single trace-level score."""
    if not scores:
        return 0.0
    if how == "mean":
        return sum(scores) / len(scores)
    if how == "min":
        return min(scores)
    if how == "last":
        return scores[-1]
    if how == "prod":
        return math.prod(scores)
    raise ValueError(f"unknown aggregate '{how}' (mean | min | last | prod).")


class MockProcessVerifier:
    """A seeded, imperfect PRM simulator: correct traces score higher, with noise.

    `accuracy` is the simulated PRM's skill: 1.0 cleanly separates correct from wrong
    (≈ oracle), 0.5 is essentially random. Scores are deterministic given the seed and
    the trace text, so selection is reproducible.
    """

    name = "mock-prm"

    def __init__(self, *, accuracy: float = 0.8, seed: int = 0) -> None:
        self.accuracy = accuracy
        self.seed = seed

    def score_steps(self, problem: Problem, prefix: list[Step]) -> list[float]:
        text = "\n\n".join(s.text for s in prefix)
        pred = extract_final_answer(text)
        correct = (
            pred is not None and problem.answer is not None and math_equal(pred, problem.answer)
        )
        rng = random.Random(f"{self.seed}:{problem.id}:{text}")
        base = 0.5 + (0.4 if correct else -0.4) * self.accuracy
        score = base + rng.uniform(-0.3, 0.3)
        steps = prefix or [Step(text="", token_count=0)]
        return [score for _ in steps]


class PRMVerifier:
    """A real open PRM via `transformers` — the Skywork-o1-Open-PRM (Qwen2-reward) family.

    Convention (validated 2026-07-12 against Skywork-o1-Open-PRM-Qwen-2.5-1.5B on an
    RTX 5070 Ti): the input is ``bos + problem + "\\n"`` followed by each reasoning step,
    every step terminated by a newline whose token position is a *reward position*. The
    model's per-token value head (``v_head``) gives a scalar; ``sigmoid`` maps it to a
    "good so far" probability, read off at each step's newline → one reward per step.

    Needs the `prm` extra (torch + transformers **<5** — the model's custom code targets
    the 4.x cache API). Runs on the GPU; not exercised in CI (the mock PRM is).
    """

    name = "prm"

    def __init__(self, model_id: str, *, device: str | None = None) -> None:
        self.model_id = model_id
        self.device = device
        self._model: Any = None
        self._tokenizer: Any = None
        self._head: Any = None

    def _ensure_loaded(self) -> None:  # pragma: no cover - needs a GPU + the prm extra
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise NotImplementedError(
                'the PRM backend needs the `prm` extra: pip install -e ".[prm]"'
            ) from exc
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            self.model_id, trust_remote_code=True, torch_dtype=dtype
        )
        self._model = model.to(self.device).eval()
        # The per-token value head: prefer the named `v_head`, else the lone Linear→1.
        self._head = getattr(model, "v_head", None)
        if self._head is None:
            for module in model.modules():
                if isinstance(module, torch.nn.Linear) and module.out_features == 1:
                    self._head = module
                    break

    def score_steps(self, problem: Problem, prefix: list[Step]) -> list[float]:  # pragma: no cover
        self._ensure_loaded()
        import torch

        tok = self._tokenizer
        ids: list[int] = list(tok.encode((tok.bos_token or "") + problem.prompt + "\n"))
        flags: list[int] = [0] * len(ids)
        newline = tok.encode("\n")[-1]
        for step in prefix:
            step_ids = tok.encode(step.text) + [newline]
            ids.extend(step_ids)
            flags.extend([0] * (len(step_ids) - 1) + [1])

        reward_positions = [i for i, f in enumerate(flags) if f == 1]
        if not reward_positions:
            return [0.0]

        inp = torch.tensor([ids], device=self.device)
        with torch.no_grad():
            out = self._model(
                input_ids=inp, attention_mask=torch.ones_like(inp), output_hidden_states=True
            )
        per_token = torch.sigmoid(self._head(out.hidden_states[-1]).float().reshape(-1))
        per_token = per_token.detach().cpu()
        return [float(per_token[i]) for i in reward_positions]
