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
    """A real open PRM via `transformers` (e.g. Qwen2.5-Math-PRM-7B).

    Targets the Qwen-PRM scoring convention: steps are joined with a separator token and
    the model emits a per-step "good" probability. The exact tensor wiring is
    model-specific — **verify against the model card on the first real run**; the mock
    PRM carries the unit-tested logic. Needs the `prm` extra (torch + transformers).
    """

    name = "prm"

    def __init__(
        self,
        model_id: str,
        *,
        device: str | None = None,
        step_separator: str = "\n\n",
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.step_separator = step_separator
        self._model: Any = None
        self._tokenizer: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise NotImplementedError(
                'the PRM backend needs the `prm` extra: pip install -e ".[prm]"'
            ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(
            self.model_id, trust_remote_code=True, torch_dtype="auto"
        ).eval()
        if self.device:
            self._model = self._model.to(self.device)

    def score_steps(self, problem: Problem, prefix: list[Step]) -> list[float]:  # pragma: no cover
        self._ensure_loaded()
        import torch

        steps_text = [s.text for s in prefix] or [""]
        prompt = f"{problem.prompt}\n\n" + self.step_separator.join(steps_text)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        if self.device:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        # Reduce the model's per-token reward signal to one score per step. Different
        # PRMs expose this differently; mean-pool the last hidden/logit as a sane default.
        logits = getattr(outputs, "logits", None)
        if logits is None:
            logits = outputs[0]
        reward = torch.sigmoid(logits.float().mean()).item()
        return [float(reward) for _ in steps_text]
