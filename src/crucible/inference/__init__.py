"""Inference adapters: one `PolicyModel` port, several backends.

- `ScriptedPolicy` (mock) — deterministic replay of canned traces; no GPU, no
  network. The M0 spine and the test suite run on this.
- `SyntheticPolicy` (synthetic) — a seeded simulator of a known accuracy; drives the
  cold best-of-N lift-curve demo (M2).
- `OllamaPolicy` — the default real backend (native Windows), used from M1.

vLLM and a hosted OpenAI-compatible adapter slot in behind the same port later.
"""

from __future__ import annotations

from crucible.inference.cassette import (
    CassetteBundle,
    CassettePolicy,
    CassetteProcessVerifier,
    RecordingPolicy,
    RecordingProcessVerifier,
    load_bundle,
    load_cassette,
    load_prm_cassette,
    step_key,
)
from crucible.inference.mock import ScriptedPolicy
from crucible.inference.ollama import OllamaPolicy
from crucible.inference.synthetic import SyntheticPolicy

__all__ = [
    "CassetteBundle",
    "CassettePolicy",
    "CassetteProcessVerifier",
    "OllamaPolicy",
    "RecordingPolicy",
    "RecordingProcessVerifier",
    "ScriptedPolicy",
    "SyntheticPolicy",
    "load_bundle",
    "load_cassette",
    "load_prm_cassette",
    "step_key",
]
