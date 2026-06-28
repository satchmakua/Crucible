"""Search strategies — the project's actual IP.

All strategies satisfy the `SearchStrategy` port and consume the same policy +
verifier, so they are interchangeable from the runner's point of view. M0 ships
`pass1` (the verifier-free single-sample baseline); best-of-N, beam, and MCTS arrive
in later milestones (ROADMAP M2/M4/M6).
"""

from __future__ import annotations

from crucible.search.best_of_n import BestOfNStrategy
from crucible.search.pass1 import Pass1Strategy
from crucible.search.registry import available_methods, get_strategy

__all__ = ["BestOfNStrategy", "Pass1Strategy", "available_methods", "get_strategy"]
