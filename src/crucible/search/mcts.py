"""mcts — Monte Carlo Tree Search over the reasoning-step tree (the headline method).

PUCT over partial traces, with the PRM as the value function (rStar-Math's recipe,
minus the training loop). Each simulation:

1. **Select** — descend from the root by maximizing
   ``Q(child) + c_puct · P(child) · sqrt(N(parent)) / (1 + N(child))`` until a leaf.
2. **Expand** — sample `beam_expansions` continuations of the leaf via `sample_step`.
3. **Evaluate** — value the new node by its PRM score (no gold answer needed).
4. **Backup** — propagate the value up the path, incrementing visit counts.

The most expressive search and the most expensive: it allocates expansions adaptively to
promising branches. The budget is `budget_tokens` (total policy + verifier tokens), so
MCTS is comparable to beam/best-of-N on the same axis. Every step sampled and every PRM
forward pass is counted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from crucible.config import RunConfig
from crucible.domain.ports import OutcomeVerifier, PolicyModel, ProcessVerifier
from crucible.domain.types import Compute, Problem, Step, Trace
from crucible.verify import aggregate_scores, has_explicit_answer

_DEFAULT_BUDGET_TOKENS = 4000


@dataclass
class _Node:
    steps: list[Step]
    parent: _Node | None
    terminal: bool
    children: list[_Node] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    @property
    def expanded(self) -> bool:
        return bool(self.children)


def _terminal(steps: list[Step]) -> bool:
    return has_explicit_answer("\n\n".join(s.text for s in steps))


class MCTSStrategy:
    """UCT/PUCT search over reasoning steps, valued by the PRM."""

    name = "mcts"

    def search(
        self,
        problem: Problem,
        policy: PolicyModel,
        outcome: OutcomeVerifier,
        process: ProcessVerifier | None,
        config: RunConfig,
    ) -> Trace:
        if process is None:
            raise ValueError("mcts needs a process verifier — pass --prm.")
        c_puct = config.mcts_c_puct
        m = max(1, config.beam_expansions)
        budget = config.budget_tokens or _DEFAULT_BUDGET_TOKENS

        compute = Compute()
        root = _Node(steps=[], parent=None, terminal=False)
        best: _Node | None = None
        best_value = float("-inf")
        sims = 0

        def evaluate(node: _Node) -> float:
            nonlocal compute, best, best_value
            scores = process.score_steps(problem, node.steps)
            compute = compute + Compute(
                verifier_forward_calls=1,
                verifier_gen_tokens=sum(s.token_count for s in node.steps),
            )
            value = aggregate_scores(scores, config.prm_aggregate)
            if node.terminal and value > best_value:
                best_value, best = value, node
            return value

        while compute.total_tokens < budget and sims < config.mcts_max_sims:
            sims += 1

            # 1. Select a leaf by PUCT.
            node = root
            path = [root]
            while node.expanded and not node.terminal:
                node = self._best_child(node, c_puct)
                path.append(node)

            # 2. Expand the leaf, evaluating every child to seed its value; the node's
            #    backed-up value is its best child (terminal/depth-capped leaves are
            #    valued directly).
            if not node.terminal and len(node.steps) < config.max_steps:
                next_steps = policy.sample_step(
                    problem,
                    node.steps,
                    n=m,
                    temperature=config.policy.temperature,
                    max_tokens=config.policy.max_tokens,
                )
                compute = compute + Compute(
                    policy_gen_tokens=sum(s.token_count for s in next_steps),
                    policy_forward_calls=len(next_steps),
                )
                for step in next_steps:
                    child_steps = [*node.steps, step]
                    child = _Node(steps=child_steps, parent=node, terminal=_terminal(child_steps))
                    child.value_sum = evaluate(child)
                    child.visits = 1
                    node.children.append(child)
                # NB: `max(..., default=evaluate(node))` would call evaluate() eagerly
                # every time (Python evaluates kwargs before max runs), burning — and
                # counting — a wasted PRM forward pass. Only evaluate when there are no
                # children.
                value = max(c.q for c in node.children) if node.children else evaluate(node)
            else:
                value = evaluate(node)

            # 3. Backup along the selected path.
            for visited in path:
                visited.visits += 1
                visited.value_sum += value

        chosen = best if best is not None else self._most_visited_leaf(root)
        return Trace(steps=chosen.steps, final_answer=None, compute=compute)

    @staticmethod
    def _best_child(node: _Node, c_puct: float) -> _Node:
        prior = 1.0 / len(node.children)  # uniform prior (no policy value head)
        sqrt_parent = math.sqrt(node.visits)

        def puct(child: _Node) -> float:
            return child.q + c_puct * prior * sqrt_parent / (1 + child.visits)

        return max(node.children, key=puct)

    @staticmethod
    def _most_visited_leaf(root: _Node) -> _Node:
        node = root
        while node.children:
            node = max(node.children, key=lambda c: c.visits)
        return node
