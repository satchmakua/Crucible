# Crucible

> A verifier-guided reasoning engine: generate many reasoning traces from a small
> open model, score them with a verifier (programmatic checker or a process-reward
> model), and **search** — best-of-N → beam → MCTS over reasoning steps — to turn
> test-time compute into measurable accuracy on math and code.

Most people only *consume* reasoning models; Crucible builds the machinery underneath
and **measures the lift** — accuracy as a function of test-time compute over a small
open policy model.

**Status:** Design draft. The full spec lives in [DESIGN.md](DESIGN.md).
**Next step:** run **`/scaffold`** to turn this into a running project.
