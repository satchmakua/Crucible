# docs/

Deeper and longer-form documentation that doesn't belong in the top-level files.

The four root docs (`README`, `DESIGN`, `ROADMAP`, `PROGRESS`) stay lean and
authoritative. Everything that would bloat them goes here:

- **[`RESULTS.md`](RESULTS.md)** — **the headline deliverable**: the real
  accuracy-vs-compute curves, interpreted honestly (§0 is the real 3-seed MATH-500 run;
  §1–§5 are the cold simulator validation; §6 the threats to validity). The figures beside
  it are its artifacts: `math500-lift-curve.png` (+ `.json` cells), `math500-hard-search.png`
  (real beam/MCTS), and `gsm8k-lift-curve.png` (the earlier single-seed capture).
- **`adr/`** — Architecture Decision Records: one short file per significant,
  hard-to-reverse decision, capturing the context, the choice, and the
  consequences. See [`adr/0001-record-architecture-decisions.md`](adr/0001-record-architecture-decisions.md).
- Long-form design notes, research, API references, runbooks, diagrams, etc.

The authoritative design lives in [`../DESIGN.md`](../DESIGN.md); ADRs here record
the *why* behind individual choices over time.
