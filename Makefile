# Crucible — the short path in. `make demo` is the cold-clone entry point (DoD §5):
# no GPU, no network, no model — it replays the REAL captured runs from committed
# cassettes and regenerates the headline accuracy-vs-compute curve.
#
# Works with GNU make on Linux/macOS and on Windows (Git Bash / WSL / `make` from
# choco or scoop). PY is overridable: `make demo PY=.venv/Scripts/python.exe`.

PY ?= python
FIXTURES := tests/fixtures/math500-bestofn-seed0.json \
            tests/fixtures/math500-bestofn-seed1.json \
            tests/fixtures/math500-bestofn-seed2.json

.DEFAULT_GOAL := help
.PHONY: help install demo curve check lint typecheck test clean

help: ## Show this help
	@echo "Crucible — make targets:"
	@echo "  make install    Install the package + dev tooling (editable)"
	@echo "  make demo       THE DEMO: replay the real captured runs, no GPU (DoD §5)"
	@echo "  make curve      Regenerate the real 3-seed MATH-500 lift curve from cassettes"
	@echo "  make check      lint + typecheck + test (what CI runs)"
	@echo "  make test       pytest"
	@echo "  make clean      Remove generated run output + caches"

install: ## Install the package and dev tooling
	$(PY) -m pip install -e ".[dev]"

demo: ## The cold-clone demo: real captured results, replayed offline (no GPU/network)
	@echo "== 1/3  Offline spine: pass@1 on the bundled sample set (mock policy) =="
	$(PY) -m crucible run --method pass1 --dataset sample --policy mock --no-save
	@echo
	@echo "== 2/3  The REAL headline: 3-seed MATH-500 lift curve, replayed from cassettes =="
	$(PY) -m crucible bench curve $(FIXTURES) --out-dir runs/demo-curve
	@echo
	@echo "== 3/3  Every real number above is regression-tested offline =="
	$(PY) -m pytest tests/test_cassette.py -q
	@echo
	@echo "Done. Curve: runs/demo-curve/curve.png  |  Full write-up: docs/RESULTS.md"

curve: ## Regenerate the real 3-seed MATH-500 curve from the committed cassettes
	$(PY) -m crucible bench curve $(FIXTURES) --out-dir runs/demo-curve

check: lint typecheck test ## Everything CI runs

lint: ## ruff
	$(PY) -m ruff check .

typecheck: ## mypy (strict)
	$(PY) -m mypy src

test: ## pytest
	$(PY) -m pytest

clean: ## Remove generated output and caches
	rm -rf runs/demo-curve .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
