# BITCOIN4Traders – Developer Makefile
# Usage: make <target>
# Requires: Python 3.11+, pip

.PHONY: help install install-dev test test-fast lint typecheck format clean train colab

PYTHON     := python3
PIP        := pip3
SRC        := src
TESTS      := tests
PYTHONPATH := $(PWD)/src

help:
	@echo ""
	@echo "BITCOIN4Traders – available commands:"
	@echo ""
	@echo "  Setup:"
	@echo "    make install        Install runtime dependencies"
	@echo "    make install-dev    Install runtime + dev dependencies"
	@echo "    make setup-env      Copy .env.example to .env (first time)"
	@echo ""
	@echo "  Testing:"
	@echo "    make test           Run all tests with coverage report"
	@echo "    make test-fast      Run tests without coverage (faster)"
	@echo "    make test-file f=tests/test_math_models.py  Run single file"
	@echo ""
	@echo "  Code Quality:"
	@echo "    make lint           Check code style with ruff"
	@echo "    make typecheck      Type check with mypy"
	@echo "    make format         Auto-format code with ruff"
	@echo ""
	@echo "  Training:"
	@echo "    make train          Start local training (CPU)"
	@echo "    make train-gpu      Start local training (GPU)"
	@echo "    make train-8h       Run 8-hour automated training"
	@echo "    make train-12h      Run 12-hour automated training"
	@echo ""
	@echo "  Data:"
	@echo "    make download       Download historical BTC data"
	@echo ""
	@echo "  Utils:"
	@echo "    make clean          Remove cache and temp files"
	@echo "    make colab          Print Colab URL"
	@echo ""

# ── Setup ────────────────────────────────────────────────────────────

install:
	$(PIP) install -r requirements.txt

install-dev: install
	$(PIP) install -r requirements-dev.txt

setup-env:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo ".env created from .env.example – fill in your credentials!"; \
	else \
		echo ".env already exists, skipping."; \
	fi

# ── Testing ──────────────────────────────────────────────────────────

test:
	PYTHONPATH=$(PYTHONPATH) pytest $(TESTS)/ \
		--cov=$(SRC) \
		--cov-report=term-missing \
		--timeout=120 \
		-v

test-fast:
	PYTHONPATH=$(PYTHONPATH) pytest $(TESTS)/ --timeout=60 -q

test-file:
	PYTHONPATH=$(PYTHONPATH) pytest $(f) -v --timeout=120

# ── Code Quality ─────────────────────────────────────────────────────

lint:
	ruff check $(SRC)/

typecheck:
	mypy $(SRC)/ --ignore-missing-imports

format:
	ruff check $(SRC)/ --fix
	ruff format $(SRC)/

# ── Training ─────────────────────────────────────────────────────────

train:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) train.py --device cpu

train-gpu:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) train.py --device cuda

train-8h:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) auto_train.py

train-12h:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) auto_12h_train.py

# ── Data ─────────────────────────────────────────────────────────────

download:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) download_historical_data.py

# ── Utils ────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -f coverage.xml .coverage
	@echo "Clean done."

colab:
	@echo ""
	@echo "Open in Google Colab:"
	@echo "https://colab.research.google.com/github/juancarlosrial76-code/BITCOIN4Traders/blob/main/BITCOIN4Traders_Colab.ipynb"
	@echo ""
