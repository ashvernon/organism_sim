# Simple runner for organism_sim
# Usage:
#   make run
#   make fmt
#   make lint

PY=python

run:
	$(PY) main.py

# Optional helpers (won't fail your build if not installed)
fmt:
	$(PY) -m ruff format . || true

lint:
	$(PY) -m ruff check . || true
