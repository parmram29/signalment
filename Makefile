.PHONY: install test lint eval sweep stale drift check

install:
	python3 -m pip install -e ".[dev]"

test:
	PYTHONPATH=src python3 -m pytest -q

lint:
	PYTHONPATH=src python3 -m vetrag lint

eval:
	PYTHONPATH=src python3 -m vetrag eval --out runs/latest.json

sweep:
	PYTHONPATH=src python3 -m vetrag sweep

stale:
	PYTHONPATH=src python3 -m vetrag stale --days 180

drift:
	PYTHONPATH=src python3 -m vetrag drift

# Everything that must pass before a commit.
check: lint test drift
	@echo "ok"
