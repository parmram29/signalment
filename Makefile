.PHONY: install test lint eval

install:
	python3 -m pip install -e ".[dev]"

test:
	PYTHONPATH=src python3 -m pytest -q

lint:
	PYTHONPATH=src python3 -m vetrag lint

eval:
	PYTHONPATH=src python3 -m vetrag eval --out runs/latest.json
