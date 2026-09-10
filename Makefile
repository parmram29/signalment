.PHONY: install test lint eval sweep stale drift check accel rust-test bench

install:
	python3 -m pip install -e ".[dev]"

test:
	PYTHONPATH=src python3 -m pytest -q

lint:
	PYTHONPATH=src python3 -m signalment lint

# Python style, the counterpart to cargo fmt/clippy on the Rust side.
ruff:
	python3 -m ruff check src tests benchmarks
	python3 -m ruff format --check src tests benchmarks

eval:
	PYTHONPATH=src python3 -m signalment eval --out runs/latest.json

# Adds dense and BM25+dense fusion rows. Downloads model weights on first run.
eval-dense:
	PYTHONPATH=src python3 -m signalment eval --dense --out runs/dense.json

sweep:
	PYTHONPATH=src python3 -m signalment sweep

stale:
	PYTHONPATH=src python3 -m signalment stale --days 180

drift:
	PYTHONPATH=src python3 -m signalment drift

# Build the optional Rust accelerator into the active environment.
# `maturin develop` needs an active virtualenv; `source .venv/bin/activate`
# first. CI builds a wheel instead, for a runner that has no venv.
accel:
	python3 -m pip install maturin
	maturin develop -m crates/signalment_bm25/Cargo.toml --release

# Core scoring logic only: needs no libpython, no headers, no interpreter.
# EXPECTED TO FAIL until crates/signalment_bm25/src/bm25.rs is implemented — the
# 17 tests there are the spec, and going red-to-green is the exercise.
rust-test:
	cargo test --manifest-path crates/signalment_bm25/Cargo.toml --no-default-features

bench:
	PYTHONPATH=src python3 benchmarks/bench_retrieval.py --scale 200

# Everything that must pass before a commit.
check: lint ruff test drift
	@echo "ok"
