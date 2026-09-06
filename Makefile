.PHONY: install test lint eval sweep stale drift check accel rust-test bench

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

# Build the optional Rust accelerator into the active environment.
accel:
	python3 -m pip install maturin
	maturin develop -m crates/vetrag_bm25/Cargo.toml --release

# Core scoring logic only: needs no libpython, no headers, no interpreter.
# EXPECTED TO FAIL until crates/vetrag_bm25/src/bm25.rs is implemented — the
# 17 tests there are the spec, and going red-to-green is the exercise.
rust-test:
	cargo test --manifest-path crates/vetrag_bm25/Cargo.toml --no-default-features

bench:
	PYTHONPATH=src python3 benchmarks/bench_retrieval.py --scale 200

# Everything that must pass before a commit.
check: lint test drift
	@echo "ok"
