# signalment_bm25

Accelerated BM25 scoring for the signalment retrieval harness.

**This crate is an accelerator, not a replacement.** `src/signalment/retrieval/bm25.py`
stays in the repository permanently as the reference implementation. It is the
oracle this crate is diffed against, and the path that runs when the extension
is not built.

## Layout

| File | Contains | Contains no |
|---|---|---|
| `src/bm25.rs` | All scoring logic. Pure Rust. | PyO3 |
| `src/python.rs` | PyO3 bindings. | Scoring logic |
| `src/lib.rs` | Module wiring. | Both |

The split is why `cargo test --no-default-features` works with no libpython,
no Python headers and no interpreter present — and why a bug is always in one
file or the other rather than smeared across both.

## Build

```bash
pip install maturin
maturin develop -m crates/signalment_bm25/Cargo.toml --release
```

`--release` matters: a debug build of a scoring loop is slow enough to make the
benchmark meaningless.

## Test

```bash
# Core logic, no Python involved.
cargo test --manifest-path crates/signalment_bm25/Cargo.toml --no-default-features

# Equivalence with the Python reference. Skips if the extension is not built.
pytest tests/test_backend_equivalence.py
```

The Rust unit tests in `bm25.rs` are the local red/green loop.
`tests/test_backend_equivalence.py` is the real specification: identical
rankings, identical chunk attribution, and scores agreeing to 1e-9 against the
Python implementation across ordinary queries and the tokenizer edge cases.

## Benchmark

```bash
PYTHONPATH=src python3 benchmarks/bench_retrieval.py --scale 200
```

Reports build and per-query search time for every available backend. The
comparison is only meaningful at `--release`.
