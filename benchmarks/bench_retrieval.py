"""Time each available retrieval backend.

The number this produces is the whole justification for the Rust crate, so it
has to be honest:

* Every backend sees the identical chunk list and the identical queries.
* The corpus is scaled by replication, because a 4-note corpus times nothing
  useful. Replicated chunks get distinct ids so deduplication still has work
  to do and the comparison stays fair.
* Index construction and query time are reported separately. They have
  different shapes: build is once per launch, search is per keystroke.

Run:  PYTHONPATH=src python3 benchmarks/bench_retrieval.py --scale 200
"""

from __future__ import annotations

import argparse
import statistics
import time

from signalment.corpus import Chunk, chunk_corpus, load_corpus
from signalment.retrieval.backends import available_backends, make_retriever
from signalment.retrieval.base import Query

QUERIES = [
    "permethrin sodium channels cats",
    "why are kittens prone to low blood sugar",
    "phase II conjugation glucuronidation deficiency",
    "insulinoma episodic collapse after exercise",
    "hypoglycaemia neonate glycogen thermoregulation",
]


def scale_corpus(chunks: list[Chunk], factor: int) -> list[Chunk]:
    """Replicate the corpus `factor` times with distinct note and chunk ids."""
    if factor < 1:
        raise ValueError("scale must be at least 1")
    return [
        Chunk(
            chunk_id=f"{chunk.chunk_id}-r{i}",
            note_id=f"{chunk.note_id}-r{i}",
            text=chunk.text,
            heading=chunk.heading,
        )
        for i in range(factor)
        for chunk in chunks
    ]


def time_it(fn, repeats: int) -> tuple[float, float]:
    """Return (median, best) seconds over `repeats` runs."""
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples), min(samples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="corpus/notes")
    parser.add_argument("--scale", type=int, default=200,
                        help="replicate the corpus this many times")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    notes, errors = load_corpus(args.corpus)
    if errors:
        print(f"{len(errors)} invalid note(s); run `signalment lint`")
        return 1

    chunks = scale_corpus(chunk_corpus(notes), args.scale)
    backends = available_backends()

    print(f"corpus: {len(notes)} notes x{args.scale} -> {len(chunks)} chunks")
    print(f"queries: {len(QUERIES)}, k={args.k}, repeats={args.repeats}")
    print(f"backends: {', '.join(backends)}\n")

    if len(backends) == 1:
        print("only one backend available — build the extension for a comparison:")
        print("  pip install maturin")
        print("  maturin develop -m crates/signalment_bm25/Cargo.toml --release\n")

    header = f"| {'backend':<12} | {'build (ms)':>11} | {'search (ms)':>12} | {'speedup':>8} |"
    print(header)
    print("|" + "|".join("-" * len(c) for c in header.split("|")[1:-1]) + "|")

    baseline = None
    for name in backends:
        _, build_best = time_it(lambda: make_retriever(chunks, backend=name), args.repeats)
        retriever = make_retriever(chunks, backend=name)

        def run_queries():
            for text in QUERIES:
                retriever.search(Query(text=text), args.k)

        search_median, _ = time_it(run_queries, args.repeats)
        per_query = search_median / len(QUERIES)
        baseline = baseline if baseline is not None else per_query
        speedup = baseline / per_query if per_query else float("inf")

        print(f"| {name:<12} | {build_best * 1e3:>11.2f} | {per_query * 1e3:>12.3f} "
              f"| {speedup:>7.1f}x |")

    print("\nsearch is per query, median of repeats; build is best of repeats.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
