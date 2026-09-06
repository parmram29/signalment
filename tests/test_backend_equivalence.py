"""Differential tests: the Rust backend must match the Python reference.

This suite is the specification the crate is written against. It skips
entirely when the extension is not built, so a clone with no Rust toolchain
still gets a green run.

Equivalence is asserted on **ranking order first**, then scores. Order is what
retrieval quality metrics consume, so an ordering divergence is a real defect
even if every score is within tolerance.
"""

from __future__ import annotations

import pytest

from signalment.corpus import chunk_corpus, load_corpus
from signalment.retrieval.backends import RustBM25Retriever, rust_available
from signalment.retrieval.base import Query
from signalment.retrieval.bm25 import BM25Retriever

pytestmark = pytest.mark.skipif(
    not rust_available(),
    reason="signalment_bm25 extension not built (maturin develop -m crates/signalment_bm25/Cargo.toml)",
)

TOLERANCE = 1e-9

# Ordinary queries plus the cases where a reimplementation typically diverges.
QUERIES = [
    "permethrin sodium channels",
    "why are kittens prone to low blood sugar",
    "hypoglycaemia glycogen neonate",
    "conjugation glucuronidation",
    "insulinoma episodic collapse after exercise",
    "phase II conjugation pathway cats deficient",
    # Repeated terms: must count more than once.
    "permethrin permethrin",
    "cat cat cat insulin",
    # Tokenizer edges.
    "0.5-1 mg/kg",
    "cushing's disease",
    "-leading-hyphen",
    "abécd",
    "CAPITALS AND lowercase",
    # No match at all.
    "cruciate ligament tibial osteotomy",
    "zzzqqq nonexistent",
    # Degenerate.
    "",
    "   ",
]


@pytest.fixture(scope="module")
def chunks():
    notes, errors = load_corpus("corpus/notes")
    assert errors == []
    return chunk_corpus(notes)


@pytest.fixture(scope="module")
def pair(chunks):
    return BM25Retriever(chunks), RustBM25Retriever(chunks)


@pytest.mark.parametrize("text", QUERIES)
def test_identical_ranking(pair, text):
    python, rust = pair
    query = Query(text=text)
    assert [h.note_id for h in rust.search(query, 5)] == [
        h.note_id for h in python.search(query, 5)
    ], f"ranking diverged for {text!r}"


@pytest.mark.parametrize("text", QUERIES)
def test_scores_agree(pair, text):
    python, rust = pair
    query = Query(text=text)
    for got, want in zip(rust.search(query, 5), python.search(query, 5)):
        assert got.score == pytest.approx(want.score, abs=TOLERANCE), (
            f"score diverged for {text!r} on {want.note_id}"
        )


@pytest.mark.parametrize("text", QUERIES)
def test_same_chunk_attribution(pair, text):
    # Deduplication must keep the same chunk, not merely the same note: the
    # reported chunk is what a citation points at.
    python, rust = pair
    query = Query(text=text)
    assert [h.chunk_id for h in rust.search(query, 5)] == [
        h.chunk_id for h in python.search(query, 5)
    ]


@pytest.mark.parametrize("k", [1, 2, 3, 5, 50])
def test_identical_across_k(pair, k):
    python, rust = pair
    query = Query(text="permethrin cats conjugation")
    assert [h.note_id for h in rust.search(query, k)] == [
        h.note_id for h in python.search(query, k)
    ]


def test_empty_corpus_matches(pair):
    python, rust = BM25Retriever([]), RustBM25Retriever([])
    query = Query(text="anything")
    assert rust.search(query, 5) == python.search(query, 5) == []


def test_index_length_matches_corpus(chunks, pair):
    _, rust = pair
    assert len(rust) == len(chunks)


def test_backend_reports_its_name(pair):
    _, rust = pair
    assert rust.name == "bm25-rust"
