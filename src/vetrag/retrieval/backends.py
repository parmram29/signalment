"""Retrieval backend selection.

The pure-Python BM25 is the reference implementation and is never removed. The
Rust extension is an accelerator: optional, selected at runtime, and verified
against the Python results by `tests/test_backend_equivalence.py`. A clone with
no Rust toolchain still runs everything.

This is the pattern `orjson` and `numpy` use — an accelerated path when it is
available, a correct path always.
"""

from __future__ import annotations

from ..corpus import Chunk
from .base import Hit, Query
from .bm25 import BM25Retriever

#: Backends in preference order for `backend="auto"`.
PREFERENCE = ("rust", "python")


def rust_available() -> bool:
    """Whether the compiled extension can be imported."""
    try:
        import vetrag_bm25  # noqa: F401
    except ImportError:
        return False
    return True


def available_backends() -> tuple[str, ...]:
    return tuple(name for name in PREFERENCE if name != "rust" or rust_available())


class RustBM25Retriever:
    """Thin adapter over the compiled index.

    Contains no scoring logic — if a result differs from `BM25Retriever`, the
    bug is in the crate, not here. Deduplication by note and truncation to k
    both happen inside Rust, so this only converts tuples into `Hit`s.
    """

    name = "bm25-rust"

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        import vetrag_bm25

        self.chunks = chunks
        self._index = vetrag_bm25.Bm25Index(
            [chunk.indexed_text for chunk in chunks],
            [chunk.note_id for chunk in chunks],
            [chunk.chunk_id for chunk in chunks],
            k1,
            b,
        )

    def search(self, query: Query, k: int) -> list[Hit]:
        return [
            Hit(note_id=note_id, score=score, chunk_id=chunk_id)
            for note_id, score, chunk_id in self._index.search(query.text, k)
        ]

    def __len__(self) -> int:
        return len(self._index)


def make_retriever(
    chunks: list[Chunk],
    backend: str = "auto",
    k1: float = 1.5,
    b: float = 0.75,
):
    """Build a BM25 retriever using the named backend.

    `auto` prefers the compiled extension and falls back silently to Python.
    An explicitly requested backend that is unavailable raises instead — a
    benchmark that quietly measured the wrong implementation would be worse
    than one that failed.
    """
    if backend == "auto":
        backend = available_backends()[0]

    if backend == "python":
        return BM25Retriever(chunks, k1=k1, b=b)
    if backend == "rust":
        if not rust_available():
            raise RuntimeError(
                "the vetrag_bm25 extension is not built. Run:\n"
                "  pip install maturin\n"
                "  maturin develop -m crates/vetrag_bm25/Cargo.toml --release"
            )
        return RustBM25Retriever(chunks, k1=k1, b=b)

    raise ValueError(f"unknown backend {backend!r}; expected one of {PREFERENCE + ('auto',)}")
