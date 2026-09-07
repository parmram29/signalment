"""SQLite FTS5 retrieval backend.

A third BM25 implementation, and a deliberately different *kind* of thing from
the Rust crate:

* `crates/signalment_bm25` is an **accelerator**. It must produce results
  identical to the Python reference — same ranking, same scores to 1e-9 — and
  `tests/test_backend_equivalence.py` enforces that. A divergence is a bug.
* This backend is an **alternative implementation**, written by someone else
  (SQLite's authors) with its own tokenizer and its own BM25 constants
  (`k1 = 1.2`, `b = 0.75`, against our `1.5 / 0.75`). It will *not* rank
  identically, and demanding that it should would be a category error.

So it is held to a different standard: not equality, but competitiveness on the
gold set. That makes it an ablation row rather than a correctness check — and a
useful one, because a third opinion from an independent implementation is the
cheapest evidence available that our own retrieval is not quietly wrong.

Requires no new dependency: FTS5 ships with CPython's bundled SQLite.
"""

from __future__ import annotations

import sqlite3

from ..corpus import Chunk
from .base import Hit, Query, dedup_by_note, tokenize


def fts5_available() -> bool:
    """Whether this interpreter's SQLite was compiled with FTS5."""
    try:
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        finally:
            conn.close()
    except sqlite3.OperationalError:
        return False
    return True


def to_match_expression(text: str) -> str:
    """Turn free text into a safe FTS5 MATCH expression.

    User text cannot go into MATCH unescaped: bare `AND`, `OR`, `NOT`, `NEAR`,
    `*`, `^`, `:` and unbalanced quotes are all query *syntax* to FTS5, so a
    student typing `dose OR overdose` would get operator behaviour they did not
    ask for, and a stray quote raises OperationalError mid-search.

    Reusing the project tokenizer strips every operator by construction and
    keeps this backend seeing the same terms as the other two. Each token is
    then quoted as a literal phrase, with internal quotes doubled per SQL
    string-escaping rules, and the terms are OR-ed so the query is a bag of
    words rather than a conjunction.
    """
    tokens = tokenize(text)
    return " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens)


class FTS5Retriever:
    """BM25 over an in-memory SQLite FTS5 index.

    The index is rebuilt on construction, which at this corpus size costs
    milliseconds and removes any question of a stale index on disk.
    """

    name = "bm25-sqlite"

    def __init__(self, chunks: list[Chunk]) -> None:
        if not fts5_available():
            raise RuntimeError(
                "this Python's SQLite was built without FTS5; use the python or "
                "rust backend instead"
            )
        self.chunks = chunks
        self._conn = sqlite3.connect(":memory:")
        # note_id and chunk_id are UNINDEXED: they are payload to return, not
        # text to match on. Indexing them would let a query accidentally match
        # an identifier.
        self._conn.execute(
            "CREATE VIRTUAL TABLE chunks USING fts5("
            "  note_id UNINDEXED, chunk_id UNINDEXED, body, tokenize='unicode61'"
            ")"
        )
        self._conn.executemany(
            "INSERT INTO chunks (note_id, chunk_id, body) VALUES (?, ?, ?)",
            [(c.note_id, c.chunk_id, c.text) for c in chunks],
        )
        self._conn.commit()

    def search(self, query: Query, k: int) -> list[Hit]:
        expression = to_match_expression(query.text)
        if not expression:
            return []

        # SQLite's bm25() returns a *negative* score, most relevant most
        # negative, so it can be ORDER BY'd ascending. Negate it to match this
        # project's convention of higher-is-better.
        rows = self._conn.execute(
            "SELECT note_id, chunk_id, -bm25(chunks) AS score "
            "FROM chunks WHERE chunks MATCH ? ORDER BY score DESC LIMIT ?",
            (expression, k * 5),
        ).fetchall()

        hits = [
            Hit(note_id=note_id, score=score, chunk_id=chunk_id)
            for note_id, chunk_id, score in rows
            if score > 0
        ]
        return dedup_by_note(hits, k)

    def __len__(self) -> int:
        return len(self.chunks)

    def close(self) -> None:
        """Release the SQLite connection.

        Harmless to skip in a test run, where the process exits anyway. It is
        not harmless in the Phase 6 API, which builds retrievers per request:
        an unclosed connection there leaks a file descriptor per call.
        """
        self._conn.close()

    def __enter__(self) -> FTS5Retriever:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        # Last resort for callers that neither close() nor use `with`.
        # __init__ can raise before _conn exists, hence the guard.
        conn = getattr(self, "_conn", None)
        if conn is not None:
            conn.close()
