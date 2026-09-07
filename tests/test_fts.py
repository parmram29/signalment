"""SQLite FTS5 backend.

Held to a different standard than the Rust crate. Rust is an accelerator and
must match the Python reference exactly; SQLite is an independent
implementation with its own tokenizer and its own BM25 constants (k1=1.2 against
our 1.5), so identical ranking is neither expected nor desirable to demand.

What is asserted instead: it is safe against query syntax, and it is
*competitive* on the gold set. A third implementation agreeing on which notes
answer a question is the cheapest evidence available that our own retrieval is
not quietly wrong.
"""

from __future__ import annotations

import pytest

from signalment.corpus import chunk_corpus, load_corpus
from signalment.evaluation.dataset import load_gold
from signalment.evaluation.runner import evaluate
from signalment.retrieval.base import Query
from signalment.retrieval.bm25 import BM25Retriever
from signalment.retrieval.fts import FTS5Retriever, fts5_available, to_match_expression

pytestmark = pytest.mark.skipif(
    not fts5_available(), reason="this Python's SQLite was built without FTS5"
)


@pytest.fixture(scope="module")
def notes():
    loaded, errors = load_corpus("corpus/notes")
    assert errors == []
    return loaded


@pytest.fixture(scope="module")
def chunks(notes):
    return chunk_corpus(notes)


@pytest.fixture(scope="module")
def fts(chunks):
    return FTS5Retriever(chunks)


class TestMatchExpression:
    def test_operators_are_neutralised(self):
        # Bare OR/AND/NOT/NEAR are FTS5 syntax. A student typing them must get
        # words, not operator semantics.
        # Five tokens (dose, or, overdose, not, near) joined by four separators.
        expression = to_match_expression("dose OR overdose NOT NEAR")
        assert expression.count(" OR ") == 4
        assert '"or"' in expression and '"not"' in expression and '"near"' in expression

    def test_quotes_are_escaped(self):
        assert to_match_expression('say "hi"') == '"say" OR "hi"'

    def test_wildcards_and_colons_are_stripped(self):
        # `*`, `^` and `:` are all FTS5 syntax; the tokenizer drops them.
        assert to_match_expression("perme* ^start col:on") == (
            '"perme" OR "start" OR "col" OR "on"'
        )

    def test_punctuation_only_yields_nothing(self):
        assert to_match_expression("... ??? !!!") == ""


class TestSearch:
    def test_finds_the_obvious_note(self, fts):
        assert fts.search(Query("permethrin sodium channels"), 3)[0].note_id == "vet-tox-0007"

    def test_no_match_returns_nothing(self, fts):
        assert fts.search(Query("cruciate ligament tibial osteotomy"), 5) == []

    def test_empty_query_returns_nothing(self, fts):
        assert fts.search(Query(""), 5) == []
        assert fts.search(Query("   ..."), 5) == []

    def test_operator_query_does_not_raise(self, fts):
        # Unescaped, this is a syntax error inside SQLite, not a search.
        assert fts.search(Query('dose OR "unbalanced'), 5) is not None

    def test_deduplicated_by_note(self, fts):
        hits = fts.search(Query("hypoglycaemia glycogen neonate"), 5)
        assert len({h.note_id for h in hits}) == len(hits)

    def test_respects_k(self, fts):
        assert len(fts.search(Query("species conjugation cats"), 2)) <= 2

    def test_scores_descending_and_positive(self, fts):
        hits = fts.search(Query("permethrin cats conjugation"), 5)
        assert hits == sorted(hits, key=lambda h: -h.score)
        assert all(h.score > 0 for h in hits)

    def test_index_length_matches_corpus(self, fts, chunks):
        assert len(fts) == len(chunks)


class TestCompetitiveness:
    def test_matches_python_recall_on_the_gold_set(self, notes, chunks):
        # The real bar: not identical ranking, but finding the same answers.
        gold = load_gold("eval/gold/gold_v1.jsonl", {n.id for n in notes})
        python = evaluate(BM25Retriever(chunks), gold, k=5)
        sqlite = evaluate(FTS5Retriever(chunks), gold, k=5)
        assert sqlite.retrieval["recall@5"] >= python.retrieval["recall@5"]

    def test_agrees_with_python_on_the_top_note(self, chunks):
        # Independent implementations disagreeing about the single best answer
        # on unambiguous queries would mean one of them is wrong.
        python, sqlite = BM25Retriever(chunks), FTS5Retriever(chunks)
        for text in (
            "permethrin sodium channels",
            "neonatal kitten glycogen reserves",
            "insulinoma episodic collapse",
        ):
            query = Query(text)
            assert sqlite.search(query, 1)[0].note_id == python.search(query, 1)[0].note_id, (
                text
            )
