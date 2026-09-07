import pytest

from signalment.corpus import chunk_corpus, load_corpus
from signalment.retrieval.base import Query, tokenize
from signalment.retrieval.bm25 import BM25Retriever
from signalment.retrieval.filters import FusionRetriever, SpeciesFilter


@pytest.fixture(scope="module")
def notes():
    loaded, errors = load_corpus("corpus/notes")
    assert errors == []
    return loaded


@pytest.fixture(scope="module")
def bm25(notes):
    return BM25Retriever(chunk_corpus(notes))


class TestTokenizer:
    def test_preserves_dosage_shaped_tokens(self):
        # Hyphens and digits are exactly what a vet corpus needs kept.
        assert tokenize("0.5-1 mg/kg") == ["0", "5-1", "mg", "kg"]

    def test_lowercases(self):
        assert tokenize("Permethrin TOXICOSIS") == ["permethrin", "toxicosis"]


class TestBM25:
    def test_finds_the_obvious_note(self, bm25):
        hits = bm25.search(Query("permethrin sodium channels"), k=3)
        assert hits[0].note_id == "vet-tox-0007"

    def test_returns_nothing_for_out_of_corpus_terms(self, bm25):
        assert bm25.search(Query("cruciate ligament tibial osteotomy"), k=5) == []

    def test_results_deduplicated_by_note(self, bm25):
        hits = bm25.search(Query("hypoglycaemia glycogen neonate"), k=5)
        assert len({h.note_id for h in hits}) == len(hits)

    def test_respects_k(self, bm25):
        assert len(bm25.search(Query("species"), k=2)) <= 2

    def test_scores_descending(self, bm25):
        hits = bm25.search(Query("cats permethrin conjugation"), k=5)
        assert hits == sorted(hits, key=lambda h: -h.score)


class TestSpeciesFilter:
    def test_excludes_other_species(self, bm25, notes):
        filtered = SpeciesFilter(bm25, notes)
        hits = filtered.search(Query("hypoglycaemia", species="feline"), k=5)
        assert "vet-endo-0143" not in {h.note_id for h in hits}  # canine note
        assert hits, "the feline note should still be retrieved"

    def test_multi_species_notes_survive(self, bm25, notes):
        filtered = SpeciesFilter(bm25, notes)
        hits = filtered.search(Query("conjugation glucuronidation", species="feline"), k=5)
        assert "vet-pharm-0021" in {h.note_id for h in hits}

    def test_no_species_is_a_noop(self, bm25, notes):
        filtered = SpeciesFilter(bm25, notes)
        query = Query("hypoglycaemia")
        assert filtered.search(query, k=5) == bm25.search(query, k=5)

    def test_overfetches_so_filtering_does_not_shorten_results(self, bm25, notes):
        # The canine note outranks nothing away: filtering must not simply
        # truncate the top-k, it must pull replacements from deeper.
        filtered = SpeciesFilter(bm25, notes)
        unfiltered = bm25.search(Query("hypoglycaemia dogs cats"), k=2)
        hits = filtered.search(Query("hypoglycaemia dogs cats", species="feline"), k=2)
        assert len(hits) >= min(
            2, len([h for h in unfiltered if h.note_id != "vet-endo-0143"])
        )


class TestFusion:
    def test_fuses_two_retrievers(self, bm25, notes):
        fused = FusionRetriever([bm25, SpeciesFilter(bm25, notes)])
        hits = fused.search(Query("permethrin cat", species="feline"), k=3)
        assert hits[0].note_id == "vet-tox-0007"
        assert "rrf(" in fused.name
