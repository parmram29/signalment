import pytest

from signalment.corpus import chunk_corpus, load_corpus, parse_note
from signalment.evaluation.dataset import GoldItem
from signalment.evaluation.drift import (
    LABELS_UNREACHABLE,
    MAY_BE_ANSWERABLE,
    check_drift,
    format_warnings,
)
from signalment.retrieval.bm25 import BM25Retriever

NEW_NOTE = """---
id: vet-ortho-0001
title: Cranial cruciate ligament rupture in dogs
species: [canine]
system: musculoskeletal
note_type: differential
confidence: verified
provenance: self-authored
authored: 2026-09-01
---

## Surgical approaches

Recommended surgical approach for cranial cruciate ligament rupture depends on
the dog. Tibial osteotomy techniques alter joint biomechanics; suture-based
techniques aim to stabilise the stifle directly.
"""


@pytest.fixture
def notes():
    loaded, errors = load_corpus("corpus/notes")
    assert errors == []
    return loaded


@pytest.fixture
def gold():
    return [
        GoldItem(
            qid="a1",
            question="Why are kittens prone to low blood sugar?",
            answerable=True,
            relevant_note_ids=("vet-endo-0142",),
        ),
        GoldItem(
            qid="a2",
            question="How does permethrin affect sodium channels?",
            answerable=True,
            relevant_note_ids=("vet-tox-0007",),
        ),
        GoldItem(
            qid="u1",
            answerable=False,
            question="What is the recommended surgical approach for a canine "
            "cranial cruciate ligament rupture?",
        ),
    ]


class TestDrift:
    def test_clean_gold_set_has_no_warnings(self, notes, gold):
        retriever = BM25Retriever(chunk_corpus(notes))
        assert check_drift(retriever, gold, k=5) == []

    def test_detects_unanswerable_item_that_became_answerable(self, notes, gold):
        # The Phase 2 exit criterion: adding a note that answers a
        # previously-unanswerable gold item must produce a visible warning.
        grown = [*notes, parse_note(NEW_NOTE)]
        retriever = BM25Retriever(chunk_corpus(grown))
        warnings = check_drift(retriever, gold, k=5)

        assert len(warnings) == 1
        assert warnings[0].qid == "u1"
        assert warnings[0].kind == MAY_BE_ANSWERABLE
        assert "vet-ortho-0001" in warnings[0].retrieved

    def test_threshold_self_calibrates_from_answerable_items(self, notes, gold):
        # No hardcoded score constant: the bar is the median top score of
        # answerable items, so it keeps meaning something as the corpus grows.
        grown = [*notes, parse_note(NEW_NOTE)]
        retriever = BM25Retriever(chunk_corpus(grown))
        assert check_drift(retriever, gold, k=5, strong_score=1e9) == []
        assert check_drift(retriever, gold, k=5, strong_score=0.1) != []

    def test_detects_labels_that_match_nothing(self, notes):
        item = GoldItem(
            qid="a9",
            question="zzzqqq nonexistent terminology",
            answerable=True,
            relevant_note_ids=("vet-endo-0142",),
        )
        retriever = BM25Retriever(chunk_corpus(notes))
        warnings = check_drift(retriever, [item], k=5)
        assert [w.kind for w in warnings] == [LABELS_UNREACHABLE]

    def test_no_answerable_baseline_means_no_false_alarms(self, notes):
        # With nothing to calibrate against, an unanswerable item cannot be
        # judged suspicious — silence beats a guess.
        item = GoldItem(qid="u9", question="permethrin cats", answerable=False)
        retriever = BM25Retriever(chunk_corpus(notes))
        assert check_drift(retriever, [item], k=5) == []


class TestFormatting:
    def test_clean_message(self):
        assert format_warnings([]) == "gold set: no drift detected"

    def test_warning_message_names_the_item_and_action(self, notes, gold):
        grown = [*notes, parse_note(NEW_NOTE)]
        text = format_warnings(check_drift(BM25Retriever(chunk_corpus(grown)), gold, k=5))
        assert "u1" in text and "relabel" in text
