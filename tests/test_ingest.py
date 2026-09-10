"""Ingest scaffolding and the promotion gate.

The gate is the interesting part. It cannot judge whether a note is good — only
whether the scaffolding was removed, which is the one thing about authorship a
machine can check. These tests pin that it cannot be walked past.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from signalment.ingest import (
    SOURCE_OPEN,
    TODO_MARKER,
    Extraction,
    IngestError,
    check_promotable,
    draft_note,
    normalise,
    read_source,
    suggest_id,
)

WRITTEN = """---
id: vet-x-0001
title: T
species: [feline]
system: endocrine
note_type: mechanism
confidence: reviewed
provenance: self-authored
authored: 2026-09-10
---

## Scope

A real note is at least a couple of paragraphs of prose explaining something,
which is what this stands in for. It has to clear the minimum length check that
stops an empty file being promoted as though it were a note someone wrote.
"""


class TestNormalise:
    def test_removes_soft_hyphens(self):
        # Invisible, survives NFKC, and silently breaks matching inside a drug
        # name. Written as an escape: an invisible literal in source is a trap.
        assert normalise("per\u00admethrin") == "permethrin"

    def test_rejoins_words_broken_across_lines(self):
        assert normalise("hypogly-\ncaemia") == "hypoglycaemia"

    def test_converts_non_breaking_space(self):
        # Handled by NFKC rather than by an explicit replace; pinned so that
        # staying implicit is a decision rather than an accident.
        assert normalise("5\u00a0mg") == "5 mg"

    def test_collapses_runs_of_blank_lines(self):
        assert normalise("a\n\n\n\n\nb") == "a\n\nb"

    def test_leaves_ordinary_text_alone(self):
        assert normalise("Feline hypoglycaemia.") == "Feline hypoglycaemia."


class TestExtraction:
    def test_flags_a_page_with_almost_no_text(self):
        # A picture of handwriting, not text. Reporting it beats returning noise.
        assert Extraction(text="x" * 30, pages=1, chars_per_page=30.0).looks_handwritten

    def test_accepts_a_normal_text_layer(self):
        assert not Extraction(text="x" * 900, pages=1, chars_per_page=900.0).looks_handwritten


class TestReadSource:
    def test_reads_plain_text(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("Some converted handwriting.")
        assert read_source(f).text == "Some converted handwriting."

    def test_missing_file(self, tmp_path):
        with pytest.raises(IngestError, match="no such file"):
            read_source(tmp_path / "nope.txt")

    def test_unsupported_type_names_the_alternative(self, tmp_path):
        f = tmp_path / "a.docx"
        f.write_text("x")
        with pytest.raises(IngestError, match="convert it to text"):
            read_source(f)


class TestSuggestId:
    def test_first_id_for_a_topic(self):
        assert suggest_id("cardio", set()) == "vet-cardio-0001"

    def test_continues_after_the_highest(self):
        existing = {"vet-cardio-0001", "vet-cardio-0007", "vet-endo-0100"}
        assert suggest_id("cardio", existing) == "vet-cardio-0008"

    def test_ignores_other_topics(self):
        assert suggest_id("cardio", {"vet-endo-0042"}) == "vet-cardio-0001"

    def test_normalises_the_topic(self):
        assert suggest_id("Cardio Vascular!", set()) == "vet-cardiovascular-0001"


class TestDraftNote:
    @pytest.fixture
    def extraction(self):
        return Extraction(text="source text here", pages=3, chars_per_page=400.0)

    def test_contains_both_gates(self, extraction):
        text = draft_note(Path("a.pdf"), extraction, "vet-x-0001", "T", "feline", "endocrine")
        assert TODO_MARKER in text
        assert SOURCE_OPEN in text

    def test_starts_as_draft_confidence(self, extraction):
        # Never verified on creation: nobody has checked it yet.
        text = draft_note(Path("a.pdf"), extraction, "vet-x-0001", "T", "feline", "endocrine")
        assert "confidence: draft" in text

    def test_cites_the_source_file(self, extraction):
        text = draft_note(Path("lecture.pdf"), extraction, "vet-x-0001", "T", "feline", "endo")
        assert "lecture.pdf" in text

    def test_rejects_an_unknown_species(self, extraction):
        with pytest.raises(IngestError, match="species must be one of"):
            draft_note(Path("a.pdf"), extraction, "vet-x-0001", "T", "griffin", "endocrine")

    def test_rejects_an_unknown_note_type(self, extraction):
        with pytest.raises(IngestError, match="note_type must be one of"):
            draft_note(Path("a.pdf"), extraction, "vet-x-0001", "T", "feline", "endo", "essay")


class TestPromotionGate:
    def test_a_fresh_draft_is_refused_on_every_count(self):
        draft = draft_note(
            Path("a.pdf"),
            Extraction(text="src", pages=1, chars_per_page=400.0),
            "vet-x-0001",
            "T",
            "feline",
            "endocrine",
        )
        problems = check_promotable(draft)
        assert len(problems) >= 3
        assert any("TODO" in p for p in problems)
        assert any("source block" in p for p in problems)
        assert any("confidence" in p for p in problems)

    def test_a_rewritten_draft_passes(self):
        assert check_promotable(WRITTEN) == []

    def test_pasting_the_source_back_is_caught(self):
        # Deleting the marker but keeping the source is the obvious shortcut.
        assert any("source block" in p for p in check_promotable(WRITTEN + SOURCE_OPEN))

    def test_an_empty_note_is_caught(self):
        stub = WRITTEN.split("## Scope")[0] + "## Scope\n\nToo short.\n"
        assert any("too short" in p for p in check_promotable(stub))

    def test_draft_confidence_alone_blocks_promotion(self):
        assert any(
            "confidence" in p
            for p in check_promotable(
                WRITTEN.replace("confidence: reviewed", "confidence: draft")
            )
        )
