import pytest

from vetrag.corpus import chunk_note, load_corpus, parse_note
from vetrag.schema import ValidationError

VALID = """---
id: vet-test-0001
title: Test note
species: [feline]
system: endocrine
note_type: mechanism
confidence: verified
provenance: self-authored
authored: 2026-01-01
---

## Section one

Body text here.
"""


class TestParsing:
    def test_valid_note(self):
        note = parse_note(VALID)
        assert note.id == "vet-test-0001"
        assert note.species == ("feline",)

    def test_context_header_carries_species(self):
        assert "feline" in parse_note(VALID).context_header

    def test_missing_front_matter(self):
        with pytest.raises(ValidationError, match="front-matter"):
            parse_note("# Just a heading\n")

    def test_empty_body_rejected(self):
        with pytest.raises(ValidationError, match="empty"):
            parse_note(VALID.split("## Section")[0])

    def test_unknown_species_rejected(self):
        with pytest.raises(ValidationError, match="unknown species"):
            parse_note(VALID.replace("[feline]", "[griffin]"))

    def test_bad_id_format_rejected(self):
        with pytest.raises(ValidationError, match="vet-<topic>"):
            parse_note(VALID.replace("vet-test-0001", "note1"))

    def test_draft_dosage_note_rejected(self):
        # Clinical numbers must not reach a student marked unreviewed.
        text = VALID.replace("note_type: mechanism", "note_type: dosage").replace(
            "confidence: verified", "confidence: draft"
        )
        with pytest.raises(ValidationError, match="may not be confidence 'draft'"):
            parse_note(text)

    def test_all_errors_reported_together(self):
        text = VALID.replace("vet-test-0001", "bad").replace("[feline]", "[griffin]")
        with pytest.raises(ValidationError) as exc:
            parse_note(text)
        assert "vet-<topic>" in str(exc.value) and "unknown species" in str(exc.value)


class TestChunking:
    def test_chunks_carry_context_header(self):
        chunks = chunk_note(parse_note(VALID))
        assert chunks
        assert all("Test note" in c.text for c in chunks)
        assert all(c.note_id == "vet-test-0001" for c in chunks)

    def test_long_section_is_windowed(self):
        long_note = VALID.replace("Body text here.", " ".join(["word"] * 700))
        chunks = chunk_note(parse_note(long_note), max_tokens=300, overlap=50)
        assert len(chunks) > 1

    def test_chunk_ids_unique(self):
        long_note = VALID.replace("Body text here.", " ".join(["word"] * 700))
        chunks = chunk_note(parse_note(long_note))
        assert len({c.chunk_id for c in chunks}) == len(chunks)


class TestCorpusLoading:
    def test_shipped_corpus_is_valid(self):
        notes, errors = load_corpus("corpus/notes")
        assert errors == []
        assert len(notes) >= 4

    def test_duplicate_ids_reported(self, tmp_path):
        (tmp_path / "a.md").write_text(VALID)
        (tmp_path / "b.md").write_text(VALID)
        _, errors = load_corpus(tmp_path)
        assert any("duplicate id" in msg for _, msg in errors)
