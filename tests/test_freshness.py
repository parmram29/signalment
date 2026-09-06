from datetime import date

import pytest

from signalment.corpus import parse_note
from signalment.freshness import stale_notes

TODAY = date(2026, 9, 1)

TEMPLATE = """---
id: {id}
title: {title}
species: [feline]
system: endocrine
note_type: {note_type}
confidence: verified
provenance: self-authored
authored: {authored}
{reviewed}---

Body text.
"""


def make(note_id, authored, reviewed=None, note_type="mechanism", title="T"):
    line = f"last_reviewed: {reviewed}\n" if reviewed else ""
    return parse_note(TEMPLATE.format(
        id=note_id, title=title, note_type=note_type, authored=authored, reviewed=line
    ))


class TestStaleNotes:
    def test_fresh_note_not_reported(self):
        note = make("vet-a-0001", "2020-01-01", reviewed="2026-08-01")
        assert stale_notes([note], 180, today=TODAY) == []

    def test_stale_by_review_date(self):
        note = make("vet-a-0001", "2020-01-01", reviewed="2025-01-01")
        result = stale_notes([note], 180, today=TODAY)
        assert len(result) == 1
        assert result[0].never_reviewed is False
        assert result[0].basis == "reviewed"

    def test_falls_back_to_authored_when_never_reviewed(self):
        # An unreviewed note's age is measured from authoring — the honest
        # lower bound on how long it has gone unchecked.
        note = make("vet-a-0001", "2025-01-01")
        result = stale_notes([note], 180, today=TODAY)
        assert result[0].never_reviewed is True
        assert result[0].basis == "authored"
        assert result[0].days == (TODAY - date(2025, 1, 1)).days

    def test_review_date_overrides_old_authoring(self):
        note = make("vet-a-0001", "2010-01-01", reviewed="2026-08-20")
        assert stale_notes([note], 180, today=TODAY) == []

    def test_boundary_is_exclusive(self):
        # Exactly at the limit is not yet stale; one day past is.
        at = make("vet-a-0001", "2026-03-05")   # 180 days before TODAY
        assert (TODAY - date(2026, 3, 5)).days == 180
        assert stale_notes([at], 180, today=TODAY) == []
        assert len(stale_notes([at], 179, today=TODAY)) == 1

    def test_clinical_notes_sort_first_despite_being_fresher(self):
        old_definition = make("vet-a-0001", "2020-01-01", note_type="definition")
        newer_protocol = make("vet-b-0002", "2025-06-01", note_type="protocol")
        result = stale_notes([old_definition, newer_protocol], 30, today=TODAY)
        assert result[0].note.id == "vet-b-0002"
        assert result[0].is_clinical is True

    def test_sorted_by_age_within_a_class(self):
        a = make("vet-a-0001", "2024-01-01")
        b = make("vet-b-0002", "2025-01-01")
        result = stale_notes([a, b], 30, today=TODAY)
        assert [s.note.id for s in result] == ["vet-a-0001", "vet-b-0002"]

    def test_negative_age_rejected(self):
        with pytest.raises(ValueError):
            stale_notes([], -1, today=TODAY)

    def test_empty_corpus(self):
        assert stale_notes([], 180, today=TODAY) == []
