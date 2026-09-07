"""Note front-matter schema and validation.

The schema is the project's contract with the corpus. Metadata is attached at
ingestion, never bolted on afterwards, because retrieval filters and citation
provenance both read from it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

# Species is a hard retrieval filter, not a hint. Veterinary medicine's
# characteristic hazard is a canine answer returned for a feline question:
# the two are near-identical semantically and can differ lethally in dosing.
SPECIES = {
    "canine",
    "feline",
    "equine",
    "bovine",
    "ovine",
    "caprine",
    "porcine",
    "avian",
    "lagomorph",
    "rabbit",
    "ferret",
    "rodent",
    "reptile",
    "multi",
}

NOTE_TYPES = {"definition", "mechanism", "differential", "protocol", "dosage", "case"}

# Trust axis for a self-authored corpus. Notes are reliable to their author,
# but a medical assistant must be able to rank and filter by review state.
CONFIDENCE = {"draft", "reviewed", "verified"}

PROVENANCE = {
    "self-authored",
    "lecture",
    "textbook",
    "guideline",
    "primary-literature",
    "ai-drafted",
}

# Content a model wrote. It may be a useful starting point, but "verified" is a
# claim that a human checked it against a source, and a model cannot make that
# claim on its own behalf. Promoting an ai-drafted note is a deliberate human
# act: read it, check the citation, then change this field.
UNVERIFIABLE_PROVENANCE = frozenset({"ai-drafted"})

ID_PATTERN = re.compile(r"^vet-[a-z0-9]+-\d{4}$")

REQUIRED = (
    "id",
    "title",
    "species",
    "system",
    "note_type",
    "confidence",
    "provenance",
    "authored",
)


@dataclass(frozen=True)
class Note:
    """A validated atomic note. One idea, standalone, addressable by id."""

    id: str
    title: str
    species: tuple[str, ...]
    system: str
    note_type: str
    confidence: str
    provenance: str
    authored: date
    body: str
    path: str = ""
    life_stage: tuple[str, ...] = ()
    source_refs: tuple[str, ...] = ()
    last_reviewed: date | None = None
    prerequisites: tuple[str, ...] = ()
    see_also: tuple[str, ...] = ()
    tags_vetsct: tuple[str, ...] = ()

    @property
    def context_header(self) -> str:
        """Deterministic context prepended to each chunk before indexing.

        This is the cheap half of Anthropic's contextual retrieval: the
        document-level context that a bare chunk loses is already present in
        the front-matter, so it costs no LLM call and cannot hallucinate.
        """
        return f"[{self.title} | {'/'.join(self.species)} | {self.system}]"


class ValidationError(Exception):
    """Raised with every problem found in a note, not just the first."""


def _as_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    raise ValidationError(
        f"{field_name}: expected a string or list, got {type(value).__name__}"
    )


def _as_date(value: object, field_name: str) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValidationError(f"{field_name}: {exc}") from exc
    raise ValidationError(f"{field_name}: expected an ISO date, got {type(value).__name__}")


def validate(meta: dict, body: str = "", path: str = "") -> Note:
    """Build a Note from parsed front-matter, collecting all errors at once."""
    errors: list[str] = []

    for key in REQUIRED:
        if meta.get(key) in (None, "", []):
            errors.append(f"missing required field: {key}")
    if errors:
        raise ValidationError("; ".join(errors))

    note_id = str(meta["id"])
    if not ID_PATTERN.match(note_id):
        errors.append(f"id {note_id!r} must match vet-<topic>-<4 digits>")

    species = _as_tuple(meta.get("species"), "species")
    unknown = sorted(set(species) - SPECIES)
    if unknown:
        errors.append(f"unknown species: {', '.join(unknown)}")

    if meta["note_type"] not in NOTE_TYPES:
        errors.append(f"note_type must be one of {sorted(NOTE_TYPES)}")
    if meta["confidence"] not in CONFIDENCE:
        errors.append(f"confidence must be one of {sorted(CONFIDENCE)}")
    if meta["provenance"] not in PROVENANCE:
        errors.append(f"provenance must be one of {sorted(PROVENANCE)}")

    # A dosage or protocol note carries clinical numbers; an unreviewed one
    # must not reach a student unmarked, so the linter blocks it at ingestion.
    if meta["note_type"] in {"dosage", "protocol"} and meta["confidence"] == "draft":
        errors.append(f"note_type {meta['note_type']!r} may not be confidence 'draft'")

    # "verified" means a person checked this against its source. Generated
    # content cannot assert that about itself, and a corpus that lets it is no
    # longer a corpus of things known to be true.
    if meta["provenance"] in UNVERIFIABLE_PROVENANCE and meta["confidence"] == "verified":
        errors.append(
            f"provenance {meta['provenance']!r} may not be confidence 'verified': "
            f"verification is a human act. Read it, check the source, then set "
            f"provenance to what you actually confirmed."
        )

    try:
        authored = _as_date(meta["authored"], "authored")
    except ValidationError as exc:
        errors.append(str(exc))
        authored = date.today()

    last_reviewed = None
    if meta.get("last_reviewed"):
        try:
            last_reviewed = _as_date(meta["last_reviewed"], "last_reviewed")
        except ValidationError as exc:
            errors.append(str(exc))

    if not body.strip():
        errors.append("note body is empty")

    if errors:
        raise ValidationError("; ".join(errors))

    return Note(
        id=note_id,
        title=str(meta["title"]),
        species=species,
        system=str(meta["system"]),
        note_type=str(meta["note_type"]),
        confidence=str(meta["confidence"]),
        provenance=str(meta["provenance"]),
        authored=authored,
        body=body.strip(),
        path=path,
        life_stage=_as_tuple(meta.get("life_stage"), "life_stage"),
        source_refs=_as_tuple(meta.get("source_refs"), "source_refs"),
        last_reviewed=last_reviewed,
        prerequisites=_as_tuple(meta.get("prerequisites"), "prerequisites"),
        see_also=_as_tuple(meta.get("see_also"), "see_also"),
        tags_vetsct=_as_tuple(meta.get("tags_vetsct"), "tags_vetsct"),
    )
