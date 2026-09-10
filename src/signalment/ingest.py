"""Turning source material into note drafts.

The corpus only ever contains notes a person wrote. That rule is what makes
every number this project reports mean anything, and it is also, bluntly, a
chore: opening a PDF, reading it, and hand-typing front-matter for each note is
the slowest part of using this system.

This module removes the chore without removing the rule. It extracts text from
a source file and scaffolds a **draft** — front-matter filled in, source text
attached for reference — into `drafts/`, which nothing indexes. Promoting a
draft into `corpus/notes/` requires that the scaffolding be gone: the TODO
marker removed and the source block deleted. You cannot promote a file you have
not actually rewritten, because the parts you would have to delete are exactly
the parts that prove you did not.

So the human step is preserved by construction rather than by good intentions.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .schema import CONFIDENCE, NOTE_TYPES, PROVENANCE, SPECIES

#: Removing these is the act that says "I rewrote this."
TODO_MARKER = "TODO: replace this with your own note"
SOURCE_OPEN = "<!-- SOURCE TEXT — reference only. Delete this whole block before promoting."
SOURCE_CLOSE = "-->"

#: Below this, a PDF page has a picture of words rather than words.
MIN_CHARS_PER_PAGE = 40


class IngestError(Exception):
    """Raised with an explanation a beta user can act on without reading code."""


@dataclass(frozen=True)
class Extraction:
    text: str
    pages: int
    chars_per_page: float

    @property
    def looks_handwritten(self) -> bool:
        return self.chars_per_page < MIN_CHARS_PER_PAGE


def extract_pdf(path: Path) -> Extraction:
    """Pull the text layer out of a PDF.

    A PDF of handwriting has no text layer, and no amount of parsing invents
    one. That case is detected and reported rather than returning a few
    characters of noise that look like success.
    """
    try:
        import logging

        from pypdf import PdfReader

        # pypdf narrates malformed cross-reference entries at WARNING. They are
        # normal in exported PDFs, it recovers from them, and a wall of
        # "Ignoring wrong pointing object" tells a beta user nothing.
        logging.getLogger("pypdf").setLevel(logging.ERROR)
    except ImportError as exc:
        raise IngestError(
            'reading PDFs needs pypdf. Install it with:\n  pip install -e ".[ingest]"'
        ) from exc

    try:
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:
        raise IngestError(f"could not read {path.name}: {type(exc).__name__}: {exc}") from exc

    if not pages:
        raise IngestError(f"{path.name} has no pages")

    text = normalise("\n\n".join(pages))
    return Extraction(text=text, pages=len(pages), chars_per_page=len(text) / len(pages))


def normalise(text: str) -> str:
    """Tidy extracted text without editing its content.

    PDF extraction yields ligatures, non-breaking spaces, soft hyphens and
    lines broken mid-word by the original layout. Left alone they follow the
    text into a note and then into the index, where a soft hyphen inside a drug
    name quietly makes it unsearchable.
    """
    # NFKC already folds ligatures and turns a non-breaking space into a plain
    # one. It does *not* remove the soft hyphen, which survives as an invisible
    # character inside a word and quietly makes that word unmatchable.
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "")
    # A hyphen at end of line is the layout breaking a word, not part of it.
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_source(path: Path) -> Extraction:
    """Extract from a supported source file."""
    if not path.exists():
        raise IngestError(f"no such file: {path}")
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path)
    if suffix in {".txt", ".md"}:
        text = normalise(path.read_text(encoding="utf-8", errors="replace"))
        return Extraction(text=text, pages=1, chars_per_page=float(len(text)))
    raise IngestError(
        f"unsupported file type {suffix or '(none)'}. Supported: .pdf, .txt, .md.\n"
        f"For handwriting, convert it to text in your notes app first and save "
        f"the result as .txt — stroke recognition beats any OCR on pixels."
    )


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "untitled"


def suggest_id(topic: str, existing: set[str]) -> str:
    """Next free `vet-<topic>-NNNN` for this topic."""
    topic = re.sub(r"[^a-z0-9]", "", topic.lower()) or "misc"
    used = {
        int(m.group(1))
        for note_id in existing
        if (m := re.fullmatch(rf"vet-{re.escape(topic)}-(\d{{4}})", note_id))
    }
    nxt = max(used) + 1 if used else 1
    if nxt > 9999:
        raise IngestError(f"topic {topic!r} has no free ids left")
    return f"vet-{topic}-{nxt:04d}"


def draft_body(source_text: str, source_name: str) -> str:
    """The draft's body: an empty place to write, and the source beneath it."""
    return f"""## {TODO_MARKER}

Write what you understood, in your own words. Not a summary of the text below —
an explanation you could give someone else. Where you cannot explain it without
looking, that is the gap worth going back for.

Delete this heading and the source block below before promoting. `promote`
refuses a draft that still contains either, which is the only guarantee this
system has that a note was written rather than pasted.

{SOURCE_OPEN}
Extracted from: {source_name}

{source_text}
{SOURCE_CLOSE}
"""


def draft_note(
    source: Path,
    extraction: Extraction,
    note_id: str,
    title: str,
    species: str,
    system: str,
    note_type: str = "mechanism",
    authored: str | None = None,
) -> str:
    """Render a complete draft file, front-matter and all."""
    from datetime import date

    for value, allowed, field in (
        (species, SPECIES, "species"),
        (note_type, NOTE_TYPES, "note_type"),
    ):
        if value not in allowed:
            raise IngestError(f"{field} must be one of {sorted(allowed)}, got {value!r}")

    return f"""---
id: {note_id}
title: {title}
species: [{species}]
system: {system}
note_type: {note_type}
confidence: draft
provenance: self-authored
authored: {authored or date.today().isoformat()}
source_refs: ["{source.name}"]
---

{draft_body(extraction.text, source.name)}"""


def check_promotable(text: str) -> list[str]:
    """Reasons this draft is not ready, empty if it is.

    Deliberately mechanical. It cannot judge whether a note is good — only
    whether the scaffolding was removed, which is the one thing a machine can
    check about authorship.
    """
    problems = []
    if TODO_MARKER in text:
        problems.append("the TODO heading is still present — write the note first")
    if SOURCE_OPEN in text or "SOURCE TEXT" in text:
        problems.append("the source block is still present — delete it, do not paste from it")
    body = text.split("---", 2)[-1]
    prose = re.sub(r"^#.*$", "", body, flags=re.MULTILINE).strip()
    if len(prose) < 120:
        problems.append(f"the note is only {len(prose)} characters — too short to be a note")
    if "confidence: draft" in text:
        problems.append(
            "confidence is still 'draft' — set it to 'reviewed' or 'verified' "
            "once you have checked it against a source"
        )
    return problems


def parse_confidence_hint(text: str) -> str | None:
    """The confidence value declared in a draft, if any."""
    match = re.search(r"^confidence:\s*(\w[\w-]*)", text, re.MULTILINE)
    if match and match.group(1) in CONFIDENCE:
        return match.group(1)
    return None


def parse_provenance_hint(text: str) -> str | None:
    match = re.search(r"^provenance:\s*([\w-]+)", text, re.MULTILINE)
    if match and match.group(1) in PROVENANCE:
        return match.group(1)
    return None
