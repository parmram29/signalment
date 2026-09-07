"""Corpus loading and parent/child chunking."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from .schema import Note, ValidationError, validate

FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", re.DOTALL)


def parse_note(text: str, path: str = "") -> Note:
    """Parse a Markdown file with YAML front-matter into a validated Note."""
    match = FRONT_MATTER.match(text)
    if not match:
        raise ValidationError("file does not begin with a --- front-matter block")
    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        raise ValidationError(f"invalid YAML front-matter: {exc}") from exc
    if not isinstance(meta, dict):
        raise ValidationError("front-matter must be a mapping")
    return validate(meta, body=match.group(2), path=path)


def load_corpus(root: str | Path) -> tuple[list[Note], list[tuple[str, str]]]:
    """Load every .md note under root.

    Returns the valid notes and a list of (path, error) for the rest, so the
    linter can report all failures in one pass instead of stopping at the first.
    """
    root = Path(root)
    notes: list[Note] = []
    errors: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*.md")):
        # Documentation living beside the notes is not itself a note.
        if path.name == "README.md" or path.name.startswith("_"):
            continue
        try:
            notes.append(parse_note(path.read_text(encoding="utf-8"), path=str(path)))
        except ValidationError as exc:
            errors.append((str(path), str(exc)))

    seen: dict[str, str] = {}
    for note in notes:
        if note.id in seen:
            errors.append(
                (note.path, f"duplicate id {note.id!r}, first seen in {seen[note.id]}")
            )
        seen[note.id] = note.path
    return notes, errors


@dataclass(frozen=True)
class Chunk:
    """A child chunk. Indexed for matching; its parent note is what gets read."""

    chunk_id: str
    note_id: str
    text: str
    heading: str = ""


def chunk_note(note: Note, max_tokens: int = 300, overlap: int = 50) -> list[Chunk]:
    """Split a note into child chunks along Markdown headings.

    Headings are the author's own semantic boundaries, so they beat a fixed
    window. Sections longer than max_tokens are windowed with overlap. Every
    chunk is prefixed with the note's context header so a retrieved fragment
    still carries its title, species and body system.
    """
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in note.body.splitlines():
        if line.startswith("#"):
            sections.append((line.lstrip("#").strip(), []))
        else:
            sections[-1][1].append(line)

    chunks: list[Chunk] = []
    for heading, lines in sections:
        words = "\n".join(lines).split()
        if not words:
            continue
        step = max(max_tokens - overlap, 1)
        for start in range(0, len(words), step):
            window = words[start : start + max_tokens]
            if not window:
                continue
            head = f" ## {heading}" if heading else ""
            chunks.append(
                Chunk(
                    chunk_id=f"{note.id}#{len(chunks)}",
                    note_id=note.id,
                    heading=heading,
                    text=f"{note.context_header}{head}\n{' '.join(window)}",
                )
            )
            if start + max_tokens >= len(words):
                break
    return chunks


def chunk_corpus(notes: list[Note], **kwargs) -> list[Chunk]:
    return [chunk for note in notes for chunk in chunk_note(note, **kwargs)]
