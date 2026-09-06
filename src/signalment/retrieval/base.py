"""Retriever interface shared by every ablation row."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

TOKEN = re.compile(r"[a-z0-9][a-z0-9\-']*")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, keeping digits and internal hyphens.

    Dosages ("0.5-1", "mg/kg") and hyphenated drug names are exactly the terms
    a dense retriever handles worst, so the lexical side must not destroy them.
    """
    return TOKEN.findall(text.lower())


@dataclass(frozen=True)
class Hit:
    note_id: str
    score: float
    chunk_id: str = ""


@dataclass(frozen=True)
class Query:
    text: str
    species: str | None = None


class Retriever(Protocol):
    name: str

    def search(self, query: Query, k: int) -> list[Hit]:
        """Return at most k hits, best first, deduplicated by note_id."""
        ...


def dedup_by_note(hits: list[Hit], k: int) -> list[Hit]:
    """Collapse child-chunk hits to their parent notes, keeping the best rank."""
    best: dict[str, Hit] = {}
    for hit in hits:
        if hit.note_id not in best or hit.score > best[hit.note_id].score:
            best[hit.note_id] = hit
    return sorted(best.values(), key=lambda h: (-h.score, h.note_id))[:k]
