"""Gold evaluation set: loading and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GoldItem:
    """One evaluation question.

    An unanswerable item has no relevant notes: the corpus genuinely cannot
    support an answer and the correct behaviour is to abstain. A gold set with
    no unanswerable items cannot detect the failure mode that matters most in
    a clinical domain, so the loader refuses to accept one.
    """

    qid: str
    question: str
    answerable: bool
    relevant_note_ids: tuple[str, ...] = ()
    species: str | None = None
    expected_answer: str = ""
    rationale: str = ""

    @classmethod
    def from_dict(cls, raw: dict) -> GoldItem:
        missing = {"qid", "question", "answerable"} - raw.keys()
        if missing:
            raise ValueError(f"gold item missing fields: {sorted(missing)}")
        answerable = bool(raw["answerable"])
        ids = tuple(raw.get("relevant_note_ids") or ())
        if answerable and not ids:
            raise ValueError(f"{raw['qid']}: answerable item needs relevant_note_ids")
        if not answerable and ids:
            raise ValueError(f"{raw['qid']}: unanswerable item must have no relevant_note_ids")
        return cls(
            qid=str(raw["qid"]),
            question=str(raw["question"]),
            answerable=answerable,
            relevant_note_ids=ids,
            species=raw.get("species"),
            expected_answer=raw.get("expected_answer", ""),
            rationale=raw.get("rationale", ""),
        )


def load_gold(path: str | Path, known_note_ids: set[str] | None = None) -> list[GoldItem]:
    """Load a JSONL gold set, failing loudly on a malformed or unusable one."""
    items: list[GoldItem] = []
    seen: set[str] = set()
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            item = GoldItem.from_dict(json.loads(line))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{path}:{lineno}: {exc}") from exc
        if item.qid in seen:
            raise ValueError(f"{path}:{lineno}: duplicate qid {item.qid!r}")
        seen.add(item.qid)
        items.append(item)

    if not items:
        raise ValueError(f"{path}: gold set is empty")
    if not any(not i.answerable for i in items):
        raise ValueError(
            f"{path}: no unanswerable items. Abstention cannot be measured without them."
        )

    if known_note_ids is not None:
        # A gold label pointing at a note that no longer exists silently caps
        # recall at less than 1.0 and looks like a retrieval bug for weeks.
        dangling = {
            f"{i.qid} -> {nid}"
            for i in items
            for nid in i.relevant_note_ids
            if nid not in known_note_ids
        }
        if dangling:
            raise ValueError(
                f"{path}: gold labels reference unknown notes: {sorted(dangling)}"
            )
    return items
