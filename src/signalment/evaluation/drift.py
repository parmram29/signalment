"""Gold-set drift detection.

The trap this exists to close: adding notes can make a previously
*unanswerable* gold question answerable. Nothing crashes. The abstention
metrics quietly start measuring the wrong thing, and every number reported
after that is wrong.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from ..retrieval.base import Query
from .dataset import GoldItem

MAY_BE_ANSWERABLE = "may_be_answerable"
LABELS_UNREACHABLE = "labels_unreachable"


@dataclass(frozen=True)
class DriftWarning:
    qid: str
    kind: str
    detail: str
    top_score: float
    retrieved: tuple[str, ...] = ()


def _top_score(retriever, item: GoldItem, k: int) -> tuple[float, tuple[str, ...]]:
    hits = retriever.search(Query(text=item.question, species=item.species), k)
    return (hits[0].score if hits else 0.0, tuple(h.note_id for h in hits))


def check_drift(
    retriever,
    gold: list[GoldItem],
    k: int = 5,
    strong_score: float | None = None,
) -> list[DriftWarning]:
    """Flag gold items whose labels may no longer match the corpus.

    `strong_score` defaults to the median top score across answerable items,
    which makes the check self-calibrating. That matters because raw lexical
    scores are not comparable across corpora or query sets, so any hardcoded
    constant would silently stop meaning anything the moment the corpus grew.
    """
    scores = {item.qid: _top_score(retriever, item, k) for item in gold}

    answerable_tops = [
        scores[item.qid][0] for item in gold if item.answerable and scores[item.qid][0] > 0
    ]
    if strong_score is None:
        # With no answerable item scoring at all there is no scale to compare
        # against, so no unanswerable item can be judged suspicious.
        strong_score = statistics.median(answerable_tops) if answerable_tops else float("inf")

    warnings: list[DriftWarning] = []
    for item in gold:
        top, retrieved = scores[item.qid]

        if not item.answerable and top >= strong_score:
            warnings.append(
                DriftWarning(
                    qid=item.qid,
                    kind=MAY_BE_ANSWERABLE,
                    detail=(
                        f"labelled unanswerable but retrieves as strongly as a typical "
                        f"answerable question (top {top:.2f} >= {strong_score:.2f}). "
                        f"Confirm it is still unanswerable, or relabel it."
                    ),
                    top_score=top,
                    retrieved=retrieved,
                )
            )

        # Nothing in the corpus matches this query at all. For a question with
        # gold labels that means the labelled notes were deleted or rewritten,
        # not that retrieval merely ranked them poorly.
        elif item.answerable and top == 0.0:
            warnings.append(
                DriftWarning(
                    qid=item.qid,
                    kind=LABELS_UNREACHABLE,
                    detail=(
                        f"labelled answerable via {', '.join(item.relevant_note_ids)} but "
                        f"nothing in the corpus matches the query at all. The note may "
                        f"have been rewritten."
                    ),
                    top_score=top,
                )
            )
    return warnings


def format_warnings(warnings: list[DriftWarning]) -> str:
    if not warnings:
        return "gold set: no drift detected"
    lines = [f"gold set: {len(warnings)} drift warning(s)"]
    for w in warnings:
        lines.append(f"  [{w.kind}] {w.qid}: {w.detail}")
        if w.retrieved:
            lines.append(f"      retrieved: {', '.join(w.retrieved[:3])}")
    return "\n".join(lines)
