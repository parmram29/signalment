"""Run a retriever over the gold set and score it."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..retrieval.base import Query
from . import metrics
from .dataset import GoldItem


@dataclass
class RunResult:
    config: str
    retrieval: dict[str, float]
    abstention: dict[str, float]
    per_query: list[dict] = field(default_factory=list)


def evaluate(
    retriever,
    gold: list[GoldItem],
    k: int = 5,
    abstain_threshold: float = 0.0,
) -> RunResult:
    """Score retrieval quality and abstention behaviour in one pass.

    Abstention is decided from the retrieval score alone: if nothing clears
    `abstain_threshold` the system declines. That makes abstention measurable
    before a generator exists, and keeps the decision auditable — the reason to
    decline is "no supporting evidence", not a model's mood.

    Retrieval metrics are averaged over answerable items only. Averaging in
    unanswerable items, which have no relevant documents, would be undefined.
    """
    recalls: list[float] = []
    precisions: list[float] = []
    ndcgs: list[float] = []
    rrs: list[float] = []
    answerable_flags: list[bool] = []
    abstained_flags: list[bool] = []
    per_query: list[dict] = []

    for item in gold:
        hits = retriever.search(Query(text=item.question, species=item.species), k)
        ranked = [h.note_id for h in hits]
        top_score = hits[0].score if hits else 0.0
        abstained = top_score <= abstain_threshold

        answerable_flags.append(item.answerable)
        abstained_flags.append(abstained)

        row = {
            "qid": item.qid,
            "answerable": item.answerable,
            "abstained": abstained,
            "top_score": round(top_score, 4),
            "retrieved": ranked,
            "relevant": list(item.relevant_note_ids),
        }
        if item.answerable:
            row["recall_at_k"] = metrics.recall_at_k(ranked, item.relevant_note_ids, k)
            row["ndcg_at_k"] = metrics.ndcg_at_k(ranked, item.gains, k)
            recalls.append(row["recall_at_k"])
            precisions.append(metrics.precision_at_k(ranked, item.relevant_note_ids, k))
            ndcgs.append(row["ndcg_at_k"])
            rrs.append(metrics.reciprocal_rank(ranked, item.relevant_note_ids))
        per_query.append(row)

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return RunResult(
        config=getattr(retriever, "name", retriever.__class__.__name__),
        retrieval={
            f"recall@{k}": mean(recalls),
            f"precision@{k}": mean(precisions),
            f"ndcg@{k}": mean(ndcgs),
            "mrr": mean(rrs),
            "n_queries": len(recalls),
        },
        abstention=dict(metrics.abstention_scores(answerable_flags, abstained_flags)),
        per_query=per_query,
    )
