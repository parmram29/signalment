"""Retrieval and abstention metrics.

Deliberately dependency-free so the harness can run and be tested before any
model, index, or GPU is involved. Retrieval quality and answer behaviour are
measured separately: that separation is what makes a bad answer debuggable
(retrieval miss vs. generation failure).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

# Re-exported: rank fusion is a retrieval operation and lives there now, but it
# is scored alongside these metrics often enough that the old name still works.
from ..retrieval.fusion import reciprocal_rank_fusion  # noqa: F401


def _dedup(ranked: Sequence[str]) -> list[str]:
    """Keep first occurrence only; a repeated id must not be counted twice."""
    seen: set[str] = set()
    out: list[str] = []
    for item in ranked:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def recall_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of relevant documents that appear in the top k.

    The most actionable retrieval metric for RAG: anything not recalled here
    cannot possibly be used by the generator.
    """
    rel = set(relevant)
    if not rel:
        raise ValueError("recall_at_k is undefined with no relevant documents")
    top = _dedup(ranked)[:k]
    return len(rel.intersection(top)) / len(rel)


def precision_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the top k that is relevant. Denominator is k, not len(top)."""
    if k <= 0:
        raise ValueError("k must be positive")
    rel = set(relevant)
    top = _dedup(ranked)[:k]
    return len(rel.intersection(top)) / k


def reciprocal_rank(ranked: Sequence[str], relevant: Iterable[str]) -> float:
    """1/rank of the first relevant hit, else 0.0. Mean over queries is MRR."""
    rel = set(relevant)
    for i, doc_id in enumerate(_dedup(ranked), start=1):
        if doc_id in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(
    ranked: Sequence[str],
    relevant: Iterable[str] | dict[str, float],
    k: int,
) -> float:
    """Normalised discounted cumulative gain over the top k.

    `relevant` may be a set of ids (binary gain) or an id -> gain mapping for
    graded judgements. Ideal DCG is computed from the k highest available
    gains, so a query with fewer relevant docs than k is not penalised.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    gains = (
        dict(relevant) if isinstance(relevant, dict) else {doc_id: 1.0 for doc_id in relevant}
    )
    if not gains:
        raise ValueError("ndcg_at_k is undefined with no relevant documents")

    top = _dedup(ranked)[:k]
    dcg = sum(
        gains.get(doc_id, 0.0) / math.log2(i + 1) for i, doc_id in enumerate(top, start=1)
    )
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(gain / math.log2(i + 1) for i, gain in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


class AbstentionScores(dict):
    """Abstention behaviour, scored as a binary classification problem.

    Positive class = "correctly declined to answer an unanswerable question".
    Reporting f1 alone is not enough: a system that abstains on everything
    scores perfect recall, so false_refusal_rate is reported alongside it.
    """

    __getattr__ = dict.__getitem__  # type: ignore[assignment]


def abstention_scores(
    answerable: Sequence[bool],
    abstained: Sequence[bool],
) -> AbstentionScores:
    """Score abstention over a gold set containing unanswerable questions.

    Args:
        answerable: whether each question is supported by the corpus.
        abstained: whether the system declined to answer it.
    """
    if len(answerable) != len(abstained):
        raise ValueError("answerable and abstained must be the same length")
    if not answerable:
        raise ValueError("cannot score an empty evaluation set")

    tp = sum(1 for a, d in zip(answerable, abstained, strict=True) if not a and d)
    fp = sum(1 for a, d in zip(answerable, abstained, strict=True) if a and d)
    fn = sum(1 for a, d in zip(answerable, abstained, strict=True) if not a and not d)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    n_answerable = sum(1 for a in answerable if a)
    n_unanswerable = len(answerable) - n_answerable

    return AbstentionScores(
        precision=precision,
        recall=recall,
        f1=f1,
        # Refusing a question the corpus can answer: the cost of over-caution.
        false_refusal_rate=fp / n_answerable if n_answerable else 0.0,
        # Answering with no support: the failure mode that matters clinically.
        unsupported_answer_rate=fn / n_unanswerable if n_unanswerable else 0.0,
        n_answerable=n_answerable,
        n_unanswerable=n_unanswerable,
    )
