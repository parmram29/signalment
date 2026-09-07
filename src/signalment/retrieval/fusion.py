"""Rank fusion.

Lives in `retrieval/` because fusing ranked lists is a retrieval operation, not
a measurement of one. It was briefly in `evaluation/metrics.py`, which forced
`retrieval` to import from `evaluation` — a layering inversion that showed up
as a function-body import to dodge the resulting cycle.
"""

from __future__ import annotations

from collections.abc import Sequence


def _dedup(ranked: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in ranked:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[str]:
    """Fuse several ranked lists into one (Cormack et al., SIGIR 2009).

    Score is sum over lists of w / (k + rank). The constant k damps the
    influence of top ranks so a single confident-but-wrong ranker cannot
    dominate the fusion. This is how BM25 and dense results are combined.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must be the same length as rankings")

    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, doc_id in enumerate(_dedup(ranking), start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
    # Ties broken by id so the output is deterministic across runs.
    return sorted(scores, key=lambda d: (-scores[d], d))
