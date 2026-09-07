"""Metadata filters applied around a retriever."""

from __future__ import annotations

from ..schema import Note
from .base import Hit, Query
from .fusion import reciprocal_rank_fusion


class SpeciesFilter:
    """Hard pre-filter on species. A safety control, not a ranking signal.

    A soft signal lets a canine dosage outrank a feline one on semantic
    similarity alone. Dropping the candidate entirely makes that impossible.
    Notes tagged 'multi' are species-agnostic and always survive the filter.
    """

    def __init__(self, inner, notes: list[Note]) -> None:
        self.inner = inner
        self.name = f"{inner.name}+species"
        self.species_by_note = {note.id: set(note.species) for note in notes}

    def search(self, query: Query, k: int) -> list[Hit]:
        if not query.species:
            return self.inner.search(query, k)
        # Over-fetch before filtering, or the filter silently shortens results.
        candidates = self.inner.search(query, k * 5)
        allowed = {query.species, "multi"}
        kept = [
            hit for hit in candidates if self.species_by_note.get(hit.note_id, set()) & allowed
        ]
        return kept[:k]


class FusionRetriever:
    """Combine several retrievers with reciprocal rank fusion."""

    def __init__(
        self, retrievers: list, weights: list[float] | None = None, rrf_k: int = 60
    ) -> None:
        self._fuse = reciprocal_rank_fusion
        self.retrievers = retrievers
        self.weights = weights
        self.rrf_k = rrf_k
        self.name = "rrf(" + "+".join(r.name for r in retrievers) + ")"

    def search(self, query: Query, k: int) -> list[Hit]:
        rankings = [[h.note_id for h in r.search(query, k * 3)] for r in self.retrievers]
        fused = self._fuse(rankings, k=self.rrf_k, weights=self.weights)
        # RRF scores are rank-derived, so the fused score is positional only.
        return [Hit(note_id=nid, score=1.0 / (i + 1)) for i, nid in enumerate(fused[:k])]
