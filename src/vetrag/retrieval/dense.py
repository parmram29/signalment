"""Dense retriever.

Left unimplemented on purpose. The embedding model is an ablation row, not an
up-front decision: MTEB rank is a weak predictor of retrieval quality on a
specific corpus, so the choice should be made against the gold set once the
BM25 baseline is on the board.

To implement: encode `chunk.indexed_text` for every chunk with a
sentence-transformers model (BGE-base-en-v1.5 is a reasonable first row,
MedCPT a domain candidate), keep the vectors in a flat exact index, and return
Hits scored by cosine similarity. Exact search is correct at this corpus size;
quantised indexes cost recall for a speedup that is not the bottleneck.
"""

from __future__ import annotations

from .base import Hit, Query


class DenseRetriever:
    name = "dense"

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "DenseRetriever is a scaffold. Implement it once the BM25 baseline "
            "is measured, so the gain from dense retrieval is attributable."
        )

    def search(self, query: Query, k: int) -> list[Hit]:  # pragma: no cover
        raise NotImplementedError
