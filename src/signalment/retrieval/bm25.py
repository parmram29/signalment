"""Okapi BM25 over child chunks. The lexical baseline every later row beats."""

from __future__ import annotations

import math
from collections import Counter

from ..corpus import Chunk
from .base import Hit, Query, dedup_by_note, tokenize


class BM25Retriever:
    """Pure-Python BM25. No index build step, no dependencies, no GPU.

    At this corpus size an exact scan is milliseconds, so approximate indexes
    would trade recall for a speedup that does not matter here.
    """

    name = "bm25"

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.docs = [tokenize(chunk.indexed_text) for chunk in chunks]
        self.tf = [Counter(doc) for doc in self.docs]
        self.lengths = [len(doc) for doc in self.docs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0

        df: Counter[str] = Counter()
        for doc in self.docs:
            df.update(set(doc))
        n = len(self.docs)
        # Robertson/Sparck-Jones idf with the +0.5 smoothing, floored at zero so
        # a term present in every document contributes nothing rather than
        # subtracting score from documents that contain it.
        self.idf = {
            term: max(math.log(1 + (n - freq + 0.5) / (freq + 0.5)), 0.0)
            for term, freq in df.items()
        }

    def search(self, query: Query, k: int) -> list[Hit]:
        terms = tokenize(query.text)
        hits: list[Hit] = []
        for i, chunk in enumerate(self.chunks):
            score = 0.0
            length = self.lengths[i]
            for term in terms:
                freq = self.tf[i].get(term)
                if not freq:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1))
                score += self.idf.get(term, 0.0) * freq * (self.k1 + 1) / denom
            if score > 0:
                hits.append(Hit(note_id=chunk.note_id, score=score, chunk_id=chunk.chunk_id))
        return dedup_by_note(hits, k)
