"""Dense bi-encoder retrieval, with optional PCA projection.

A bi-encoder embeds queries and chunks independently into one vector space and
scores by cosine similarity. That independence is the point: chunk vectors are
computed once at index time, so query cost is one forward pass plus a dot
product, unlike a cross-encoder which must run the model per (query, chunk)
pair. Cross-encoders are more accurate and belong at the rerank stage, over the
few dozen candidates this retriever returns.

**What this adds over BM25.** Lexical retrieval cannot match "cat seizure"
against "feline epileptiform activity" — no term overlaps. Dense retrieval can,
because the two phrases land near each other in embedding space. The reverse is
also true and is why BM25 stays: a dense model handles an exact rare token like
a drug name or a dose worse than a lexical index does. Neither subsumes the
other, which is the entire argument for fusing them.

**On the numbers this produces.** Retrieval quality here is only meaningful
once the corpus is real. On a handful of notes every retriever scores near
1.000 and the comparison says nothing. Build it now, believe it after Phase 3.

The model is an ablation row, not a decision made up front: leaderboard rank is
a weak predictor of retrieval quality on a specific corpus, so the default is a
starting point to be beaten, not a recommendation.
"""

from __future__ import annotations

from ..corpus import Chunk
from .base import Hit, Query, dedup_by_note

#: General-purpose starting point. Alternatives worth measuring on a real
#: corpus: BAAI/bge-large-en-v1.5 (bigger, slower), and the biomedical models
#: (MedCPT, BioLORD) whose in-domain pretraining may or may not beat a stronger
#: general model — that is exactly the sort of question the gold set settles.
DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"

#: BGE models are trained with an instruction prefix on the *query* side only.
#: Omitting it measurably degrades retrieval, and applying it to documents as
#: well degrades it differently, so it is asymmetric on purpose.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def dense_available() -> bool:
    """Whether sentence-transformers and numpy can be imported."""
    try:
        import numpy  # noqa: F401
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


def fit_pca(matrix, n_components: int):
    """Fit a PCA projection on row-vectors, returning (mean, components).

    Implemented with a plain SVD rather than scikit-learn: it is a dozen lines,
    it avoids a dependency the rest of the project does not need, and the
    mechanics stay visible.

    Reducing 768 dimensions to 128-256 shrinks the index and speeds scoring.
    Whether it *helps quality* is genuinely uncertain — dropping low-variance
    directions can remove noise or can remove the fine distinctions that
    separate two similar clinical notes. It is a row in the table, not an
    optimisation to apply blind.
    """
    import numpy as np

    if n_components < 1:
        raise ValueError("n_components must be at least 1")
    n_samples, n_features = matrix.shape
    limit = min(n_samples, n_features)
    if n_components > limit:
        raise ValueError(
            f"n_components={n_components} exceeds the rank available from "
            f"{n_samples} samples of {n_features} dimensions (max {limit}). "
            f"A small corpus cannot support a large projection."
        )

    mean = matrix.mean(axis=0)
    _, _, vt = np.linalg.svd(matrix - mean, full_matrices=False)
    return mean, vt[:n_components]


class DenseRetriever:
    """Bi-encoder retrieval over chunk embeddings.

    Vectors are L2-normalised, so a dot product *is* cosine similarity and the
    search is a single matrix multiply. Exact, not approximate: at this corpus
    size an ANN index would trade recall for a speedup that is not the
    bottleneck.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        model_name: str = DEFAULT_MODEL,
        pca_components: int | None = None,
        batch_size: int = 32,
        device: str | None = None,
        query_prefix: str | None = None,
    ) -> None:
        if not dense_available():
            raise RuntimeError(
                "dense retrieval needs sentence-transformers. Install with:\n"
                '  pip install -e ".[retrieval]"'
            )
        import numpy as np
        from sentence_transformers import SentenceTransformer

        self._np = np
        self.chunks = chunks
        self.model_name = model_name
        self.pca_components = pca_components
        # Only BGE models take the instruction prefix; applying it to a model
        # not trained with one hurts rather than helps.
        self.query_prefix = (
            query_prefix
            if query_prefix is not None
            else (BGE_QUERY_PREFIX if "bge" in model_name.lower() else "")
        )
        self.name = f"dense[{model_name.split('/')[-1]}]"
        if pca_components:
            self.name += f"+pca{pca_components}"

        self.model = SentenceTransformer(model_name, device=device)

        if chunks:
            embeddings = self.model.encode(
                [chunk.text for chunk in chunks],
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype(np.float32)
        else:
            embeddings = np.zeros(
                (0, self.model.get_sentence_embedding_dimension()), np.float32
            )

        self._pca = (
            fit_pca(embeddings, pca_components) if pca_components and len(chunks) else None
        )
        self._matrix = self._project(embeddings)

    def _project(self, embeddings):
        """Apply PCA if fitted, then L2-normalise so dot product is cosine."""
        np = self._np
        if self._pca is not None:
            mean, components = self._pca
            embeddings = (embeddings - mean) @ components.T
        # Normalising *after* projection matters: PCA does not preserve norms,
        # so normalising first would be undone by the projection.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-12)

    def search(self, query: Query, k: int) -> list[Hit]:
        if not query.text.strip() or not self.chunks:
            return []
        np = self._np
        vector = self.model.encode(
            [self.query_prefix + query.text], convert_to_numpy=True, show_progress_bar=False
        ).astype(np.float32)
        scores = (self._matrix @ self._project(vector)[0]).astype(float)

        # Over-fetch before collapsing to notes: several chunks of one note may
        # occupy the top-k, which would otherwise yield fewer than k notes.
        take = min(len(scores), max(k * 5, k))
        top = (
            np.argpartition(-scores, take - 1)[:take]
            if take < len(scores)
            else np.arange(len(scores))
        )

        hits = [
            Hit(
                note_id=self.chunks[i].note_id,
                score=float(scores[i]),
                chunk_id=self.chunks[i].chunk_id,
            )
            for i in top
            # Cosine runs [-1, 1]; a negative or zero score is not evidence.
            if scores[i] > 0
        ]
        return dedup_by_note(hits, k)

    def __len__(self) -> int:
        return len(self.chunks)
