"""Dense retrieval and PCA.

Split by what can be verified without a model. `fit_pca` is pure numpy and is
tested properly. The retriever itself needs model weights, so those tests skip
when sentence-transformers or the weights are unavailable — the same discipline
as the Rust extension: absent optional dependency means skip, never fail.
"""

from __future__ import annotations

import pytest

from signalment.retrieval.dense import dense_available, fit_pca

np = pytest.importorskip("numpy", reason="numpy not installed")


class TestPCA:
    @pytest.fixture
    def matrix(self):
        rng = np.random.default_rng(0)
        return rng.normal(size=(50, 32)).astype(np.float32)

    def test_shapes(self, matrix):
        mean, components = fit_pca(matrix, 8)
        assert mean.shape == (32,)
        assert components.shape == (8, 32)

    def test_components_are_orthonormal(self, matrix):
        _, components = fit_pca(matrix, 8)
        assert np.allclose(components @ components.T, np.eye(8), atol=1e-5)

    def test_components_ordered_by_variance(self, matrix):
        # The defining property: component 1 captures the most variance. If this
        # ordering were wrong, truncating to n components would discard signal
        # instead of noise, which is the whole point of the reduction.
        mean, components = fit_pca(matrix, 8)
        variances = ((matrix - mean) @ components.T).var(axis=0)
        assert np.all(np.diff(variances) <= 1e-6)

    def test_rejects_more_components_than_rank(self, matrix):
        # A small corpus cannot support a large projection, and failing loudly
        # beats silently returning a degenerate basis.
        with pytest.raises(ValueError, match="exceeds the rank"):
            fit_pca(matrix, 100)

    def test_rejects_zero_components(self, matrix):
        with pytest.raises(ValueError, match="at least 1"):
            fit_pca(matrix, 0)

    def test_reconstruction_improves_with_more_components(self, matrix):
        # Sanity that the projection is real: keeping more directions must lose
        # less of the original signal.
        def residual(n):
            mean, comps = fit_pca(matrix, n)
            centred = matrix - mean
            return float(np.linalg.norm(centred - (centred @ comps.T) @ comps))

        assert residual(16) < residual(4)


@pytest.mark.skipif(not dense_available(), reason="sentence-transformers not installed")
class TestDenseRetriever:
    """Requires model weights, so these skip in any sandbox without them."""

    @pytest.fixture(scope="class")
    def retriever(self):
        from signalment.corpus import chunk_corpus, load_corpus
        from signalment.retrieval.dense import DenseRetriever

        notes, errors = load_corpus("corpus/notes")
        assert errors == []
        try:
            return DenseRetriever(chunk_corpus(notes), model_name="all-MiniLM-L6-v2")
        except Exception as exc:
            pytest.skip(f"model unavailable: {type(exc).__name__}")

    def test_finds_semantically_related_note(self, retriever):
        from signalment.retrieval.base import Query

        # No lexical overlap with the note's wording: this is what dense buys.
        hits = retriever.search(Query("flea treatment poisoning in a cat"), 3)
        assert "vet-tox-0007" in {h.note_id for h in hits}

    def test_empty_query_returns_nothing(self, retriever):
        from signalment.retrieval.base import Query

        assert retriever.search(Query(""), 5) == []

    def test_deduplicated_and_bounded(self, retriever):
        from signalment.retrieval.base import Query

        hits = retriever.search(Query("hypoglycaemia in young animals"), 2)
        assert len(hits) <= 2
        assert len({h.note_id for h in hits}) == len(hits)

    def test_scores_are_cosine_bounded(self, retriever):
        from signalment.retrieval.base import Query

        for hit in retriever.search(Query("insulin secreting tumour"), 5):
            assert 0.0 < hit.score <= 1.0 + 1e-6
