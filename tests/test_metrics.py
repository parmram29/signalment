import math

import pytest

from signalment.evaluation import metrics


class TestRecall:
    def test_all_relevant_in_top_k(self):
        assert metrics.recall_at_k(["a", "b", "c"], {"a", "b"}, 3) == 1.0

    def test_partial(self):
        assert metrics.recall_at_k(["a", "x", "y"], {"a", "b"}, 3) == 0.5

    def test_cutoff_excludes_late_hit(self):
        assert metrics.recall_at_k(["x", "y", "a"], {"a"}, 2) == 0.0

    def test_duplicates_do_not_inflate(self):
        # A repeated id must not count twice, nor consume two of the k slots
        # in a way that differs from the deduplicated ranking.
        assert metrics.recall_at_k(["a", "a", "b"], {"a", "b"}, 2) == 1.0

    def test_no_relevant_docs_is_an_error(self):
        with pytest.raises(ValueError):
            metrics.recall_at_k(["a"], set(), 5)


class TestPrecision:
    def test_denominator_is_k_not_result_count(self):
        # Only one result returned, k=5: precision is 1/5, not 1/1.
        assert metrics.precision_at_k(["a"], {"a"}, 5) == pytest.approx(0.2)

    def test_rejects_zero_k(self):
        with pytest.raises(ValueError):
            metrics.precision_at_k(["a"], {"a"}, 0)


class TestReciprocalRank:
    def test_first_position(self):
        assert metrics.reciprocal_rank(["a", "b"], {"a"}) == 1.0

    def test_third_position(self):
        assert metrics.reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)

    def test_miss(self):
        assert metrics.reciprocal_rank(["x", "y"], {"a"}) == 0.0


class TestNDCG:
    def test_perfect_ranking(self):
        assert metrics.ndcg_at_k(["a", "b"], {"a", "b"}, 2) == pytest.approx(1.0)

    def test_ideal_dcg_capped_at_k(self):
        # One relevant doc, k=5: a first-position hit is perfect, and is not
        # penalised for the four empty slots.
        assert metrics.ndcg_at_k(["a", "x", "y"], {"a"}, 5) == pytest.approx(1.0)

    def test_discount_matches_definition(self):
        # Relevant doc at rank 2: DCG = 1/log2(3), IDCG = 1/log2(2) = 1.
        assert metrics.ndcg_at_k(["x", "a"], {"a"}, 2) == pytest.approx(1 / math.log2(3))

    def test_graded_relevance(self):
        # The higher-gain document ranked first must beat the reverse ordering.
        graded = {"a": 3.0, "b": 1.0}
        assert metrics.ndcg_at_k(["a", "b"], graded, 2) > metrics.ndcg_at_k(
            ["b", "a"], graded, 2
        )


class TestReciprocalRankFusion:
    def test_agreement_wins(self):
        fused = metrics.reciprocal_rank_fusion([["a", "b"], ["a", "c"]])
        assert fused[0] == "a"

    def test_promotes_consensus_over_single_top_rank(self):
        # "b" is second in both lists; "a" is first in one and absent from the
        # other. Consensus should win — that is the point of RRF.
        fused = metrics.reciprocal_rank_fusion([["a", "b"], ["c", "b"]])
        assert fused[0] == "b"

    def test_weights_shift_the_outcome(self):
        rankings = [["a", "b"], ["b", "a"]]
        assert metrics.reciprocal_rank_fusion(rankings, weights=[5.0, 1.0])[0] == "a"
        assert metrics.reciprocal_rank_fusion(rankings, weights=[1.0, 5.0])[0] == "b"

    def test_deterministic_on_ties(self):
        rankings = [["b", "a"], ["a", "b"]]
        assert metrics.reciprocal_rank_fusion(rankings) == metrics.reciprocal_rank_fusion(
            rankings
        )

    def test_rejects_mismatched_weights(self):
        with pytest.raises(ValueError):
            metrics.reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])


class TestAbstention:
    def test_perfect_behaviour(self):
        scores = metrics.abstention_scores(
            answerable=[True, True, False, False],
            abstained=[False, False, True, True],
        )
        assert scores["f1"] == 1.0
        assert scores["false_refusal_rate"] == 0.0
        assert scores["unsupported_answer_rate"] == 0.0

    def test_always_abstaining_is_caught_by_false_refusal(self):
        # Recall is perfect and f1 looks respectable; false_refusal_rate is the
        # metric that exposes a system which simply refuses everything.
        scores = metrics.abstention_scores(
            answerable=[True, True, False, False],
            abstained=[True, True, True, True],
        )
        assert scores["recall"] == 1.0
        assert scores["false_refusal_rate"] == 1.0

    def test_never_abstaining(self):
        scores = metrics.abstention_scores(
            answerable=[True, False],
            abstained=[False, False],
        )
        assert scores["f1"] == 0.0
        assert scores["unsupported_answer_rate"] == 1.0

    def test_mismatched_lengths_rejected(self):
        with pytest.raises(ValueError):
            metrics.abstention_scores([True], [True, False])

    def test_empty_set_rejected(self):
        with pytest.raises(ValueError):
            metrics.abstention_scores([], [])
