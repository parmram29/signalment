import json

import pytest

from signalment.corpus import chunk_corpus, load_corpus
from signalment.evaluation.dataset import load_gold
from signalment.evaluation.report import markdown_table
from signalment.evaluation.runner import evaluate
from signalment.retrieval.bm25 import BM25Retriever
from signalment.retrieval.filters import SpeciesFilter

GOLD_PATH = "eval/gold/gold_v1.jsonl"


@pytest.fixture(scope="module")
def notes():
    loaded, errors = load_corpus("corpus/notes")
    assert errors == []
    return loaded


class TestGoldLoading:
    def test_shipped_gold_set_loads(self, notes):
        gold = load_gold(GOLD_PATH, known_note_ids={n.id for n in notes})
        assert len(gold) >= 12

    def test_contains_unanswerable_items(self, notes):
        gold = load_gold(GOLD_PATH)
        assert sum(not g.answerable for g in gold) >= 4

    def test_rejects_gold_set_with_no_unanswerable_items(self, tmp_path):
        path = tmp_path / "g.jsonl"
        path.write_text(json.dumps(
            {"qid": "q1", "question": "?", "answerable": True, "relevant_note_ids": ["vet-a-0001"]}
        ))
        with pytest.raises(ValueError, match="Abstention cannot be measured"):
            load_gold(path)

    def test_rejects_dangling_note_reference(self, tmp_path):
        path = tmp_path / "g.jsonl"
        path.write_text(
            json.dumps({"qid": "q1", "question": "?", "answerable": True,
                        "relevant_note_ids": ["vet-ghost-0001"]}) + "\n"
            + json.dumps({"qid": "q2", "question": "?", "answerable": False})
        )
        with pytest.raises(ValueError, match="unknown notes"):
            load_gold(path, known_note_ids={"vet-real-0001"})

    def test_rejects_answerable_item_with_no_labels(self, tmp_path):
        path = tmp_path / "g.jsonl"
        path.write_text(json.dumps({"qid": "q1", "question": "?", "answerable": True}))
        with pytest.raises(ValueError, match="needs relevant_note_ids"):
            load_gold(path)

    def test_rejects_duplicate_qid(self, tmp_path):
        path = tmp_path / "g.jsonl"
        row = json.dumps({"qid": "q1", "question": "?", "answerable": False})
        path.write_text(row + "\n" + row)
        with pytest.raises(ValueError, match="duplicate qid"):
            load_gold(path)


class TestRunner:
    def test_end_to_end_run(self, notes):
        gold = load_gold(GOLD_PATH, known_note_ids={n.id for n in notes})
        retriever = BM25Retriever(chunk_corpus(notes))
        result = evaluate(retriever, gold, k=5)

        assert 0.0 <= result.retrieval["recall@5"] <= 1.0
        assert 0.0 <= result.abstention["f1"] <= 1.0
        assert len(result.per_query) == len(gold)

    def test_retrieval_metrics_exclude_unanswerable_items(self, notes):
        gold = load_gold(GOLD_PATH, known_note_ids={n.id for n in notes})
        result = evaluate(BM25Retriever(chunk_corpus(notes)), gold, k=5)
        # Recall is undefined without relevant docs, so those items must not be
        # averaged in — only the answerable ones count toward n_queries.
        assert result.retrieval["n_queries"] == sum(g.answerable for g in gold)

    def test_species_filter_does_not_lose_recall(self, notes):
        gold = load_gold(GOLD_PATH, known_note_ids={n.id for n in notes})
        chunks = chunk_corpus(notes)
        bm25 = BM25Retriever(chunks)
        base = evaluate(bm25, gold, k=5)
        filtered = evaluate(SpeciesFilter(bm25, notes), gold, k=5)
        # The filter removes wrong-species notes; it must never remove a
        # correctly-labelled one, so recall cannot fall.
        assert filtered.retrieval["recall@5"] >= base.retrieval["recall@5"]

    def test_high_threshold_forces_abstention(self, notes):
        gold = load_gold(GOLD_PATH, known_note_ids={n.id for n in notes})
        result = evaluate(BM25Retriever(chunk_corpus(notes)), gold, k=5,
                          abstain_threshold=1e9)
        assert result.abstention["false_refusal_rate"] == 1.0
        assert all(row["abstained"] for row in result.per_query)


class TestReport:
    def test_table_renders_one_row_per_config(self, notes):
        gold = load_gold(GOLD_PATH, known_note_ids={n.id for n in notes})
        chunks = chunk_corpus(notes)
        bm25 = BM25Retriever(chunks)
        results = [evaluate(r, gold, k=5) for r in (bm25, SpeciesFilter(bm25, notes))]
        table = markdown_table(results)
        assert table.count("\n") == len(results) + 1  # header + separator + rows
        assert "bm25" in table and "species" in table
