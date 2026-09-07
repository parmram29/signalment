"""Render run results as the ablation table."""

from __future__ import annotations

from .runner import RunResult

COLUMNS = [
    ("config", "Config", "{}"),
    ("recall", "Recall@k", "{:.3f}"),
    ("ndcg", "nDCG@k", "{:.3f}"),
    ("mrr", "MRR", "{:.3f}"),
    ("abstention_f1", "Abstain F1", "{:.3f}"),
    ("false_refusal", "False refusal", "{:.3f}"),
    ("unsupported", "Unsupported ans.", "{:.3f}"),
]


def _row(result: RunResult) -> dict:
    recall_key = next(k for k in result.retrieval if k.startswith("recall@"))
    ndcg_key = next(k for k in result.retrieval if k.startswith("ndcg@"))
    return {
        "config": result.config,
        "recall": result.retrieval[recall_key],
        "ndcg": result.retrieval[ndcg_key],
        "mrr": result.retrieval["mrr"],
        "abstention_f1": result.abstention["f1"],
        "false_refusal": result.abstention["false_refusal_rate"],
        "unsupported": result.abstention["unsupported_answer_rate"],
    }


def markdown_table(results: list[RunResult]) -> str:
    """The table that is the point of the project: every row an ablation."""
    rows = [_row(r) for r in results]
    widths = {
        key: max(len(header), *(len(fmt.format(row[key])) for row in rows))
        for key, header, fmt in COLUMNS
    }
    lines = [
        "| " + " | ".join(h.ljust(widths[k]) for k, h, _ in COLUMNS) + " |",
        "|" + "|".join("-" * (widths[k] + 2) for k, _, _ in COLUMNS) + "|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(fmt.format(row[k]).ljust(widths[k]) for k, _, fmt in COLUMNS)
            + " |"
        )
    return "\n".join(lines)
