"""Command line entry point: `python -m vetrag <command>`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .corpus import chunk_corpus, load_corpus
from .evaluation.dataset import load_gold
from .evaluation.report import markdown_table
from .evaluation.runner import evaluate
from .retrieval.bm25 import BM25Retriever
from .retrieval.filters import SpeciesFilter

DEFAULT_CORPUS = "corpus/notes"
DEFAULT_GOLD = "eval/gold/gold_v1.jsonl"


def cmd_lint(args) -> int:
    notes, errors = load_corpus(args.corpus)
    for path, message in errors:
        print(f"{path}: {message}", file=sys.stderr)
    print(f"{len(notes)} valid note(s), {len(errors)} invalid", file=sys.stderr)
    return 1 if errors else 0


def cmd_eval(args) -> int:
    notes, errors = load_corpus(args.corpus)
    if errors:
        print(f"refusing to evaluate: {len(errors)} invalid note(s). Run `lint`.", file=sys.stderr)
        return 1

    gold = load_gold(args.gold, known_note_ids={n.id for n in notes})
    chunks = chunk_corpus(notes)
    bm25 = BM25Retriever(chunks)

    # Each configuration isolates one change from the row above it, so any
    # difference in the table is attributable to that change alone.
    configs = [bm25, SpeciesFilter(bm25, notes)]

    results = [
        evaluate(r, gold, k=args.k, abstain_threshold=args.abstain_threshold)
        for r in configs
    ]

    print(f"corpus: {len(notes)} notes -> {len(chunks)} chunks")
    print(f"gold:   {len(gold)} questions "
          f"({sum(g.answerable for g in gold)} answerable, "
          f"{sum(not g.answerable for g in gold)} unanswerable)\n")
    print(markdown_table(results))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps([r.__dict__ for r in results], indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")
    return 0


def cmd_sweep(args) -> int:
    """Sweep the abstention threshold.

    Raw BM25 scores are not comparable across queries, so a single global
    threshold is a blunt instrument. The sweep makes that visible rather than
    hiding it behind a default: read off the operating point you want and
    treat score calibration as the next piece of work.
    """
    notes, errors = load_corpus(args.corpus)
    if errors:
        print(f"refusing to evaluate: {len(errors)} invalid note(s). Run `lint`.", file=sys.stderr)
        return 1

    gold = load_gold(args.gold, known_note_ids={n.id for n in notes})
    retriever = SpeciesFilter(BM25Retriever(chunk_corpus(notes)), notes)

    header = f"| {'thr':>5} | {'recall@k':>8} | {'abstain F1':>10} | {'false refusal':>13} | {'unsupported':>11} |"
    print(header)
    print("|" + "|".join("-" * len(c) for c in header.split("|")[1:-1]) + "|")
    for threshold in args.thresholds:
        result = evaluate(retriever, gold, k=args.k, abstain_threshold=threshold)
        print(f"| {threshold:>5.1f} | {result.retrieval[f'recall@{args.k}']:>8.3f} "
              f"| {result.abstention['f1']:>10.3f} "
              f"| {result.abstention['false_refusal_rate']:>13.3f} "
              f"| {result.abstention['unsupported_answer_rate']:>11.3f} |")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vetrag")
    sub = parser.add_subparsers(dest="command", required=True)

    lint = sub.add_parser("lint", help="validate note front-matter")
    lint.add_argument("--corpus", default=DEFAULT_CORPUS)
    lint.set_defaults(func=cmd_lint)

    ev = sub.add_parser("eval", help="run the ablation over the gold set")
    ev.add_argument("--corpus", default=DEFAULT_CORPUS)
    ev.add_argument("--gold", default=DEFAULT_GOLD)
    ev.add_argument("-k", type=int, default=5, help="cutoff for retrieval metrics")
    ev.add_argument("--abstain-threshold", type=float, default=0.0,
                    help="decline when the top retrieval score does not exceed this")
    ev.add_argument("--out", help="write full results as JSON")
    ev.set_defaults(func=cmd_eval)

    sweep = sub.add_parser("sweep", help="sweep the abstention threshold")
    sweep.add_argument("--corpus", default=DEFAULT_CORPUS)
    sweep.add_argument("--gold", default=DEFAULT_GOLD)
    sweep.add_argument("-k", type=int, default=5)
    sweep.add_argument("--thresholds", type=float, nargs="+",
                       default=[0, 1, 2, 3, 4, 5, 6, 8, 10])
    sweep.set_defaults(func=cmd_sweep)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
