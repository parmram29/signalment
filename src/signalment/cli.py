"""Command line entry point: `python -m signalment <command>`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .corpus import chunk_corpus, load_corpus
from .evaluation.dataset import load_gold
from .evaluation.drift import check_drift, format_warnings
from .evaluation.report import markdown_table
from .evaluation.runner import evaluate
from .freshness import stale_notes
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

    warnings = check_drift(configs[-1], gold, k=args.k)
    if warnings:
        print(format_warnings(warnings), file=sys.stderr)
        print(file=sys.stderr)

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


def cmd_stale(args) -> int:
    """Report notes overdue for review."""
    notes, errors = load_corpus(args.corpus)
    for path, message in errors:
        print(f"{path}: {message}", file=sys.stderr)

    stale = stale_notes(notes, args.days)
    if not stale:
        print(f"all {len(notes)} note(s) reviewed within {args.days} days")
        return 0

    width = max(len(s.note.id) for s in stale)
    print(f"{len(stale)} of {len(notes)} note(s) unreviewed for over {args.days} days\n")
    for item in stale:
        mark = "!" if item.is_clinical else " "
        print(f"{mark} {item.note.id.ljust(width)}  {item.days:>5}d since {item.basis}"
              f"  {item.note.note_type:<12} {item.note.title}")
    print("\n! = dosage or protocol note; clinical numbers age worst.")
    return 1 if args.strict else 0


def cmd_drift(args) -> int:
    """Check the gold set against the current corpus."""
    notes, errors = load_corpus(args.corpus)
    if errors:
        print(f"refusing to check: {len(errors)} invalid note(s). Run `lint`.", file=sys.stderr)
        return 1

    gold = load_gold(args.gold, known_note_ids={n.id for n in notes})
    retriever = SpeciesFilter(BM25Retriever(chunk_corpus(notes)), notes)
    warnings = check_drift(retriever, gold, k=args.k, strong_score=args.strong)
    print(format_warnings(warnings))
    return 1 if warnings and args.strict else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="signalment")
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

    stale = sub.add_parser("stale", help="list notes overdue for review")
    stale.add_argument("--corpus", default=DEFAULT_CORPUS)
    stale.add_argument("--days", type=int, default=180,
                       help="review interval before a note counts as stale")
    stale.add_argument("--strict", action="store_true",
                       help="exit non-zero when stale notes exist (for CI)")
    stale.set_defaults(func=cmd_stale)

    drift = sub.add_parser("drift", help="check the gold set against the corpus")
    drift.add_argument("--corpus", default=DEFAULT_CORPUS)
    drift.add_argument("--gold", default=DEFAULT_GOLD)
    drift.add_argument("-k", type=int, default=5)
    drift.add_argument("--strong", type=float, default=None,
                       help="score above which an unanswerable item is suspicious "
                            "(default: median top score of answerable items)")
    drift.add_argument("--strict", action="store_true",
                       help="exit non-zero when drift is detected (for CI)")
    drift.set_defaults(func=cmd_drift)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
