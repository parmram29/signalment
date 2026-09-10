"""Command line entry point: `python -m signalment <command>`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .corpus import chunk_corpus, load_corpus, parse_note
from .evaluation.dataset import load_gold
from .evaluation.drift import check_drift, format_warnings
from .evaluation.report import markdown_table
from .evaluation.runner import evaluate
from .freshness import stale_notes
from .ingest import IngestError, check_promotable, draft_note, read_source, suggest_id
from .retrieval.backends import available_backends, make_retriever
from .retrieval.bm25 import BM25Retriever
from .retrieval.filters import FusionRetriever, SpeciesFilter
from .schema import ValidationError

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
        print(
            f"refusing to evaluate: {len(errors)} invalid note(s). Run `lint`.",
            file=sys.stderr,
        )
        return 1

    gold = load_gold(args.gold, known_note_ids={n.id for n in notes})
    chunks = chunk_corpus(notes)
    bm25 = BM25Retriever(chunks)

    # Each configuration isolates one change from the row above it, so any
    # difference in the table is attributable to that change alone.
    configs = [bm25, SpeciesFilter(bm25, notes)]

    # SQLite is an independent implementation rather than an accelerator, so it
    # appears as its own row for comparison, not as a substitute for the
    # reference. The Rust backend is deliberately absent: it is asserted
    # identical to the reference by the differential suite, so a row for it
    # would duplicate the one above.
    if "sqlite" in available_backends():
        sqlite = make_retriever(chunks, backend="sqlite")
        configs += [sqlite, SpeciesFilter(sqlite, notes)]

    # Dense is opt-in: constructing it downloads model weights and embeds the
    # whole corpus, which is not something `make eval` should do by default.
    if args.dense:
        if "dense" not in available_backends():
            print(
                'dense needs sentence-transformers: pip install -e ".[retrieval]"',
                file=sys.stderr,
            )
            return 1
        dense = make_retriever(chunks, backend="dense", pca_components=args.pca)
        # Lexical and dense fail in different directions — exact rare tokens
        # versus paraphrase — so fusing them is the row worth looking at, not
        # either alone. Fusing two lexical retrievers would only average noise.
        fused = FusionRetriever([bm25, dense])
        configs += [dense, SpeciesFilter(dense, notes), fused, SpeciesFilter(fused, notes)]

    results = [
        evaluate(r, gold, k=args.k, abstain_threshold=args.abstain_threshold) for r in configs
    ]

    warnings = check_drift(configs[-1], gold, k=args.k)
    if warnings:
        print(format_warnings(warnings), file=sys.stderr)
        print(file=sys.stderr)

    print(f"corpus: {len(notes)} notes -> {len(chunks)} chunks")
    print(
        f"gold:   {len(gold)} questions "
        f"({sum(g.answerable for g in gold)} answerable, "
        f"{sum(not g.answerable for g in gold)} unanswerable)\n"
    )
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
        print(
            f"refusing to evaluate: {len(errors)} invalid note(s). Run `lint`.",
            file=sys.stderr,
        )
        return 1

    gold = load_gold(args.gold, known_note_ids={n.id for n in notes})
    retriever = SpeciesFilter(BM25Retriever(chunk_corpus(notes)), notes)

    header = (
        f"| {'thr':>5} | {'recall@k':>8} | {'abstain F1':>10} "
        f"| {'false refusal':>13} | {'unsupported':>11} |"
    )
    print(header)
    print("|" + "|".join("-" * len(c) for c in header.split("|")[1:-1]) + "|")
    for threshold in args.thresholds:
        result = evaluate(retriever, gold, k=args.k, abstain_threshold=threshold)
        print(
            f"| {threshold:>5.1f} | {result.retrieval[f'recall@{args.k}']:>8.3f} "
            f"| {result.abstention['f1']:>10.3f} "
            f"| {result.abstention['false_refusal_rate']:>13.3f} "
            f"| {result.abstention['unsupported_answer_rate']:>11.3f} |"
        )
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
        print(
            f"{mark} {item.note.id.ljust(width)}  {item.days:>5}d since {item.basis}"
            f"  {item.note.note_type:<12} {item.note.title}"
        )
    print("\n! = dosage or protocol note; clinical numbers age worst.")
    return 1 if args.strict else 0


def cmd_drift(args) -> int:
    """Check the gold set against the current corpus."""
    notes, errors = load_corpus(args.corpus)
    if errors:
        print(
            f"refusing to check: {len(errors)} invalid note(s). Run `lint`.", file=sys.stderr
        )
        return 1

    gold = load_gold(args.gold, known_note_ids={n.id for n in notes})
    retriever = SpeciesFilter(BM25Retriever(chunk_corpus(notes)), notes)
    warnings = check_drift(retriever, gold, k=args.k, strong_score=args.strong)
    print(format_warnings(warnings))
    return 1 if warnings and args.strict else 0


def cmd_ingest(args) -> int:
    """Scaffold a note draft from a source file."""
    source = Path(args.source)
    try:
        extraction = read_source(source)
    except IngestError as exc:
        print(f"{exc}", file=sys.stderr)
        return 1

    if extraction.looks_handwritten:
        print(
            f"{source.name}: {extraction.chars_per_page:.0f} characters per page across "
            f"{extraction.pages} pages.\n"
            f"That is a picture of handwriting, not text — there is nothing to extract.\n"
            f"Convert the ink to text in your notes app (it reads pen strokes, not\n"
            f"pixels, so it is far more accurate than OCR), save as .txt, and ingest that.",
            file=sys.stderr,
        )
        return 1

    notes, _ = load_corpus(args.corpus)
    drafts = Path(args.drafts)
    existing = {n.id for n in notes} | {p.stem for p in drafts.glob("*.md")}
    note_id = args.id or suggest_id(args.topic, existing)
    if note_id in existing:
        print(f"id {note_id} already exists", file=sys.stderr)
        return 1

    try:
        content = draft_note(
            source=source,
            extraction=extraction,
            note_id=note_id,
            title=args.title or source.stem,
            species=args.species,
            system=args.system,
            note_type=args.note_type,
        )
    except IngestError as exc:
        print(f"{exc}", file=sys.stderr)
        return 1

    drafts.mkdir(parents=True, exist_ok=True)
    out = drafts / f"{note_id}.md"
    out.write_text(content, encoding="utf-8")

    print(f"wrote {out}")
    print(f"  {extraction.pages} pages, {len(extraction.text)} characters of source text")
    print()
    print("Next: open it, write the note above the source block, delete the block,")
    print(f"      set confidence, then: signalment promote {out}")
    return 0


def cmd_promote(args) -> int:
    """Move a finished draft into the corpus, if it is finished."""
    draft = Path(args.draft)
    if not draft.exists():
        print(f"no such draft: {draft}", file=sys.stderr)
        return 1

    text = draft.read_text(encoding="utf-8")
    problems = check_promotable(text)
    if problems:
        print(f"{draft.name} is not ready:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    # The schema is the real gate; this only catches it earlier and by name.
    try:
        note = parse_note(text, path=str(draft))
    except ValidationError as exc:
        print(f"{draft.name}: {exc}", file=sys.stderr)
        return 1

    destination = Path(args.corpus) / f"{note.id}.md"
    if destination.exists():
        print(f"{destination} already exists", file=sys.stderr)
        return 1

    destination.write_text(text, encoding="utf-8")
    draft.unlink()
    print(f"promoted {draft.name} -> {destination}")
    print()
    print("Commit it now. A note that exists only in your working tree is one")
    print("careless edit away from not existing:")
    print(f"  SIGNALMENT_HUMAN_EDIT=1 git add {destination} && \\")
    print(f'    SIGNALMENT_HUMAN_EDIT=1 git commit -m "Add note: {note.title}"')
    return 0


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
    ev.add_argument(
        "--abstain-threshold",
        type=float,
        default=0.0,
        help="decline when the top retrieval score does not exceed this",
    )
    ev.add_argument(
        "--dense",
        action="store_true",
        help="add dense and BM25+dense fusion rows (downloads model weights)",
    )
    ev.add_argument(
        "--pca",
        type=int,
        default=None,
        help="project dense embeddings to this many dimensions",
    )
    ev.add_argument("--out", help="write full results as JSON")
    ev.set_defaults(func=cmd_eval)

    sweep = sub.add_parser("sweep", help="sweep the abstention threshold")
    sweep.add_argument("--corpus", default=DEFAULT_CORPUS)
    sweep.add_argument("--gold", default=DEFAULT_GOLD)
    sweep.add_argument("-k", type=int, default=5)
    sweep.add_argument(
        "--thresholds", type=float, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 8, 10]
    )
    sweep.set_defaults(func=cmd_sweep)

    stale = sub.add_parser("stale", help="list notes overdue for review")
    stale.add_argument("--corpus", default=DEFAULT_CORPUS)
    stale.add_argument(
        "--days", type=int, default=180, help="review interval before a note counts as stale"
    )
    stale.add_argument(
        "--strict", action="store_true", help="exit non-zero when stale notes exist (for CI)"
    )
    stale.set_defaults(func=cmd_stale)

    drift = sub.add_parser("drift", help="check the gold set against the corpus")
    drift.add_argument("--corpus", default=DEFAULT_CORPUS)
    drift.add_argument("--gold", default=DEFAULT_GOLD)
    drift.add_argument("-k", type=int, default=5)
    drift.add_argument(
        "--strong",
        type=float,
        default=None,
        help="score above which an unanswerable item is suspicious "
        "(default: median top score of answerable items)",
    )
    drift.add_argument(
        "--strict", action="store_true", help="exit non-zero when drift is detected (for CI)"
    )
    drift.set_defaults(func=cmd_drift)

    ing = sub.add_parser("ingest", help="scaffold a note draft from a source file")
    ing.add_argument("source", help="a .pdf, .txt or .md file, usually under sources/")
    ing.add_argument("--topic", default="misc", help="short topic slug for the note id")
    ing.add_argument("--species", default="multi")
    ing.add_argument("--system", default="general")
    ing.add_argument("--note-type", default="mechanism", dest="note_type")
    ing.add_argument("--title", default=None)
    ing.add_argument("--id", default=None, help="override the generated note id")
    ing.add_argument("--corpus", default=DEFAULT_CORPUS)
    ing.add_argument("--drafts", default="drafts")
    ing.set_defaults(func=cmd_ingest)

    pro = sub.add_parser("promote", help="move a finished draft into the corpus")
    pro.add_argument("draft")
    pro.add_argument("--corpus", default=DEFAULT_CORPUS)
    pro.set_defaults(func=cmd_promote)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
