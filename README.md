# Signalment

A retrieval system over a structured corpus of veterinary notes, built for
**graduate veterinary students** — DVM-level and above, not pre-vet. Built
**evaluation harness first**. Every architectural claim in this repository is a
number measured against a gold set, and every number can be reproduced with one
command.

> **sig·nal·ment** *(n.)* — in veterinary practice, the description of a patient
> that opens every clinical record: species, breed, age, sex. It is the first
> thing written down because it changes what everything after it means.

That is the name because it is also the architecture. Species is a hard
retrieval filter here, not a ranking signal, for the same reason a clinician
writes the signalment first: a canine answer to a feline question is not a
near miss, it is a different answer.

```bash
make install     # editable install with dev deps
make lint        # validate the corpus front-matter
make eval        # ablation table
make sweep       # abstention threshold curve
make stale       # notes overdue for review
make drift       # gold set vs. current corpus
make check       # lint + tests + drift, the pre-commit gate
```

## Setup

Requires Python 3.10+.

```bash
git clone https://github.com/parmram29/signalment.git
cd signalment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
make check                       # should print: 76 passed ... ok
```

Open the folder in VS Code and accept the recommended extensions. `.vscode/`
sets the interpreter to `.venv`, puts `src` on `PYTHONPATH`, and enables pytest
discovery, so tests are runnable from the Testing panel with no further config.

## Why it is built this way

The usual failure mode for a project like this is a chatbot with no way to tell
whether it works. Retrieval quality and answer behaviour are therefore measured
**separately** — that separation is what makes a bad answer debuggable at all
(retrieval miss vs. generation failure), and it is standard practice in the RAG
evaluation literature.

Two design decisions carry most of the weight:

**1. Species is a hard retrieval filter, not a ranking signal.** Veterinary
medicine's characteristic hazard is a canine answer surfacing for a feline
question. The two documents are semantically near-identical and can be
clinically opposite — a dog-labelled permethrin spot-on is routine in dogs and
causes seizures in cats. A soft similarity signal cannot be relied on to
separate them; dropping wrong-species candidates before ranking can.

**2. The system must be able to say "I don't know."** A quarter of the gold set
consists of plausible questions the corpus genuinely cannot answer. Abstention
is scored as a classification problem, and **both** sides are reported: F1 on
correctly declining, *and* the false-refusal rate on answerable questions. A
system that abstains on everything scores perfect abstention recall, so F1
alone would hide it.

## Current results

Seed corpus of 4 notes, 11 chunks, 12 gold questions (8 answerable, 4
unanswerable). These numbers are a **harness demonstration, not a result** — the
corpus is far too small to conclude anything. They are here to show the
measurement works end to end.

| Config       | Recall@5 | nDCG@5 | MRR   |
|--------------|----------|--------|-------|
| bm25         | 1.000    | 0.862  | 0.812 |
| bm25+species | 1.000    | **1.000** | **1.000** |

The species filter moves nDCG@5 from 0.862 to 1.000 without costing recall —
it demotes wrong-species notes that BM25 ranked above the correct one. That
recall cannot fall is asserted as a test, not assumed.

### Abstention is the unsolved part

`python -m signalment sweep` produces the honest picture:

| threshold | recall@5 | abstain F1 | false refusal | unsupported answers |
|-----------|----------|------------|---------------|---------------------|
| 0.0       | 1.000    | 0.000      | 0.000         | **1.000**           |
| 4.0       | 1.000    | 0.750      | 0.125         | 0.250               |
| 5.0       | 1.000    | 0.800      | 0.250         | **0.000**           |
| 6.0+      | 1.000    | 0.615      | 0.625         | 0.000               |

At threshold 0 the system answers **every** unanswerable question. The worst
offender is *"What dose of insulin should I give a diabetic cat?"* — it scores
higher than any other unanswerable query because it shares vocabulary
(`insulin`, `cat`) with adjacent endocrine notes, while the corpus contains no
dosing information whatsoever. That is precisely the failure a clinical
assistant must not have, and it is caught here by construction rather than
discovered by a user.

Eliminating it costs a 25% false-refusal rate. **Known limitation:** raw BM25
scores are not comparable across queries, so a single global threshold is a
blunt instrument. Score calibration is the next piece of work, not a detail —
see `docs/evaluation.md`.

## Repository layout

```
corpus/notes/          Atomic notes, YAML front-matter, one idea per file
eval/gold/             Gold question set (JSONL) — the project's ground truth
src/signalment/
  schema.py            Note schema + validation rules
  corpus.py            Loading and parent/child chunking
  retrieval/           BM25 baseline, species filter, RRF fusion, dense stub
  evaluation/          Metrics, gold-set loader, runner, ablation report
crates/signalment_bm25/    Optional Rust BM25 accelerator (Cargo)
benchmarks/            Backend timing harness
tests/                 76 tests, no network, no models
docs/                  Note schema and evaluation methodology
```

## Design notes

**Chunking.** Notes are split along the author's own Markdown headings, with
long sections windowed at 300 tokens with 50-token overlap. Child chunks are
what get matched; the parent note is what gets read. Small chunks to *find*,
large chunks to *understand*.

**Contextual headers, for free.** Every chunk is prefixed with
`[title | species | body system]` drawn from the note's front-matter. This is
the cheap half of contextual retrieval: the document-level context a bare chunk
loses is already in the metadata, so it costs no LLM call and cannot
hallucinate.

**No embeddings yet, on purpose.** `retrieval/dense.py` is a documented stub.
Leaderboard rank is a weak predictor of retrieval quality on a specific corpus,
so the embedding model is an ablation row to be decided against this gold set —
not a choice made up front and then defended.

**The corpus reports its own decay.** `make stale` lists notes overdue for
review, with dosage and protocol notes sorted first because clinical numbers age
worst. `make drift` catches the failure mode that would otherwise be invisible:
adding notes can make a previously *unanswerable* gold question answerable, at
which point the abstention metrics silently start measuring the wrong thing. The
threshold for "answers too strongly" is the median top score of the answerable
items, so it self-calibrates instead of decaying into a meaningless constant as
the corpus grows.

**Two BM25 implementations, one interface.** `src/signalment/retrieval/bm25.py` is
the reference and is never removed; `crates/signalment_bm25` is an optional Rust
accelerator selected at runtime by `retrieval/backends.py`. A clone with no Rust
toolchain runs everything — the differential suite simply skips. This is the
pattern `orjson` and `numpy` use: an accelerated path when available, a correct
path always. `tests/test_backend_equivalence.py` asserts identical rankings,
identical chunk attribution, and scores agreeing to 1e-9, so the Python file is
the specification the crate is written against.

**No fine-tuning.** For injecting knowledge, retrieval outperforms unsupervised
fine-tuning; fine-tuning teaches form, not facts. It is out of scope until a
measured *style* gap justifies it.

## The corpus is example data

The four notes shipped here are schema examples so the harness has something to
run against. None contains a numeric dose — seeded placeholder content must
never be mistakable for clinical guidance. This is a study aid built over one
student's own notes; it is not clinical decision support.

## Status

The eval harness, corpus schema, BM25 baseline, species filter and RRF fusion
are implemented and tested. Dense retrieval, reranking, generation with enforced
citations, and the API layer are not yet built — the harness exists so that each
of those can be justified by a measured delta when it lands.

See `docs/evaluation.md` for methodology and `docs/note-schema.md` for the
corpus contract.
