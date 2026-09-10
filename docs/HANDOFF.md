# Signalment — Engineering Handoff

**Purpose:** a cold-start briefing for an engineer or a language model that has
never seen this repository. Read this top to bottom and you should be able to
make a correct change without re-deriving the architecture.

| | |
|---|---|
| Commit described | `8b220e5` — *Fix lint failures pushed with the ask command* |
| Date of review | 2026-09-10 |
| Size | ~4,280 lines of Python + Rust across `src/`, `crates/`, `tests/`, `benchmarks/` |
| History | 23 commits |
| Status | Phases 1–2 complete; Phase 3 (real corpus + dense retrieval) not started |

Everything in §7 (**Verified state**) and §8 (**Findings**) was reproduced by
running the code at this commit, not inferred from reading it. Commands are
given so you can re-run them. Where a claim is *not* verified, it says so.

---

## 1. The one rule that overrides everything

**Never create, edit, generate, or delete files in `corpus/notes/` or
`eval/gold/`.** Also `corpus/quarantine/` and `eval/quarantine/`.

This is not style guidance. Those directories hold notes a human wrote and
questions a human asked, and their entire value is that provenance. Generated
content there is invisible to every check the project has — the linter passes,
the tests pass, the metrics *improve* — while every reported number silently
stops meaning anything, because the system ends up measuring whether it can
retrieve text that was written in order to be retrieved.

This already happened, in commit `1e8df26`: 51 generated notes marked
`confidence: verified` carrying specific clinical thresholds and named journal
citations nobody had checked, plus 100 gold questions generated from those same
notes. Both sets are parked in `corpus/quarantine/` (53 files) and
`eval/quarantine/gold_generated.jsonl` (100 lines), unloaded by anything, with
`corpus/quarantine/README.md` documenting the per-note audit required to promote
any of it back.

If a task appears to require new notes or gold questions: **stop and say so.**
That work belongs to the repository owner. See `.clinerules`, `CLAUDE.md`,
`AGENTS.md`, `QWEN.md` (the last three are near-identical copies of the same
rules for different assistants).

> ⚠️ The mechanical enforcement of this rule is **not installed** in a fresh
> clone, and the documented command to install it does not exist. See
> finding **F1**.

---

## 2. What the project is, in one sentence

> Answers questions about *your* veterinary notes, cites which note the answer
> came from, and tells you when your notes don't cover it.

Audience is **graduate veterinary students (DVM-level and above)**, which is a
scope decision with consequences: clinical vocabulary is assumed, no glossary
layer, and abstention matters *more* because the reader is closer to acting on
what they read.

The name is the architecture. A *signalment* is the species/breed/age/sex line
that opens a veterinary record, written first because it changes the meaning of
everything after it. Here, **species is a hard retrieval filter, not a ranking
signal**, for the same reason.

### The thesis

Build the measurement layer before the thing being measured. The evaluation
harness is not scaffolding for the product — it *is* the product's
differentiator. The assistant is the demo that sits on top.

Concretely: no component enters the system without an ablation row showing a
measured delta attributable to that change alone. This is why
`retrieval/dense.py` is a documented stub rather than an implementation, and why
there are three BM25 implementations but only one of them is allowed to be
"the" one.

---

## 3. Repository map

Every file that matters, one line each.

### Data — never write to these

```
corpus/notes/              4 human-authored notes (.md + YAML front-matter)
  vet-endo-0142.md         Neonatal hypoglycaemia in kittens   (feline)
  vet-endo-0143.md         Insulinoma / exertional collapse    (canine)
  vet-tox-0007.md          Permethrin toxicosis in cats        (feline)
  vet-pharm-0021.md        Species differences in conjugation  (multi)
eval/gold/gold_v1.jsonl    12 gold questions: 8 answerable, 4 unanswerable
corpus/quarantine/notes/   53 quarantined generated notes (unloaded)
eval/quarantine/           100 quarantined generated questions (unloaded)
```

### Source — edit freely

```
src/signalment/
  cli.py         547 lines. Every command. argparse, one cmd_* per subcommand.
  schema.py      205. Note dataclass + front-matter validation rules.
  corpus.py      108. Front-matter parsing, corpus loading, parent/child chunking.
  ingest.py      280. PDF/txt extraction → drafts; the promote gate.
  freshness.py    59. Which notes are overdue for review.
  __main__.py      3. `python -m signalment`.

  retrieval/
    base.py       Tokenizer, Hit, Query, Retriever protocol, dedup_by_note.
    bm25.py       Pure-Python Okapi BM25. THE REFERENCE IMPLEMENTATION.
    fts.py        SQLite FTS5 backend — independent implementation, not a clone.
    dense.py      Bi-encoder + optional PCA. Built but unexercised (Phase 3).
    filters.py    SpeciesFilter (hard pre-filter) and FusionRetriever (RRF).
    fusion.py     Reciprocal rank fusion (Cormack et al., SIGIR 2009).
    backends.py   Runtime backend selection: auto → rust → python.

  evaluation/
    dataset.py    Gold-set JSONL loader + the guards that make it trustworthy.
    metrics.py    recall@k, precision@k, nDCG@k, MRR, abstention scores.
    runner.py     evaluate(): one pass over gold, retrieval + abstention.
    report.py     Renders the ablation table.
    drift.py      Detects gold labels that the corpus has outgrown.

crates/signalment_bm25/
  src/bm25.rs     Pure Rust BM25 + 18 unit tests. No PyO3 in this file.
  src/python.rs   PyO3 binding layer. Plumbing only, no algorithm.
  src/lib.rs      Module wiring; `python` feature is optional so the core
                  tests need no libpython.
```

### Workflow directories — gitignored, contents live only on the owner's machine

```
sources/     Lecture PDFs, GoodNotes exports. README.md tracked, contents not.
drafts/      Work in progress. Nothing indexes these. README.md tracked only.
```

Both are ignored by `.gitignore` (`sources/*`, `drafts/*`, plus blanket `*.pdf`
and `*.zip`). **Consequence for any agent working in a fresh clone or a cloud
container: these directories are empty and the owner's drafts are not
reachable.** That is deliberate, not a misconfiguration — `sources/README.md`
gives four reasons (nothing reads them, git keeps binaries forever, the
extractable text is the lecturer's slides rather than the student's words, and
redistribution of course material).

### Everything else

```
tests/           11 files, 147 test functions. No network, no model weights.
benchmarks/      bench_retrieval.py — times each backend, scales by replication.
docs/            BLUEPRINT.md (source of truth for scope), evaluation.md,
                 note-schema.md, agent-setup.md, and this file.
requirements/    dev.lock, ingest.lock, retrieval.lock (uv-compiled pins).
.githooks/       pre-commit — the enforcement for §1. See F1.
.github/workflows/ci.yml   Three jobs: python-only, rust-core, equivalence.
Makefile         install/test/lint/ruff/eval/sweep/stale/drift/check/accel/
                 rust-test/bench. NOTE: no `hooks` target — see F1.
```

---

## 4. The data model

A note is Markdown with YAML front-matter, one atomic idea per file, in git.
Three properties each earn their place: **atomic** (a good atomic note is
already a good chunk, which sidesteps most of the chunking problem),
**standalone** (a retrieved fragment must make sense alone), **in git**
(versioning, diffs and provenance for free).

```yaml
---
id: vet-tox-0007          # vet-<topic>-<4 digits>, unique, stable citation target
title: Permethrin toxicosis in cats
species: [feline]         # HARD retrieval filter
system: toxicology
note_type: mechanism      # definition|mechanism|differential|protocol|dosage|case
confidence: verified      # draft|reviewed|verified
provenance: self-authored # self-authored|lecture|textbook|guideline|
                          # primary-literature|ai-drafted
authored: 2026-03-11
last_reviewed: 2026-08-02 # optional; drives the staleness report
source_refs: ["..."]      # optional
see_also: [vet-pharm-0021]
---
```

Validation lives in `schema.py::validate` and reports **every** error at once
rather than the first. Two rules are load-bearing rather than cosmetic:

- **A `dosage` or `protocol` note may not be `confidence: draft`.** Clinical
  numbers must not reach a student marked unreviewed.
- **`provenance: ai-drafted` may not be `confidence: verified`.** "Verified"
  means a human checked it against a source; a model cannot make that claim on
  its own behalf. This is the schema-level expression of §1.

Duplicate ids across the corpus are an error. An empty body is an error.

---

## 5. The three pipelines

### 5a. Source material → draft → corpus (the human loop)

```
sources/lecture.pdf
   │  signalment ingest ... [--split pages]
   ▼
drafts/vet-topic-0001.md      front-matter filled in, TODO heading,
   │                          source text in an HTML comment block
   │  ← the human writes the note ABOVE the source block, in their own words,
   │    then deletes the TODO heading and the whole source block
   ▼
signalment promote drafts/vet-topic-0001.md
   ▼
corpus/notes/vet-topic-0001.md
```

The clever part is `check_promotable` in `ingest.py`. `promote` refuses any file
that still contains the TODO marker or the source block — **and the parts you
must delete are exactly the parts that would prove you had not rewritten it.**
So the human step is preserved by construction rather than by good intentions.
It additionally refuses a note under 120 characters of prose, and one still
marked `confidence: draft`.

What the gate explicitly *cannot* check: whether the note is any good, or true.

`--split pages` writes one draft per PDF page, on the reasoning that a 27-page
lecture is not one idea and a single draft holding all of it fights the
atomic-note rule. Pages yielding under 40 characters (`MIN_CHARS_PER_PAGE`) are
skipped as diagrams or handwriting.

**On handwriting:** `sources/README.md` documents, with numbers, why OCR is not
the answer — PaddleOCR benchmarks ~97% on clean printed English but ~72.8% on
handwriting, and the words most likely to be wrong in a clinical note are the
drug names and the numbers. The recommended path is GoodNotes' own conversion,
which reads Apple Pencil stroke data rather than pixels (90–95% on neat
printing), saved as `.txt` and ingested. **That path is currently broken — see
finding F4.**

### 5b. Corpus → answer (the product surface)

```
question + optional species
   ▼
load_corpus()          parse + validate every corpus/notes/*.md
   ▼
chunk_corpus()         split along the author's own Markdown headings;
                       sections over 300 tokens windowed with 50-token overlap
                       every chunk prefixed "[title | species | system]"
   ▼
BM25Retriever          exact scan; no index build step, no dependencies
   ▼
SpeciesFilter          over-fetch k*5, then HARD DROP wrong-species notes
   ▼                   ('multi' always survives; no species given = no-op)
abstention gate        see 5c
   ▼
answer envelope        N notes, each with title, id, confidence, provenance,
                       score, excerpt; then "Sources: ..." and a warning
                       listing any source that is not `verified`
```

Chunking rationale: child chunks are what get *matched*, the parent note is what
gets *read* — small chunks to find, large chunks to understand. The
`[title | species | system]` prefix is described in the code as "the cheap half
of contextual retrieval": the document-level context a bare chunk loses is
already in the front-matter, so it costs no LLM call and cannot hallucinate.

**There is no language model anywhere in this pipeline yet.** `ask` retrieves and
cites; Phase 5 adds prose on top. The stated reason for that order: the prose is
the part that can lie, so ship the citations first.

### 5c. The abstention gate — the most important subsystem, and the weakest

Two *different* gates exist in the codebase, and they do not agree.

**In the eval harness** (`evaluation/runner.py`), abstention is absolute:

```python
abstained = top_score <= abstain_threshold
```

**In the product** (`cli.py::cmd_ask`), abstention is relative:

```python
margin = top / second_score          # inf when there is only one hit
declined = top <= abstain_threshold or margin < min_margin   # defaults 0.0, 1.5
```

The margin gate is the right idea — `docs/BLUEPRINT.md` §5 names per-query
normalisation as fix-path step 1, precisely because raw BM25 scores are not
comparable across queries. **But `evaluate()` cannot express it, so the rule the
product actually ships has never appeared in an ablation table.** In a project
whose thesis is "every architectural claim is a measured number," this is the
gap that matters most. See finding **F3**, which measures it.

Abstention is scored as binary classification, positive class = "correctly
declined an unanswerable question":

| metric | meaning |
|---|---|
| `f1` | headline quality |
| `false_refusal_rate` | refused a question the corpus *can* answer — the cost of over-caution |
| `unsupported_answer_rate` | answered with no support — **the number that matters clinically** |

F1 alone is not enough: a system that abstains on everything scores perfect
abstention recall. Both sides are always reported.

### 5d. The two self-monitoring commands

- **`signalment stale --days N`** (`freshness.py`) — notes unreviewed for longer
  than N days, most urgent first, with `dosage` and `protocol` notes sorted
  ahead of everything else at equal staleness because clinical numbers age
  worst. A note that has never been reviewed falls back to `authored`, the
  honest lower bound.

- **`signalment drift`** (`evaluation/drift.py`) — closes a trap that would
  otherwise be invisible: **adding notes can make a previously *unanswerable*
  gold question answerable.** Nothing crashes; the abstention metrics quietly
  start measuring the wrong thing and every number after that is wrong. The
  check flags unanswerable items that now retrieve as strongly as a typical
  answerable one. The threshold defaults to the **median top score of the
  answerable items**, so it self-calibrates instead of decaying into a
  meaningless constant as the corpus grows. It also flags answerable items whose
  labelled notes no longer match anything at all. `drift` runs automatically
  before every `eval`.

---

## 6. Three BM25 implementations, and why that is not redundancy

This is the most misunderstood part of the codebase. The three are held to
**different standards on purpose**.

| Implementation | Kind | Standard it is held to |
|---|---|---|
| `retrieval/bm25.py` | **The reference.** Never removed. | It *is* the specification. |
| `crates/signalment_bm25` | **Accelerator.** Optional, runtime-selected. | Identical ranking, identical chunk attribution, scores to 1e-9. A divergence is a bug in the crate. |
| `retrieval/fts.py` | **Independent implementation** (SQLite's authors, different tokenizer, `k1=1.2` vs our `1.5`). | *Competitiveness* on the gold set, not equality. Demanding identical ranking would be a category error. |

The third one earns its place cheaply: a third opinion from code written by
other people, agreeing on which notes answer a question, is the least expensive
evidence available that the reference implementation is not quietly wrong. It
needs no new dependency — FTS5 ships with CPython's SQLite.

A clone with no Rust toolchain runs everything; the differential suite simply
skips. CI's `python-only` job exists to prove exactly that.

The scoring formula, which `.clinerules` warns is easy to get wrong:

```
                              f(t,d) * (k1 + 1)
score(t,d) = idf(t) * ───────────────────────────────────────
                       f(t,d) + k1 * (1 - b + b * |d| / avgdl)

idf(t) = max( ln(1 + (N - df + 0.5) / (df + 0.5)), 0 )
```

The `1 +` inside the logarithm is load-bearing: without it, a term present in
every document scores `ln(1) = 0` and queries silently return nothing.

`bm25.rs` documents four ways to get the port subtly wrong — repeated query
terms must count more than once, the `1 +` idf form floored at zero, empty
index must not divide by zero (`avg_len or 1`), and deduplication keeps the
**maximum** chunk score per note, not the sum and not the first.

---

## 7. Verified current state

Reproduced at commit `8b220e5` on 2026-09-10.

```bash
pip install -r requirements/dev.lock
PYTHONPATH=src python3 -m pytest -q
# → 130 passed, 60 skipped in 0.39s
#   (60 skips = 55 differential tests needing the unbuilt Rust extension,
#    plus dense tests needing numpy/sentence-transformers)

python3 -m ruff check src tests benchmarks     # All checks passed!
python3 -m ruff format --check src tests benchmarks   # 33 files already formatted

cargo test --manifest-path crates/signalment_bm25/Cargo.toml --no-default-features
# → 18 passed; 0 failed
```

`PYTHONPATH=src python3 -m signalment eval`:

```
corpus: 4 notes -> 11 chunks
gold:   12 questions (8 answerable, 4 unanswerable)

| Config              | Recall@k | nDCG@k | MRR   | Abstain F1 | False refusal | Unsupported ans. |
|---------------------|----------|--------|-------|------------|---------------|------------------|
| bm25                | 1.000    | 0.862  | 0.812 | 0.000      | 0.000         | 1.000            |
| bm25+species        | 1.000    | 1.000  | 1.000 | 0.000      | 0.000         | 1.000            |
| bm25-sqlite         | 1.000    | 0.908  | 0.875 | 0.000      | 0.000         | 1.000            |
| bm25-sqlite+species | 1.000    | 1.000  | 1.000 | 0.000      | 0.000         | 1.000            |
```

The headline claim reproduces exactly: **the species filter moves nDCG@5 from
0.862 to 1.000 without costing recall.** It demotes wrong-species notes that
BM25 ranked above the correct one. Independently, SQLite FTS5 shows the same
pattern from 0.908 — a different implementation agreeing on the direction of the
effect.

`signalment sweep` reproduces the README's abstention curve exactly (threshold
0.0 → unsupported 1.000; 5.0 → unsupported 0.000 at 0.250 false refusal).

**Read all of this as a harness demonstration, not a result.** Four notes and
twelve questions cannot support a conclusion about retrieval quality, and the
repository says so in three places. Recall is 1.000 in every row because the
corpus is too small to miss anything.

---

## 8. Findings — ranked by consequence

Each finding gives the evidence, the location, and what fixing it involves.
F1–F5 were reproduced at this commit.

---

### F1 · The rule from §1 is not actually enforced — `make hooks` does not exist

**Severity: high.** This is the guard protecting the project's core invariant.

`CLAUDE.md` states: *"This is enforced, not requested … Install it with
`make hooks`."* `.githooks/pre-commit` repeats it: *"Install: make hooks"*.

```bash
$ make -n hooks
make: *** No rule to make target 'hooks'.  Stop.
$ git config --get core.hooksPath
(unset)
$ ls .git/hooks/pre-commit
ls: cannot access '.git/hooks/pre-commit': No such file or directory
```

The Makefile's targets are `install test lint ruff eval eval-dense sweep stale
drift accel rust-test bench check`. There is no `hooks`. `README.md` compounds
it by describing `make install` as *"pinned deps + editable package + git
hooks"*, while the target is a bare `pip install -e ".[dev]"` — no hooks, and
not the pins either (see F8).

So the hook script is well-written, thoroughly reasoned, committed to the
repository — and installed in exactly zero clones. Every protection in §1 is
currently advisory, which is the state the hook's own header comment says is
insufficient.

**Fix** (small, and the highest value-per-line change available):

```makefile
hooks:
	git config core.hooksPath .githooks
	@echo "pre-commit hook active; bypass your own note edits with SIGNALMENT_HUMAN_EDIT=1"
```

Then make `install` depend on it, so the README's claim becomes true. Add a test
or a CI step asserting `git config core.hooksPath` is set, so this cannot regress
silently. Note the hook is already correct in substance — it blocks writes to the
protected paths, blocks non-note files in the corpus, and runs the linter when
the toolchain is importable.

---

### F2 · RRF fusion structurally destroys the abstention gate

**Severity: high — blocks the Phase 3 exit criterion.**

`FusionRetriever.search` (`retrieval/filters.py`) discards the fused scores and
re-scores by position:

```python
return [Hit(note_id=nid, score=1.0 / (i + 1)) for i, nid in enumerate(fused[:k])]
```

So the top hit's score is **always exactly 1.0**, for every query, answerable or
not. The abstention gate compares that constant against a threshold. Measured:

```
fused hits: [('vet-tox-0007', 1.0, '')]

thr=0.0: F1=0.000 falseref=0.000 unsupported=1.000
thr=0.5: F1=0.000 falseref=0.000 unsupported=1.000
thr=1.0: F1=0.500 falseref=1.000 unsupported=0.000
thr=2.0: F1=0.500 falseref=1.000 unsupported=0.000
```

The gate is a step function at 1.0: **answer everything, or refuse everything.**
There is no operating point in between, and no threshold sweep can find one.

The moment Phase 3 lands dense retrieval, `make eval-dense` adds four fusion
rows to the ablation table and three of its seven columns are meaningless on
those rows — while looking perfectly plausible. That is the same class of silent
corruption `drift.py` exists to prevent.

Second defect in the same three lines: `Hit.chunk_id` is dropped (`''` above), so
`cmd_ask` can never print an excerpt for a fused result — `by_chunk.get('')`
returns `None` and the excerpt is silently skipped.

**Fix.** Return the actual RRF score instead of `1.0/(i+1)` (`fusion.py` already
computes it; it is thrown away at the boundary), and carry `chunk_id` through
from the contributing retriever's hit. Then decide explicitly how abstention is
thresholded on a rank-derived score — RRF scores have a bounded, corpus-
independent range, which is arguably *better* for a global threshold than raw
BM25, but it needs its own calibration and its own ablation row. Do not ship a
fusion row until this is settled.

---

### F3 · The product's abstention rule has never been measured — and the fix is available today

**Severity: high (thesis-level), effort: low.**

`cmd_ask` gates on a **margin** (`top / second`, default 1.5); `evaluate()` gates
on an **absolute threshold**. `evaluate()` has no parameter that can express the
margin rule, so the decision rule that actually ships has never appeared in an
ablation table. The project's central claim is that every architectural decision
is backed by a measured number; this one is not.

Measured here, over all 12 gold items, `bm25+species`, k=3:

| gate | F1 | false refusal | unsupported |
|---|---|---|---|
| margin ≥ 1.5 (**the shipped default**) | 0.667 | **0.000** | **0.500** |
| absolute ≥ 5.0 (best absolute-only) | 0.800 | 0.250 | 0.000 |
| **absolute ≥ 4.0 AND margin ≥ 1.5** | **0.889** | **0.125** | **0.000** |

Two things follow.

**(a) The shipped default answers half the unanswerable questions.** Per-query
behaviour at `T=0, M=1.5`:

```
q008 answerable=False top= 3.93 margin= 1.87 declined=False  <-- unsupported answer
q009 answerable=False top= 1.21 margin=  inf declined=False  <-- unsupported answer
q010 answerable=False top= 4.62 margin= 1.46 declined=True
q011 answerable=False top= 3.54 margin= 1.49 declined=True
```

**(b) The two gates fail in complementary ways, which is *why* the combination
dominates.** The margin gate catches the vocabulary-overlap case — q010 (*"What
dose of insulin should I give a diabetic cat?"*, the failure the blueprint
singles out) and q011 both match several endocrine notes about equally. It is
structurally blind to the opposite case: q009 (*equine influenza*) returns a
**single** hit, so `margin = inf` and the gate never fires — even though its top
score of 1.21 is trivially low. Only an absolute floor catches that. Conversely,
q002 also has `margin = inf` but is answerable, so "treat inf as a refusal" would
cost a false refusal. The absolute floor is the right instrument there.

Margin catches *many weak matches*; the absolute floor catches *one weak match*.
Neither subsumes the other — the same argument the codebase already makes for
fusing lexical and dense retrieval.

**Why this matters beyond tidiness:** the Phase 4 exit criterion is
`unsupported_answer_rate = 0` at `false_refusal_rate < 0.10`. The combined gate
reaches **0.000 at 0.125** — one query away — with no cross-encoder, no
calibration, and no new dependency. Phase 4's headline goal is nearly reachable
with a two-line change to `evaluate()`.

**Do not over-read this.** n = 12, of which 4 are unanswerable, so
`unsupported_answer_rate` moves in steps of 0.25 and `false_refusal_rate` in
steps of 0.125. These numbers select a *hypothesis worth testing on the Phase 3
corpus*, not an operating point to ship. Reading a threshold off the same 12
items you then report on is exactly the "threshold overfit" risk in the
blueprint's own risk register.

**Fix.** Add `min_margin` to `evaluate()` alongside `abstain_threshold`, have
`cmd_ask` and the runner share one gate function so they cannot drift apart
again, and extend `sweep` to a 2-D grid. Reproduce with:

```bash
PYTHONPATH=src python3 - <<'EOF'
from signalment.corpus import chunk_corpus, load_corpus
from signalment.evaluation.dataset import load_gold
from signalment.evaluation.metrics import abstention_scores
from signalment.retrieval.base import Query
from signalment.retrieval.bm25 import BM25Retriever
from signalment.retrieval.filters import SpeciesFilter
notes, _ = load_corpus("corpus/notes")
gold = load_gold("eval/gold/gold_v1.jsonl", {n.id for n in notes})
r = SpeciesFilter(BM25Retriever(chunk_corpus(notes)), notes)
rows = []
for item in gold:
    hits = r.search(Query(text=item.question, species=item.species), 3)
    top = hits[0].score if hits else 0.0
    m = (top / hits[1].score) if len(hits) > 1 and hits[1].score > 0 else float("inf")
    rows.append((item.answerable, top, m))
for T, M in [(0, 1.5), (5, 0.0), (4, 1.5)]:
    s = abstention_scores([a for a, _, _ in rows], [t <= T or m < M for _, t, m in rows])
    print(f"T={T} M={M}: F1={s['f1']:.3f} falseref={s['false_refusal_rate']:.3f} "
          f"unsup={s['unsupported_answer_rate']:.3f}")
EOF
```

---

### F4 · `ingest --split pages` crashes on `.txt` / `.md` input

**Severity: medium — it blocks the documented handwriting workflow.**

`extract_pages` short-circuits non-PDFs (`ingest.py:117`) via
`return [read_source(path)]`, and `read_source` constructs an `Extraction`
without `page_number`, which defaults to `None`. `_ingest_per_page` then formats
that `None` as a column:

```
TypeError: unsupported format string passed to NoneType.__format__
  cli.py:271  print(f"  {note_id}.md   page {page:>3}  {chars:>5} chars")
```

It fails **after** writing the draft, leaving a file on disk, a traceback, and a
non-zero exit. The same missing guard produces a second symptom:

```yaml
title: Handwriting converted (pNone)
```

This sits directly on the recommended path for handwriting: GoodNotes → `.txt` →
ingest. A user following `sources/README.md` hits it immediately.

**Fix.** In `_ingest_per_page`, handle `page_number is None` (either format it as
`-`/`1`, or route non-PDF input to the single-draft path since "pages" is
meaningless for a flat text file — the latter is probably more honest). Suppress
the `(pNone)` title suffix in the same branch. Add a regression test; note that
`tests/test_ingest.py` already has 28 tests and none covers `--split pages` on a
non-PDF.

---

### F5 · Rust and Python tokenizers diverge on non-ASCII input

**Severity: low in practice, but it violates a stated guarantee.**

`bm25.rs` specifies the tokenizer as: *"lowercase the input first (Unicode-aware,
like Python's `str.lower()`)"*. The implementation uses `is_ascii_alphanumeric()`
and `to_ascii_lowercase()`, treating every non-ASCII character as a separator.
Python lowercases Unicode-aware **first**, then applies `[a-z0-9][a-z0-9\-']*`.

For characters whose Unicode lowercase *is* ASCII, the two disagree:

| input | Python `tokenize` | Rust `tokenize` |
|---|---|---|
| `U+212A` KELVIN SIGN | `['k']` | `[]` |
| `U+0130` LATIN CAPITAL I WITH DOT | `['i']` | `[]` |
| `U+FF11` FULLWIDTH ONE | `[]` | `[]` ✓ |
| `cafÉ` | `['caf']` | `['caf']` ✓ |

(Verified by running the Python tokenizer directly and a temporary integration
test against the crate, since removed.)

`.clinerules` is unambiguous about the resolution: *"When they disagree, the Rust
is wrong."* The differential suite that is supposed to catch this
(`tests/test_backend_equivalence.py`, 55 assertions) cannot, because its fixtures
are all ASCII.

Practical impact on a veterinary corpus is near zero. The reason to fix it is
that the equivalence guarantee is a *stated* one and is the crate's entire
justification for existing.

**Fix.** Either make the Rust tokenizer Unicode-lowercase before classifying
(matching the documented spec), or amend the doc comment and `.clinerules` to
state the ASCII-only restriction as intentional. Then close the test gap the
same way the bug got in: add adversarial non-ASCII vectors to the differential
suite, or property-test the two tokenizers against each other over random
Unicode strings.

---

### F6 · `ingest --split pages` has no near-duplicate detection

**Severity: medium — it corrupts the metric that Phase 3 exists to produce.**

`_ingest_per_page` writes one draft per page unconditionally. Slide decks
routinely export build-up animations as consecutive near-identical pages. In a
real run reported by the owner, drafts 0004/0005 both extracted to exactly 416
characters and 0023/0024 to exactly 483 — the signature of duplicate pages.

Writing up both produces two near-identical corpus notes. That is not merely
wasted effort: a gold question then has two "correct" retrievals where there is
really one, inflating recall without any real gain in coverage. It is the same
family of silent metric corruption as the quarantined corpus.

**Fix.** Hash the normalised extracted text per page; when a page's text is
identical (or above a similarity threshold) to one already written, skip it and
report it alongside the existing "skipped N page(s)" line. Diagnostic for
existing drafts:

```bash
for f in drafts/*.md; do
  echo "$(sed -n '/SOURCE TEXT/,/^-->/p' "$f" | sed '1,2d;$d' | md5)  $f"
done | sort
```

---

### F7 · Documentation drift

Individually trivial, collectively corrosive in a repository whose pitch is
rigour. All verified at this commit.

| Claim | Location | Reality |
|---|---|---|
| *"EXPECTED TO FAIL until `bm25.rs` is implemented — the 17 tests there are the spec"* | `Makefile:44` | It is implemented. `cargo test` → **18 passed, 0 failed**. |
| RRF fusion lives in `evaluation/metrics.py` | `docs/BLUEPRINT.md` §2 component table | Moved to `retrieval/fusion.py`. `metrics.py` re-exports the name for compatibility, and `fusion.py`'s docstring explains the move (it was a layering inversion). |
| *"`make check` … should print: 158 passed"* | `README.md` Setup | A fresh clone with no Rust extension prints **130 passed, 60 skipped**. |
| *"152 tests"* / *"76 tests"* / *"60 tests"* | `README.md` layout, BLUEPRINT §9, §12 | Three different numbers; actual is 147 test functions, 190 collected. |
| *"`make install` — pinned deps + editable package + git hooks"* | `README.md` Setup | Installs neither the pins nor the hooks (F1, F8). |
| `.PHONY` list | `Makefile:1` | Omits `ruff`, `eval-dense`, and the `hooks` target from F1. |

The `rust-test` comment is the one worth fixing first — it tells a newcomer that
a green suite is expected to be red, which trains them to ignore it.

---

### F8 · `make install` contradicts the reproducibility claim (D14)

Decision **D14** in the blueprint pins dependencies *"because embeddings are
version-sensitive, so unpinned deps let a torch upgrade move recall@5 with no
code change — which makes the reproducibility claim false."* `requirements/`
holds three uv-compiled lock files for exactly this.

But `make install` is `pip install -e ".[dev]"`, which resolves the *ranges* in
`pyproject.toml`. Only CI installs from the locks
(`pip install -r requirements/dev.lock`). So the contributor and CI can run
different versions, which is the situation D14 exists to prevent.

**Fix.** `make install` → `pip install -r requirements/dev.lock && pip install -e . --no-deps`, matching CI exactly, and fold in the `hooks` target from F1.

---

### F9 · The ingest floor and the promote floor disagree

`looks_handwritten` drops pages under **40 chars** (`MIN_CHARS_PER_PAGE`);
`check_promotable` rejects notes under **120 chars of prose**. Pages between the
two thresholds scaffold drafts with essentially nothing to write from — section
title slides, mostly. Not strictly a contradiction (the human's own words could
exceed 120 chars from a 76-char slide), but it generates drafts destined for
`rm`. Consider a second, higher advisory threshold that still writes the draft
but marks it `thin`, or simply report the char count so the user can bulk-delete.

---

### F10 · The real bottleneck is data, and no agent can remove it

Not a defect — the honest summary of where the project stands.

Everything in Phases 1 and 2 is built, tested and reproducible. The corpus is
**4 notes**; the gold set is **12 questions**. The Phase 3 exit criterion needs
50–100 real notes and 100+ gold questions. Recall is 1.000 in every ablation row
because a 4-note corpus cannot miss.

Until the corpus grows, **every retrieval number in this repository is a harness
demonstration, not a result** — which the README, the blueprint and
`dense.py`'s docstring all say explicitly, and which is the correct posture.

The consequence for anyone planning work here: engineering effort has high
leverage on *correctness* (F1–F6) and near-zero leverage on the *metrics*. The
metrics move when the owner writes notes. An agent that "improves retrieval"
against a 12-item gold set is measuring noise, and — per §1 — an agent that
generates notes to make the corpus bigger destroys the project's only real asset.

---

## 9. Executable work queue

Ordered by value per unit of risk. Each item is scoped to be a single commit
with a stated acceptance criterion. Nothing here touches protected data.

| # | Task | Files | Acceptance criterion |
|---|---|---|---|
| 1 | Add the `hooks` target; make `install` depend on it and use the locks | `Makefile`, `README.md` | `make hooks` sets `core.hooksPath`; a staged write to `corpus/notes/` is blocked in a fresh clone; CI asserts the config |
| 2 | Fix `--split pages` on `.txt`/`.md` | `cli.py`, `ingest.py`, `tests/test_ingest.py` | `ingest x.txt --split pages` exits 0 with a sane title; regression test added |
| 3 | Unify the abstention gate | `runner.py`, `cli.py`, `metrics.py`, tests | One gate function used by both `ask` and `evaluate`; `evaluate(min_margin=...)` works; `sweep` emits a 2-D grid; the F3 table reproduces from `make sweep` |
| 4 | Fix RRF scores and chunk attribution | `filters.py`, `fusion.py`, `tests/test_retrieval.py` | Fused hits carry real RRF scores and a real `chunk_id`; a threshold sweep over a fusion row produces more than two distinct operating points |
| 5 | Near-duplicate page detection in ingest | `ingest.py`, `cli.py`, tests | Identical page text produces one draft; the skip is reported |
| 6 | Close the tokenizer equivalence gap | `bm25.rs` or the spec comment, `test_backend_equivalence.py` | Python and Rust agree on `U+212A`/`U+0130`, asserted by a test that fails before the fix |
| 7 | Documentation drift sweep | `Makefile`, `README.md`, `docs/BLUEPRINT.md` | Every count and path in F7 matches reality; propose the prose, let the owner approve it (`.clinerules`: docs are the owner's) |

**Do items 1–2 first.** Item 1 restores the project's only hard safety
mechanism; item 2 unblocks the owner's actual current task (writing up the One
Health lecture, including the handwritten pages the extractor skipped).

**Item 3 is the highest-signal piece of engineering available** — it converts an
unmeasured shipped heuristic into an ablation row, which is the project's whole
thesis, and it puts the Phase 4 exit criterion within sight without a
cross-encoder.

---

## 10. Conventions for working in this repository

```bash
make check      # lint + ruff + tests + drift — the pre-commit gate. Do not commit if it fails.
make rust-test  # additionally, for any Rust change
```

- **Scope discipline.** Do the task asked. Do not add dependencies, restructure
  directories, rename things, or "improve" adjacent files unasked. If something
  else needs doing, say so and let the owner decide.
- **Commit messages describe the diff.** A commit touching 56 corpus files is
  not "Implement BM25 tokenizer in Rust". Split changes that span unrelated
  areas.
- **`docs/` and `README.md`:** propose changes; the owner writes the prose.
- **`src/`, `crates/`, `tests/`, `benchmarks/`:** edit freely, within the scope
  asked.
- **The Python BM25 is the specification.** When Rust and Python disagree, the
  Rust is wrong — unless you can show the Python is wrong and fix both together
  with a test pinning the corrected behaviour.
- **A human editing their own notes** bypasses the hook deliberately:
  `SIGNALMENT_HUMAN_EDIT=1 git commit -m "..."`.

CI runs three jobs: `python-only` (proves the repo works with no Rust
toolchain), `rust-core` (fmt + clippy `-D warnings` + tests, no Python linkage),
and `equivalence` (builds the extension via `maturin build` — not `maturin
develop`, which needs an active venv the runner does not have — and asserts the
differential suite actually *runs* rather than skips).

---

## 11. Roadmap, compressed

Full version with exit criteria in `docs/BLUEPRINT.md` §9. Do not start a phase
until the previous gate passes.

| Phase | Content | Gate | State |
|---|---|---|---|
| 1 | Schema, linter, chunking, BM25, species filter, RRF, metrics, runner | `pytest` green; `eval` prints a table | ✅ |
| 2 | `stale`, `drift` | Adding a note that answers an unanswerable gold item raises a visible warning | ✅ |
| 3 | 50–100 real notes, 100+ gold questions, dense retrieval, RRF | 4-row table on ≥100 questions; dense or fusion beats BM25 on recall@5 | ⬜ **owner-blocked** |
| 4 | Cross-encoder rerank, per-query normalisation, recalibration | `unsupported_answer_rate` = 0 at `false_refusal_rate` < 0.10 | ⬜ (see F3) |
| 5 | Generator, prompt contract, citation precision, Hermes-vs-aligned + gate-disabled row | Citation precision measured; gate contribution isolated | ⬜ |
| 6 | FastAPI, UI, feedback to SQLite, Dockerfile | `docker run` → working assistant | ⬜ |
| 7 | External vet-MCQ benchmark, active learning, VetSCT *iff* measured need | — | ⬜ |

Non-goals are binding (`BLUEPRINT.md` §0): no general veterinary question
answerer, no clinical decision support, no fine-tuned domain model, no multi-user
SaaS, no RL-from-feedback, no multimodal.

---

## 12. What this review did not verify

Stated so the next reader knows the edges of the evidence.

- **Dense retrieval** (`retrieval/dense.py`) was never executed — no
  `sentence-transformers`, no `numpy` in the environment, and its 10 tests
  skipped. The PCA logic, the BGE query-prefix asymmetry and the over-fetch in
  `search` were read, not run.
- **The Rust ↔ Python differential suite** never ran; the extension was not
  built (`maturin` unavailable in this environment). All 55 differential
  assertions skipped. F5 was established from the Rust core tests plus a
  temporary integration test, not from the differential suite.
- **The Rust accelerator's performance claim** is unmeasured here.
  `benchmarks/bench_retrieval.py` was read, not run.
- **F6's duplicate-page finding** rests on identical character counts reported
  from the owner's machine, not on the files themselves — `drafts/` is
  gitignored and empty in any clone. Identical length is strong evidence of
  duplicate text, not proof.
- **Nothing in `corpus/notes/`, `eval/gold/` or either quarantine directory was
  modified, and no note or gold question was authored**, per §1.

---

*Generated by reading the repository at commit `8b220e5`; all reproduction
commands are given inline. Sections 7 and 8 are measurements, section 9 is a
proposal.*
