# Autonomous Vet — Engineering Blueprint

Version 1.0 · Source of truth for scope, architecture, and delivery gates.

---

## 0. Scope

**The promise, in one sentence:**

> Answers questions about *your* veterinary notes, cites which note the answer
> came from, and tells you when your notes don't cover it.

Everything in this document exists to deliver that sentence and nothing wider.

### Non-goals

Named explicitly, because an unbounded scope is what makes a project feel
unexecutable:

| Not building | Why |
|---|---|
| A general veterinary question answerer | Unbounded. Cannot be evaluated, so cannot be shown to work. |
| Clinical decision support | Different product, different liability, different bar. |
| A fine-tuned domain model | Retrieval beats unsupervised fine-tuning for knowledge injection; fine-tuning teaches form, not facts. |
| Multi-user cloud SaaS | 4 test users need a local app and a Dockerfile, not Kubernetes. |
| Self-learning / RL from feedback | Requires a working, measured baseline first. Phase 7 at the earliest. |
| Multimodal (radiographs, histology) | Whole separate problem. Not in v1. |

### Success criterion

The project is done, for portfolio purposes, when a stranger can clone the repo,
run one command, and see a table showing that each architectural decision
improved a measured number — and read a written analysis of where it still fails.

---

## 1. Thesis

**Build the measurement layer before the thing being measured.**

Two independent lines of evidence converge on this:

*Engineering.* Roughly 80% of RAG failures originate in the ingestion and
chunking layer rather than the model. Without separated retrieval and generation
metrics you cannot tell a retrieval miss from a generation failure, so you cannot
tell what to fix. Every tuning decision becomes taste.

*Hiring.* The 2026 portfolio analyses are blunt: "a chatbot that calls an API is
not an ML project. Without retrieval quality metrics, hallucination analysis, or
structured evaluation, it demonstrates API usage, not ML capability." The two
scarcest signals in candidate portfolios are agentic orchestration and
**responsible-AI documentation**.

The eval harness is therefore not scaffolding for the product. It *is* the
product's differentiator. The assistant is the demo that sits on top.

### Three load-bearing decisions

**1. Species is a hard pre-filter, not a ranking signal.**
Canine and feline documents on the same drug are semantically near-identical and
can be clinically opposite — a permethrin spot-on is routine in dogs and causes
seizures in cats. A similarity score cannot be trusted to separate them. Wrong
species candidates are dropped before ranking, so no ranking accident can surface
them. Measured effect on the seed set: nDCG@5 0.862 → 1.000, recall unchanged.

**2. Abstention is decided outside the model.**
If nothing clears the retrieval threshold, the system declines — before the
generator sees anything. Two consequences: abstention is measurable before a
generator exists, and the reason to decline is auditable evidence rather than a
model's self-assessed confidence. This is also what makes a low-refusal model
(e.g. Hermes) safe to use here: the gate is external to it.

**3. Every component change is an ablation row.**
No component enters the system without a measured delta attributable to it alone.
This is why `retrieval/dense.py` is a documented stub rather than an
implementation.

---

## 2. Architecture

```
question + species
        │
        ▼
  ┌─────────────────┐
  │ species filter  │  hard drop, over-fetch k*5 first
  └────────┬────────┘
           ▼
  ┌─────────────────┐   ┌─────────────────┐
  │ BM25 (lexical)  │   │ dense (semantic)│   [dense: Phase 3]
  └────────┬────────┘   └────────┬────────┘
           └────────┬────────────┘
                    ▼
              RRF fusion            [Phase 3]
                    ▼
             cross-encoder rerank   [Phase 4]
                    ▼
        ┌───────────────────────┐
        │   ABSTENTION GATE     │  top score vs threshold
        └───┬───────────────┬───┘
    below   │               │  above
            ▼               ▼
   "not covered —      generator + enforced citations
    nearest notes X"            │
            └───────┬───────────┘
                    ▼
             answer envelope
```

### Component status

| Component | Module | Status |
|---|---|---|
| Note schema + linter | `schema.py` | Built |
| Corpus load, parent/child chunking | `corpus.py` | Built |
| BM25 baseline | `retrieval/bm25.py` | Built |
| Species hard filter | `retrieval/filters.py` | Built |
| RRF fusion | `evaluation/metrics.py` | Built |
| Retrieval metrics | `evaluation/metrics.py` | Built |
| Abstention scoring | `evaluation/metrics.py` | Built |
| Gold set loader + guards | `evaluation/dataset.py` | Built |
| Ablation runner + report | `evaluation/runner.py` | Built |
| Note staleness report | `freshness.py` | Built |
| Gold-set drift check | `evaluation/drift.py` | Built |
| Dense retriever | `retrieval/dense.py` | Phase 3 (stub) |
| Cross-encoder reranker | — | Phase 4 |
| Score calibration | — | Phase 4 |
| Generator + citation enforcement | — | Phase 5 |
| FastAPI + UI + Docker | — | Phase 6 |

---

## 3. The corpus contract

Markdown with YAML front-matter, one atomic idea per file, in git.

Three properties, each earning its place:
- **Atomic** — one idea per note. A well-written atomic note is already a good
  chunk, which sidesteps most of the chunking problem rather than solving it.
- **Standalone** — readable without its source. A retrieved fragment must make
  sense alone.
- **In git** — versioning, diffs and provenance for free, and the corpus itself
  becomes legible to anyone browsing the repo.

### Fields

| Field | Required | Role |
|---|---|---|
| `id` | yes | `vet-<topic>-<4 digits>`. Stable citation target. |
| `title` | yes | Human label; also enters the chunk context header. |
| `species` | yes | **Hard retrieval filter.** `multi` = species-agnostic, always survives. |
| `system` | yes | Body system. Context header + future filter. |
| `note_type` | yes | definition / mechanism / differential / protocol / dosage / case. Enables type-aware retrieval. |
| `confidence` | yes | draft / reviewed / verified. Trust axis for a self-authored corpus. |
| `provenance` | yes | Where the knowledge came from. |
| `authored` | yes | ISO date. |
| `last_reviewed` | no | Drives the staleness report. |
| `life_stage`, `source_refs`, `prerequisites`, `see_also` | no | Filtering, citation, graph links. |
| `tags_vetsct` | no | SNOMED CT Veterinary Extension ids. **Deferred** until measurement shows vocabulary mismatch is a real failure mode. |

### Enforced rules

- All validation errors reported at once, not one per run.
- Duplicate ids across the corpus are an error.
- Empty body is an error.
- **A `dosage` or `protocol` note may not be `confidence: draft`.** Clinical
  numbers do not reach a student marked unreviewed.

### Chunking

Split along the author's own Markdown headings; sections longer than 300 tokens
are windowed with 50-token overlap. Child chunks are matched, parent notes are
read — small to *find*, large to *understand*. This is the dominant production
pattern for 2025–26.

Every chunk is prefixed `[title | species | system]` from front-matter. This is
the cheap half of contextual retrieval: the document-level context a bare chunk
loses is already in the metadata, so it costs no LLM call and cannot hallucinate.

---

## 4. Retrieval design

| Stage | Rationale | Evidence |
|---|---|---|
| BM25 baseline | Vet queries are dense in exact rare tokens — drug names, dosages, breeds, abbreviations. Dense embeddings are weakest exactly there. | Standard IR |
| Species hard filter | See §1. | Measured: nDCG 0.862 → 1.000 |
| Dense retrieval | Catches paraphrase and vocabulary mismatch that BM25 misses. Model chosen *by ablation*, not by leaderboard — MTEB rank is a weak predictor of quality on a specific corpus. | MTEB caveat literature |
| RRF fusion | Combining lexical + dense beats either alone across metrics. | Cormack et al. SIGIR 2009 |
| Cross-encoder rerank | The single highest-ROI addition to a RAG stack. | +17.2pp MRR@3, +12.1pp Recall@5 over unreranked hybrid |

Filters over-fetch `k*5` before filtering, or the filter silently shortens the
result list instead of pulling replacements from deeper.

**Index choice:** flat exact search. At this corpus size retrieval is
milliseconds; quantised indexes trade recall for a speedup that is not the
bottleneck. Rebuild from scratch on every launch — no incremental indexing
complexity to build or debug.

---

## 5. The abstention gate

The most important subsystem, and currently the weakest.

### Mechanism

```python
abstained = top_retrieval_score <= threshold
```

Decided before generation, from evidence. Not a model disposition.

### Scoring

Binary classification, positive class = "correctly declined an unanswerable
question":

- `f1` — headline quality.
- `false_refusal_rate` — abstentions on answerable questions. **Always read with
  f1.** A system that refuses everything has perfect abstention recall.
- `unsupported_answer_rate` — unanswerable questions answered anyway. In a
  clinical domain, the number that matters most.

### Current measured state

| threshold | recall@5 | abstain F1 | false refusal | unsupported |
|---|---|---|---|---|
| 0.0 | 1.000 | 0.000 | 0.000 | **1.000** |
| 4.0 | 1.000 | 0.750 | 0.125 | 0.250 |
| 5.0 | 1.000 | 0.800 | 0.250 | **0.000** |
| 6.0+ | 1.000 | 0.615 | 0.625 | 0.000 |

At threshold 0 the system answers **every** unanswerable question. Worst case:
*"What dose of insulin should I give a diabetic cat?"* scores 4.62 — higher than
any other unanswerable query — because it shares `insulin`/`cat` with adjacent
endocrine notes while the corpus holds no dosing content at all.

### Known limitation and the fix path

Raw BM25 scores are not comparable across queries; they scale with query length,
term rarity and document length. A global threshold is therefore blunt. In order:

1. **Per-query normalisation** — margin between rank 1 and rank 2, or top score
   over mean of top k. Query-relative, needs no new model.
2. **Calibrate on a held-out split** — not on the same curve used to report.
3. **Rerank first** — a cross-encoder produces a genuinely comparable relevance
   score. Expect this to make thresholding work properly; measure rather than
   assume.

### The answer envelope

Every response has the same shape, so it cannot disappoint by surprise:

1. The answer
2. Source note(s) by title and id
3. Review state (`confidence`) of each source
4. Or, when gated: *"Not covered by your notes. Nearest: X, Y."*

---

## 6. Evaluation methodology

### Separation

Retrieval and generation are scored separately. A single end-to-end accuracy
number cannot tell you what to fix.

Retrieval metrics average over **answerable items only** — recall is undefined
without relevant documents. The runner reports `n_queries` so the denominator is
never ambiguous.

| Metric | Why |
|---|---|
| `recall@k` | Hard ceiling on answer quality; anything not recalled cannot reach the generator. |
| `precision@k` | Denominator is `k`, so one lucky hit is not scored as perfect. |
| `ndcg@k` | Rank-sensitive; catches the right note retrieved but buried ("lost in the middle"). |
| `mrr` | Position of first correct hit. |

Set `k` to what you actually pass the generator.

### Gold set

Target ~150–200 items, roughly half unanswerable. Currently 12 (seed only).

Rules the loader enforces: answerable items need labels, unanswerable items must
have none, the set must contain unanswerable items, and every referenced note id
must exist — a dangling label silently caps recall and reads as a retrieval bug
for weeks.

**Write items from questions you actually had while studying.** Questions
invented by reading notes you already wrote inherit their vocabulary and will
flatter the retriever.

### The gold-set drift trap

Adding notes can make a previously-unanswerable question answerable. Nothing
crashes — the abstention metric quietly starts measuring the wrong thing and
every number after that is wrong.

Mitigation (Phase 2): on every eval run, flag unanswerable items where retrieval
now returns a strong hit, and require confirm-or-relabel.

### Ablation protocol

One change per row. Two changes at once makes the row uninterpretable.

```
bm25                              ← baseline           [done]
bm25 + species filter                                  [done]
+ dense
+ rrf(bm25, dense)
+ cross-encoder rerank
+ calibrated abstention
```

### External benchmark

Veterinary undergraduate final-exam MCQs give an objective, domain-specific
metric alongside the private gold set — top LLMs report ~90% on a published
250-question comparison. The claim worth earning: *"local 4B + retrieval reaches
X% where the same model without retrieval reaches Y%."*

---

## 7. Generation layer

### Model choice

The generator is the last component and is deliberately swappable. Candidate
comparison, run as its own ablation table:

| Model | Size | License | Note |
|---|---|---|---|
| Hermes-4-14B | 14B | Apache-2.0 | Low-refusal by design; best license story |
| Llama-3.1-8B-Instruct | 8B | Llama 3.1 | Conventional aligned baseline |
| Qwen (current gen) | 7–8B | Apache-2.0 | Alternative baseline |

**The Hermes question is worth measuring, not assuming.** Hermes is tuned for
neutral alignment and low refusal — an advantage here, because generally-aligned
models over-refuse legitimate clinical questions from a student querying their
own notes. But a model tuned against saying "I don't know" is the most likely of
any class to answer the insulin-dose question from adjacent chunks. Because the
gate is external, this is testable:

| Generator | Citation precision | Faithfulness | Unsupported answer rate |
|---|---|---|---|
| Hermes-4-14B | | | |
| aligned baseline, same size | | | |
| **same, gate disabled** | | | |

The third row isolates how much safety comes from the gate versus the model's
disposition. If the gate does the work, using a low-refusal model in a clinical
context becomes a defensible claim backed by numbers.

### Prompt contract

- Answer **only** from provided context.
- Every claim carries a note id.
- **No dose, drug or numeric value may be emitted without a citation.** Never
  from parametric memory.
- If context is insufficient, say so — do not fill the gap.

### Latency budget

An 8B model at Q4 on laptop CPU runs roughly 3–10 tok/s. A 300-token cited
answer is 30–100 seconds. The "1–2 second" target from the original roadmap is
not achievable locally and must be resolved explicitly:

- (a) drop to 3–4B and accept ~10–20s,
- (b) accept slow local inference and stream,
- (c) hosted API for generation, retrieval stays local.

Retrieval is milliseconds either way; ~100% of latency is generation.

### On chain-of-thought

Reasoning traces are **not** explanations. Models systematically misrepresent the
actual basis for an answer in their stated reasoning (Turpin et al. 2023). The
explainability story is **citations to retrieved notes**, not "the model showed
its work." Hermes's hybrid `<think>` mode also costs latency — run it off on the
assistant path.

---

## 8. Safety and scope controls

| Control | Mechanism |
|---|---|
| Species confusion | Hard pre-filter, before ranking |
| Unsupported clinical numbers | No number without a citation; parametric memory forbidden |
| Overreach on out-of-scope questions | Abstention gate + measured `unsupported_answer_rate` |
| Unreviewed clinical content | Linter blocks `draft` dosage/protocol notes |
| User expectation | Scope sentence displayed in the UI |
| Provenance | Every answer cites note id + review state; corpus in git |

**Positioning:** a study aid over one student's own notes. Not clinical decision
support. Stated in the README, the UI, and the scope sentence.

**Licensing:** "the data is public" is a claim worth checking. Vet textbooks and
lecture PDFs are typically copyrighted and not licensed for redistribution.
Personal local use differs from serving 4 users and from a cloud deployment.
Self-authored notes avoid this entirely — another reason the corpus is
self-authored notes rather than scraped PDFs.

---

## 9. Delivery phases

Each phase has an **exit criterion** — a measurable gate, not a time estimate.
Do not start the next phase until the current one's gate passes.

### Phase 1 — Harness ✅ COMPLETE
Schema, linter, chunking, BM25, species filter, RRF, metrics, gold loader,
runner, report.
**Exit:** `pytest` green, `vetrag eval` prints an ablation table. *(76 tests)*

### Phase 2 — Freshness and drift ✅ COMPLETE
`vetrag stale --days N`; `vetrag drift`, also run automatically before every
`vetrag eval` so a corrupted gold set is never read as a clean result.
**Exit:** adding a note that answers a previously-unanswerable gold item causes
a visible warning rather than silent metric corruption. *(Verified: adding a
cruciate-ligament note flags gold item q008, "top 24.71 >= 5.77 median".)*

### Phase 3 — Vertical slice corpus + dense retrieval
Pick **one** body system or one course. 50–100 real notes. Grow gold set to
100+. Implement dense retrieval and RRF fusion.
**Exit:** the ablation table has 4 rows on ≥100 gold questions, and dense or
fusion beats BM25 on recall@5.

### Phase 4 — Reranking and calibration
Cross-encoder rerank; per-query score normalisation; recalibrated threshold.
**Exit:** `unsupported_answer_rate` = 0 at a `false_refusal_rate` below 0.10.

### Phase 5 — Generation with enforced citations
Generator interface; prompt contract; citation precision and faithfulness
metrics; the Hermes-vs-aligned comparison including the gate-disabled row.
**Exit:** citation precision measured and reported; gate-disabled row shows the
gate's contribution.

### Phase 6 — Ship it
FastAPI, thin UI showing the answer envelope, feedback logged to SQLite,
Dockerfile, README with the real tables.
**Exit:** `docker run` → working assistant. Cold clone → one command → table.

### Phase 7 — Optional, only if 1–6 are green
External vet-MCQ benchmark; active learning from logged feedback; VetSCT tagging
*if and only if* the gold set shows vocabulary mismatch is a real failure mode.

---

## 10. Risk register

| Risk | Early signal | Mitigation |
|---|---|---|
| Corpus work stalls the project | Weeks pass, note count flat | Vertical slice: one system, 50–100 notes. Completeness is not the goal; measurement is. |
| Gold set flatters the retriever | Recall near 1.0 on everything from day one | Write questions from real study sessions, not by reading your own notes |
| Gold-set drift | Abstention metrics improve for no reason | Phase 2 drift check |
| Threshold overfit | Great numbers, bad live behaviour | Held-out split for calibration |
| Latency disappoints users | Testers stop using it | Resolve the latency decision in Phase 5; stream tokens |
| Scope creep back to the 17-section plan | New tools appearing in requirements | Non-goals table in §0 is binding |
| Portfolio reads as "chatbot" | — | Lead with the ablation table and failure analysis, not the demo |

---

## 11. Learning track

What to actually understand at each phase, rather than copy.

| Phase | Concepts worth genuinely learning |
|---|---|
| 1 | IR metrics from first principles: why recall is the ceiling, why nDCG discounts logarithmically, why precision@k uses k as denominator. Implement them once — you have. |
| 2 | Data-versioning discipline; why silent metric corruption is worse than a crash. |
| 3 | BM25's term-frequency saturation and length normalisation (what `k1` and `b` actually do). Bi-encoder vs cross-encoder. Why RRF uses rank not score. |
| 4 | Score calibration; why cross-encoder scores are comparable and BM25's are not. |
| 5 | Prompt contracts as interfaces; faithfulness vs. plausibility; why CoT is not an explanation. |
| 6 | Serving, containerisation, feedback instrumentation. |

Read the code you have before adding more. `metrics.py` is ~150 lines and
contains the whole of classical IR evaluation for this project.

---

## 12. Recruiter-facing artifacts

Ranked by signal:

1. **The ablation table** in the README — proof each decision was measured.
2. **The failure analysis** — the abstention limitation, written up honestly with
   a diagnosis and a fix path. This is the scarce "responsible-AI documentation"
   signal.
3. **The test suite** — 60 tests, no network, no models. Proves the harness is
   real.
4. **The two-minute pitch:** *"Vet students ask questions their notes can answer
   and questions they can't. I built the retrieval system and the measurement
   for both. Species is a hard filter because canine and feline drug documents
   are semantically identical and clinically opposite. Abstention is decided
   outside the model, so I can use a low-refusal model safely — and I measured
   how much of the safety comes from the gate versus the model."*
5. The demo.

If the project cannot be summarised in two minutes — problem, build, result — it
will not survive a technical screen.

---

## Appendix — Decision log

| # | Decision | Rationale | What would reverse it |
|---|---|---|---|
| D1 | Eval harness before retrieval | Unfalsifiable tuning otherwise | Nothing |
| D2 | No fine-tuning | RAG beats unsupervised FT for knowledge injection | A measured *style/format* gap, not a facts gap |
| D3 | Species as hard filter | Semantically identical, clinically opposite | Measured recall loss from over-filtering |
| D4 | Abstention outside the model | Auditable; measurable pre-generator | Nothing |
| D5 | Dense retrieval deferred | Must be an attributable ablation row | Phase 3 |
| D6 | Flat index, rebuild each launch | ms-scale retrieval; quantisation costs recall | Corpus > ~10⁵ chunks |
| D7 | VetSCT tagging deferred | Expensive manual work on a hunch | Gold set shows vocabulary mismatch as a real failure mode |
| D8 | No LangChain | Fewer moving parts to debug when retrieval misbehaves | Multi-step agentic orchestration becomes a requirement |
| D9 | Self-authored corpus | Avoids the redistribution question entirely | Nothing |
| D10 | Generator swappable, chosen by measurement | The Hermes question is empirical | Nothing |
