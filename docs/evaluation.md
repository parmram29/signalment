# Evaluation methodology

## Principle

Retrieval and generation are scored separately. A single end-to-end accuracy
number cannot tell you whether a wrong answer came from a retrieval miss or a
generation failure, so it cannot tell you what to fix.

## Retrieval metrics

Averaged over **answerable items only** — recall is undefined for a question
with no relevant documents, so unanswerable items are excluded from these
averages and counted separately. The runner reports `n_queries` so the
denominator is never ambiguous.

| metric | why it is here |
|---|---|
| `recall@k` | The most actionable metric. Anything not recalled cannot reach the generator, so it is a hard ceiling on answer quality. |
| `precision@k` | Denominator is `k`, not the number of results returned, so a retriever that returns one lucky hit is not scored as perfect. |
| `ndcg@k` | Rank-sensitive. Catches the case where the right note is retrieved but buried — which matters because of "lost in the middle" effects in long contexts. |
| `mrr` | Position of the first correct hit. Complements recall. |

Set `k` to what you actually pass to the generator. Reporting `ndcg@10` while
feeding the model 5 chunks measures something the system never sees.

## Abstention metrics

Abstention is decided from the retrieval score: if nothing clears the
threshold, the system declines. Two consequences, both deliberate:

1. Abstention is measurable **before a generator exists**.
2. The reason to decline is "no supporting evidence", which is auditable, not a
   model's self-assessed confidence, which is not.

Scored as binary classification with "correctly declined an unanswerable
question" as the positive class:

- `f1` — headline abstention quality.
- `false_refusal_rate` — abstentions on answerable questions. **Always read
  with f1.** A system that refuses everything has perfect abstention recall.
- `unsupported_answer_rate` — unanswerable questions that got an answer anyway.
  In a clinical domain this is the number that matters most.

## Known limitation: score calibration

Raw BM25 scores are not comparable across queries. They scale with query
length, term rarity and document length, so a threshold tuned on one query
distribution does not transfer. The sweep in the README shows the cost
directly: driving `unsupported_answer_rate` to zero costs a 25% false-refusal
rate on a 12-item set.

Candidate fixes, in the order they should be tried and measured:

1. **Normalise per query** — score margin between rank 1 and rank 2, or the
   ratio of top score to the mean of the top k. Both are query-relative and
   need no new model.
2. **Calibrate on the gold set** — fit the threshold on a held-out split rather
   than reading it off the same curve used to report results.
3. **Rerank first** — a cross-encoder produces a genuinely comparable relevance
   score, unlike BM25. Expect this to make thresholding work properly, and
   measure it rather than assuming.

## Growing the gold set

The shipped set is 12 items: enough to prove the harness runs, far too small to
support a conclusion. Target ~150–200, roughly half unanswerable.

Write items from questions you actually had while studying. Questions invented
by reading notes you already wrote will flatter the retriever, because they
inherit its vocabulary. Record a `rationale` for each item while you still
remember why you added it.

## Running an ablation

Each config in `cli.cmd_eval` should isolate exactly one change from the row
above it, so any difference in the table is attributable to that change. Adding
two things at once makes the row uninterpretable.

Planned rows, each to be added only after the previous is measured:

```
bm25                          <- baseline, done
bm25 + species filter         <- done
dense
rrf(bm25, dense)
rrf(bm25, dense) + reranker
+ calibrated abstention
```
