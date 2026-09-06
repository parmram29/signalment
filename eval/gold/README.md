# Gold evaluation set

JSONL, one question per line. This is the project's ground truth: every design
decision downstream is justified by moving a number measured against it.

## Fields

| field | required | meaning |
|---|---|---|
| `qid` | yes | stable unique id |
| `question` | yes | the query as a student would type it |
| `answerable` | yes | whether the corpus supports an answer |
| `relevant_note_ids` | if answerable | note ids that contain the answer |
| `species` | no | species filter applied at retrieval |
| `expected_answer` | no | reference answer, for generation scoring later |
| `rationale` | no | why this item is in the set — write it while you remember |

## Rules the loader enforces

- An answerable item must list at least one relevant note.
- An unanswerable item must list none.
- The set must contain unanswerable items. A gold set of only answerable
  questions cannot measure abstention, which is the behaviour that matters most
  in a clinical domain.
- Every referenced note id must exist. A dangling label silently caps recall
  below 1.0 and looks like a retrieval bug for weeks.

## Growing it

This seed set is 12 items and is far too small to draw conclusions from — it
exists to prove the harness runs. Target ~150–200 items, roughly half
unanswerable, before treating any table row as a result. Write items from real
questions you actually had while studying; questions invented to match notes you
already wrote will flatter the system.
