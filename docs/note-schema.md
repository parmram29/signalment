# Note schema

The contract between the corpus and everything downstream. Enforced by
`src/vetrag/schema.py`; run `python -m vetrag lint` to check.

Metadata is attached **at ingestion**, never afterwards: retrieval filters and
citation provenance both read from it, and a note without it cannot be filtered
or cited correctly.

## Format

Markdown with YAML front-matter, one atomic idea per file, kept in git. Git
gives versioning, diffs and provenance for free, and a well-written atomic note
is already a good chunk — which sidesteps much of the chunking problem instead
of trying to solve it.

```markdown
---
id: vet-endo-0142            # vet-<topic>-<4 digits>, unique
title: Neonatal hypoglycaemia in kittens
species: [feline]            # HARD retrieval filter — see below
life_stage: [neonate]
system: endocrine
note_type: mechanism         # definition|mechanism|differential|protocol|dosage|case
confidence: verified         # draft|reviewed|verified
provenance: self-authored    # self-authored|lecture|textbook|guideline|primary-literature
authored: 2026-03-14
last_reviewed: 2026-08-02
source_refs: ["Ettinger 8th ed, ch.XX"]
prerequisites: [vet-physio-0031]
see_also: [vet-endo-0143]
tags_vetsct: []              # optional SNOMED CT Veterinary Extension ids
---

## Heading

Body. Headings become child-chunk boundaries, so write them as real sections.
```

## Field notes

**`species`** is the one field that is a safety control rather than metadata.
It is applied as a hard pre-filter before ranking. `multi` marks a
species-agnostic note, which always survives the filter. Rationale: canine and
feline documents on the same drug are semantically near-identical and can be
clinically opposite, and a soft similarity signal cannot be trusted to separate
them.

**`confidence`** is the trust axis for a self-authored corpus. Your notes are
reliable to you, but a medical assistant needs to rank and filter by review
state. The linter enforces one rule with teeth: a `dosage` or `protocol` note
may not be `draft`. Clinical numbers do not reach a student marked unreviewed.

**`note_type`** enables type-aware retrieval later — a "what is" question wants
a `definition`, a "how do I" question wants a `protocol`.

**`tags_vetsct`** is optional and should stay empty until measurement justifies
it. Veterinary medicine does have a real standard — the SNOMED CT Veterinary
Extension, maintained at Virginia Tech — and tagging enables synonym expansion
(student types "low blood sugar", note says "hypoglycaemia"). But it is slow
manual work. Add it once the gold set shows vocabulary mismatch is an actual
failure mode, not before.

## Validation rules

- All errors are reported at once, not one per run.
- Duplicate ids across the corpus are an error.
- An empty body is an error.
- `README.md` and files beginning with `_` are skipped, not linted as notes.

## Writing good notes

Atomic (one idea), standalone (readable without its source), linked (via
`see_also` and `prerequisites`). This is the evergreen-note discipline, and it
happens to be exactly what a retrieval chunk wants.
