# Source material

Not tracked by git. Put lecture PDFs, GoodNotes exports and archives here.

They stay out of the repository for four reasons:

1. **Nothing reads them.** `load_corpus` globs `corpus/notes/*.md`. A PDF or
   zip in the corpus is invisible to every retriever and every metric.
2. **Git keeps binaries forever.** An 11 MB archive stays in the pack after you
   delete it. On a repository whose point is to be cloned and read, that cost
   is permanent.
3. **The extractable text is usually not yours.** A GoodNotes export of an
   annotated lecture has a text layer containing *the slides* — the lecturer's
   words, under their name. Your handwriting is the part that does not extract.
4. **Redistribution.** Course slides in a public repository is publishing
   someone else's teaching material.

## Working from them

Read the source, then write the note in your own words in `corpus/notes/`.
That restatement is the whole value: it is what makes the corpus yours, what
makes `provenance: self-authored` true, and — not incidentally — it is the
Feynman step. Extracting the slide text would give you a corpus of someone
else's sentences that you have not learned from.

## Extracting text from these files

Two different problems live in one PDF, and they have different answers.

### The printed slide text

Extractable today. Of the open parsers, **Docling** (IBM) is the one to use
here: MIT licensed, runs on CPU, emits structured output built for RAG.
Marker is faster but ships GPL-3.0 code with a RAIL-M weight licence that
restricts commercial use above a revenue threshold; MinerU is stronger on
complex and CJK layouts than anything in this corpus needs.

But read what comes out before you plan around it. On these files the
extractable text is the lecturer's slides, under their name — which is why
this pipeline produces reference material to read, never notes to commit.

### Your handwriting

Not extractable by a parser, and open-source OCR is not close. PaddleOCR
benchmarks at 97.2% on clean printed English and **72.8% on handwriting**. At
that rate roughly one word in four is wrong, and the words most likely to be
wrong in a clinical note are the drug names and the numbers.

**Use GoodNotes' own conversion instead.** It reports 90-95% word accuracy on
neat printing and 80-90% on cursive, and it beats every open engine because it
reads the Apple Pencil *stroke data*, not pixels. The catch is the same fact
stated from the other side: it only converts ink drawn inside the app.
Imported PDFs and images are static content to it and cannot be converted.

### The honest summary

A parser turns a lecture PDF into text you can read while writing. It does not
turn it into your notes, because the part that is yours is the part that does
not extract. Convert your ink in GoodNotes, read the slides for context, then
write the note in your own words.
