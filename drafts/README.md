# Drafts

Work in progress. Nothing here is indexed, searched or measured — `load_corpus`
reads `corpus/notes/` only. The directory's contents are gitignored; a note
enters the repository when it is promoted, not before.

## The loop

```bash
signalment ingest sources/lecture.pdf --topic cardio --species canine \
                  --system cardiovascular --title "Mitral valve disease"
```

That writes `drafts/vet-cardio-0001.md`: front-matter filled in, and the
source text attached beneath a TODO heading.

Open it. Write the note **above** the source block, in your own words. Then
delete the TODO heading and the whole source block, set `confidence`, and:

```bash
signalment promote drafts/vet-cardio-0001.md
```

## Why promote refuses drafts

`promote` will not accept a file that still contains the TODO heading or the
source block. That is not bureaucracy — it is the only guarantee the system
has. Pasting the extracted text into the note would produce a corpus of the
source's sentences labelled `provenance: self-authored`, every metric would
still look fine, and the whole project would quietly stop measuring anything
real.

The parts you must delete are exactly the parts that would prove you had not
rewritten it. So the check is mechanical and cannot be argued with.

It also refuses a note under 120 characters of prose, and one still marked
`confidence: draft` — nobody has checked it yet, and `verified` is a claim
about a person having done so.

## What it cannot check

Whether the note is any good, or true. That is yours. `confidence: verified`
means *you* compared it against a source, and the value is worthless if you
set it without doing that.
