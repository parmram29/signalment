# Working on this repo with AI assistants

Three assistants touch this project (Claude Code, Cline, Qwen Coder in Cherry
Studio). They read different files and some read none, so the guardrail exists
at three levels. Only the last one is enforcement.

| Level | Mechanism | Holds against |
|---|---|---|
| Convention | `AGENTS.md`, `CLAUDE.md`, `QWEN.md`, `.clinerules` | an assistant that reads its context file |
| Instruction | the system prompt below | a chat app with no repo awareness |
| **Enforcement** | `.githooks/pre-commit` | **everything, including me** |

Install the hook once per clone:

```
make hooks
```

It refuses any commit that stages a file under `corpus/notes/` or `eval/gold/`.
When *you* edit your own notes, say so explicitly:

```
SIGNALMENT_HUMAN_EDIT=1 git commit -m "Add cardiology notes"
```

That is deliberately awkward. Typing it is a moment to ask whether a person
really wrote what is about to be committed.

## Cherry Studio / Qwen Coder

Cherry Studio is a chat client, not a repo-aware agent: it will not read
`QWEN.md` on its own. Paste this into the assistant's system prompt.

```
You are helping build Signalment, a retrieval system over one veterinary
student's own notes. Graduate level (DVM and above).

ABSOLUTE RULE
Never write, generate, edit or delete content for corpus/notes/ or
eval/gold/. Not one note, not one evaluation question, not even to fix a
malformed one. Those hold notes a person authored and verified, and
questions that person actually asked while studying. Generated content
there does not look like a bug — lint passes, tests pass, metrics improve
— while every number the project reports stops meaning anything, because
the system then measures whether it can retrieve text written to be
retrieved. This has already happened once and cost a day.
If a task seems to need new notes or questions: stop and say so.

WHAT YOU MAY EDIT
src/signalment/, crates/signalment_bm25/, tests/, benchmarks/.
For docs/ and README.md, propose text; the owner writes the prose.

BEFORE PROPOSING A COMMIT
`make check` must pass (lint, 152 tests, gold-set drift).
Rust changes also need `make rust-test`.
Never suggest committing when either fails.

THE PYTHON BM25 IS THE SPECIFICATION
src/signalment/retrieval/bm25.py is the reference. The Rust crate must
match it exactly; tests/test_backend_equivalence.py asserts identical
rankings and scores to 1e-9. If they disagree, the Rust is wrong, unless
you can show the Python is wrong and fix both with a test pinning the
correction. The idf form is easy to get wrong:
    idf(t) = max( ln(1 + (N - df + 0.5) / (df + 0.5)), 0 )
The leading 1 + is load-bearing. Without it a term appearing in every
document scores ln(1) = 0 and queries silently return nothing.

HOW TO ANSWER
Show a diff or a complete file, never a fragment I have to splice.
State which files you changed and what command verifies it.
If you are unsure whether something is in scope, ask instead of doing it.
```

## Approval mode: the setting that matters most

The Qwen Code CLI takes `approvalMode` in `settings.json`. `auto-edit`
approves every file write without asking. Combined with an agent that decides
it needs more corpus, that is how 51 generated notes reached `corpus/notes/`
and overwrote four real ones.

`.qwen/settings.json` in this repository sets `approvalMode: default` so edits
are confirmed while working here, without changing your global preference. If
your Qwen build does not merge project-level settings, you will still see
`auto-edit` — check with `/settings` or whatever your build exposes, and set it
per-session instead.

**Know what the hook does and does not cover.** `.githooks/pre-commit` protects
the repository's *history*. It cannot protect the *working tree*: an agent on
auto-edit can overwrite `corpus/notes/vet-cardio-0003.md` on disk right now,
and the hook only speaks up later, when something tries to commit it. By then
your text is gone unless git already had a copy.

So: **commit notes as you write them.** Not at the end of a session, not when
the set feels finished. A note that exists only in your working tree is one
careless agent away from not existing.

```bash
SIGNALMENT_HUMAN_EDIT=1 git commit -m "Add three cardiology notes"
```

## Model parameters

For work on this repository, not general chat:

| Parameter | Value | Why |
|---|---|---|
| temperature | **0.1–0.2** | Code and schema edits want reproducibility. Sampling variety produces plausible-looking wrong front-matter, which is exactly the failure this project already had. |
| top_p | 0.8 | Qwen's own default; leave it unless temperature alone is not enough. |
| max output tokens | ≥ 4096 | Whole-file rewrites get truncated below this, and a truncated file that still parses is worse than one that does not. |
| context window | as large as the machine allows | The system prompt plus one source file plus its test file is the working minimum. |

Check the model card for the build you are running; these are starting points,
not gospel. Raise temperature only for prose drafts, never for code or
front-matter.

## Local inference on this machine

The development machine is an Intel i3-1000NG4, two cores at 1.1 GHz, no
usable GPU. A 7–8B model at Q4 runs at roughly 1–2 tok/s there, so a cited
300-token answer takes minutes. Consequences, per blueprint D12:

- Cherry Studio should point at a **hosted endpoint** for anything interactive.
- Local models are viable here only at 1–3B, and only for offline experiments.
- None of this touches retrieval: all three retrieval backends are CPU-only
  and run in milliseconds on this hardware. Only generation is affected.
