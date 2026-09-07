# Rules for AI assistants — Signalment

Applies to every agent working in this repository — Cline, Claude Code, or
anything else. Full version: [`.clinerules`](.clinerules).

## Never generate the data

**Do not write, edit, generate, or delete anything in `corpus/notes/` or
`eval/gold/`.**

Those directories hold human-authored veterinary notes and human-written
evaluation questions. Their entire value is that a person authored and verified
them. Generated content there is not a visible bug — lint can pass, tests can
pass, metrics can improve — while every number the project reports quietly
becomes meaningless, because the system ends up measuring whether it can
retrieve text written to be retrieved.

If a task appears to need new notes or gold questions: **stop and say so.** That
work belongs to the repository owner.

Happened once already, commit `1e8df26`. See `corpus/quarantine/README.md`.

## This is enforced, not requested

`.githooks/pre-commit` refuses any commit that stages a file under
`corpus/notes/` or `eval/gold/`. Install it with `make hooks`. A human editing
their own notes bypasses it deliberately:

```bash
SIGNALMENT_HUMAN_EDIT=1 git commit -m "..."
```

See `docs/agent-setup.md` for per-assistant setup, including the system prompt
to paste into a chat client that cannot read this file.

## Before every commit

```bash
make check      # lint + tests + drift; do not commit if it fails
make rust-test  # additionally, for Rust changes
```

## Everything else

`src/signalment/`, `crates/`, `tests/`, `benchmarks/` — edit freely, within the
scope you were asked for.
