# Locked dependency sets

`pyproject.toml` declares *ranges* — that is how the package installs and it
must stay loose, or `pip install signalment` fights every other package in the
environment. These files pin the *exact* resolution, which is a different job.

| File | Contents |
|---|---|
| `dev.lock` | runtime + test + lint. What CI installs and what `make check` is measured against. |
| `retrieval.lock` | runtime + sentence-transformers, torch, transformers. Needed for dense retrieval. |

## Why this project in particular needs them

The claim on the front of the README is that every number can be reproduced
with one command. Unpinned dependencies quietly make that false:

- **Embeddings are version-sensitive.** A different `torch` or
  `sentence-transformers` can change the vectors a model produces, which moves
  recall@5 and nDCG@5 without a line of this project's code changing. An
  ablation table whose rows were measured against different resolutions is
  comparing more than it claims to.
- **CI goes red for reasons that are not yours.** A transitive package
  releasing a breaking version fails the build on a commit that touched
  nothing related.

`Cargo.lock` is committed for the same reason: the crate is built into a wheel,
which is an application artifact, not a library for others to resolve against.

## Regenerating

```bash
make lock
```

Requires [uv](https://github.com/astral-sh/uv). Commit the result, and say in
the message why a version moved — a lockfile diff nobody can explain is worse
than no lockfile.

## Installing

```bash
make install              # dev.lock, editable, plus git hooks
pip install -r requirements/retrieval.lock   # add dense retrieval
```
