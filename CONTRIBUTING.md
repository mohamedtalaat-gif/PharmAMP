# Contributing

This is a small, early-stage project — a metrics framework plus a benchmark
harness, not a large application. Issues and pull requests are welcome;
scope changes are easiest to agree on before writing code, so open an issue
first for anything beyond a small fix.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

The `dev` extra is enough for `pharmamp/metrics/` and `app.py`. Two more
extras exist for the rest of the codebase — see [README.md](README.md)'s
Install section for what each needs and why they're kept separate:

- `benchmark` — runs generation through the vendored OmegAMP checkpoint.
- `validate` — reproduces the wet-lab correlation check.

## Running tests

```bash
pytest
```

Tests that need the OmegAMP checkpoint/embeddings or the (non-redistributable)
wet-lab supplement file skip cleanly when those aren't present — you don't
need either to contribute to `pharmamp/metrics/` or `app.py`.

## Conventions

- No premature abstraction: this codebase intentionally avoids shared base
  classes and generic frameworks where three concrete implementations would
  do. If you're adding a fourth similar thing, that's the point to consider
  extracting one, not before.
- Every heuristic or descriptor choice should trace back to a citable
  source (a paper, a named method) in a comment or docstring — see
  `pharmamp/metrics/aggregation.py` for the pattern. "Seemed reasonable" is
  not sufficient justification for a scoring formula.
- Comments explain *why*, not *what* — if removing a comment wouldn't
  confuse a future reader, it shouldn't be there.
- Keep prose (docstrings, comments, commit messages, this file) in plain,
  direct language — no meta-narration about how or why a change was made,
  just the change and its rationale.

## Reporting bugs / requesting features

Use the issue templates. For anything touching `pharmamp/metrics/`, include
a concrete input sequence and the score you got vs. what you expected —
that's usually enough to reproduce.
