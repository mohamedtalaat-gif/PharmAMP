# Security Policy

## Supported Versions

Pre-1.0: only the latest commit on `main` is supported.

## Reporting a Vulnerability

Please report security issues privately using GitHub's
[Security Advisories](../../security/advisories/new) for this repository
("Report a vulnerability" under the Security tab), rather than opening a
public issue.

## Scope notes specific to this project

- Loading the OmegAMP checkpoint (`models/generative_model.ckpt`) involves
  unpickling it — only fetch it via the exact `gdown` command in
  [README.md](README.md), and don't point `OmegAMPGenerator`'s
  `checkpoint_path`/`embeddings_path` at a file from an untrusted source.
  The class currently only checks that the path exists, not that its
  contents are what they claim to be.
- The `data/wetlab-supplement/` and `data/generative-model-data/`,
  `models/`, and `third_party/` directories hold large or licensed external
  data and are intentionally excluded from version control — see
  `.gitignore` and [README.md](README.md) for how to obtain them.
