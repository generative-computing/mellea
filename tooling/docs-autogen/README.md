# API Documentation Build System

Automated system for generating, decorating, and validating Mellea's API
documentation for the Docusaurus 3 site.

> **Where does this fit?** This README covers the autogen tooling.
> For the full CI/CD pipeline (build, validate, deploy) see
> [`docs/PUBLISHING.md`](../../docs/PUBLISHING.md).
> For doc authoring conventions (frontmatter, admonitions, links) see
> [`docs/CONTRIBUTING_DOCS.md`](../../docs/CONTRIBUTING_DOCS.md).

## Prerequisites

```bash
uv sync --all-extras --group dev   # installs mellea + build tooling (griffe, mdxify)
```

Node.js 22 LTS + `npm ci` (run from `docs/`) is required for Docusaurus previews.

## Quick Start

```bash
uv run poe apidocs           # Generate + decorate (version auto-read from pyproject.toml)
uv run poe apidocs-preview   # Generate fresh docs to /tmp and run quality audit
uv run poe apidocs-quality   # Audit docstring quality (all public symbols incl. methods)
uv run poe apidocs-validate  # Verify coverage + MDX syntax
uv run poe apidocs-clean     # Remove all generated API artefacts
uv run poe clidocs           # Generate CLI reference page from Typer metadata
uv run poe clidocs-clean     # Remove generated CLI reference page
```

## Pipeline Overview

```text
mellea/ source code
    │
    ▼
[1] generate-ast.py
    - Runs mdxify to extract classes, functions, docstrings
    - Reorganises flat mdxify output into nested folder structure
    - Strips empty files, updates frontmatter (title, description, sidebar_label)
    - Generates docs/docs/api/index.md (landing page — auto-discovered modules)
    │
    ▼
docs/docs/api/   (fully generated, replaces previous tree entirely)
    │
    ▼
[2] decorate_api_mdx.py
    - Normalise RST ``double-backtick`` literals to single backticks
    - Fix GitHub source links to the correct git ref (main for dev, vX.Y.Z for releases)
    - Replace mdxify's hardcoded 14 px source-icon size with 0.85 em
    - Inject per-module preamble text from __init__ docstrings
    - Wrap bare >>> doctest blocks in ```python fences
    - Escape { } in code blocks so MDX doesn't interpret them as JSX
    - Add cross-reference links (e.g. `Backend` → link to its definition page)
    - Add CLASS/FUNC pills and visual dividers to headings
    │
    ▼
[3] generate-ast.py --nav-only  (Mintlify-mode only — auto-skipped for Docusaurus)
    - Re-reads decorated files to rebuild the docs.json navigation
    - Ensures api/index.md cards show full module descriptions from docstrings
    - Skipped automatically when docs.json is absent (standard Docusaurus layout)
    │
    ▼
[4] generate_cli_reference.py  (--strict in CI / apidocs; lenient via clidocs)
    - Imports Typer app, walks Click command tree
    - Extracts flags, types, defaults, help strings
    - Parses docstring sections (Prerequisites, Output, Examples, See Also)
    - Emits docs/docs/reference/cli.md (standard Markdown)

── Optional quality tools (not part of the build pipeline) ──────────────────

validate.py  (uv run poe apidocs-validate)
    - GitHub source links point to the right git ref?
    - API symbol coverage ≥ threshold (default 80%)?
    - MDX syntax valid (no unescaped braces, no RST double backticks)?
    - Internal cross-reference links resolve?
    - No duplicate heading anchors within a file?
```

## File Structure

```text
tooling/docs-autogen/
├── README.md                  # This file
├── build.py                   # Unified wrapper: runs all four pipeline steps
├── generate-ast.py            # Step 1: MDX generation + nav + landing page
├── decorate_api_mdx.py        # Step 2: decoration, escaping, cross-references
├── validate.py                # Quality validation (coverage, syntax, anchors)
├── audit_coverage.py          # Symbol coverage + docstring quality audit
├── generate_cli_reference.py  # CLI reference page generator
├── test_cli_reference.py
├── test_cross_references.py
├── test_escape_mdx.py
├── test_mintlify_anchors.py
├── test_anchor_collisions.py
└── test_validate.py

docs/docs/
├── api/                       # Fully generated — do not edit by hand
│   ├── index.md               # Auto-generated landing page (module list)
│   └── mellea/                # Per-module MDX pages
└── reference/
    └── cli.md                 # Auto-generated CLI reference — edit source in cli/
```

## Configuration

**Version** is auto-detected from `pyproject.toml`. To override:

```bash
uv run python tooling/docs-autogen/build.py --version 0.4.0
```

**Cross-repo docs generation** — build docs for a different checkout without touching it:

```bash
# Generate from ../mellea-b, write output to /tmp/preview/api
uv run python tooling/docs-autogen/build.py \
    --source-dir ../mellea-b \
    --output-dir /tmp/preview/api

# Audit the output against the same source
uv run python tooling/docs-autogen/audit_coverage.py \
    --quality --no-methods \
    --docs-dir /tmp/preview/api \
    --source-dir ../mellea-b/mellea
```

**Coverage threshold** (`validate.py`, default 80%):

```bash
uv run poe apidocs-validate
uv run python tooling/docs-autogen/validate.py docs/docs/api --coverage-threshold 50
```

**Docstring quality audit** (`audit_coverage.py --quality`):

```bash
uv run poe apidocs-quality     # all public symbols including class methods

# Extended options
uv run python tooling/docs-autogen/audit_coverage.py \
    --docs-dir docs/docs/api --quality --no-methods  # skip class methods
uv run python tooling/docs-autogen/audit_coverage.py \
    --docs-dir docs/docs/api --quality --short-threshold 15
uv run python tooling/docs-autogen/audit_coverage.py \
    --docs-dir docs/docs/api --quality --output report.json
uv run python tooling/docs-autogen/audit_coverage.py \
    --docs-dir docs/docs/api --quality --fail-on-quality  # exit 1 on any issue
```

Eight issue kinds are reported:

| Kind | Flagged when |
| --- | --- |
| `missing` | No docstring |
| `short` | Fewer than `--short-threshold` words (default 5) |
| `no_args` | Function has parameters but no `Args:`/`Parameters:` section |
| `no_returns` | Non-`None` return annotation but no `Returns:` section |
| `no_raises` | Body contains `raise` but no `Raises:` section |
| `no_class_args` | `__init__` has typed params but no `Args:` section |
| `no_attributes` | Class has public attributes but no `Attributes:` section |
| `param_mismatch` | `Args:` documents names not in the real signature |

`*args`/`**kwargs`-only forwarders are exempt from `no_args` and `param_mismatch`.

**CLI reference generator** (`generate_cli_reference.py`):

```bash
uv run poe clidocs                                  # lenient — for local iteration
uv run python tooling/docs-autogen/generate_cli_reference.py --strict
uv run python tooling/docs-autogen/generate_cli_reference.py --docs-root /tmp/preview
uv run python tooling/docs-autogen/generate_cli_reference.py --source-dir ../mellea-b
```

`--strict` fails if any command lacks a summary, `Prerequisites:`, or `Output:` section,
or has options without `help=` text. The `apidocs` poe task always uses `--strict`.

**Skipping steps** (`build.py`):

```bash
uv run python tooling/docs-autogen/build.py --skip-cli-reference
uv run python tooling/docs-autogen/build.py --skip-generation
uv run python tooling/docs-autogen/build.py --skip-decoration
```

**Full clean:**

```bash
uv run poe apidocs-clean && uv run poe clidocs-clean
```

## Cross-References

`decorate_api_mdx.py` uses [Griffe](https://mkdocstrings.github.io/griffe/) to
resolve type names to their source modules and emit hyperlinks. The symbol cache
is built once per run (`build_symbol_cache()`), making cross-reference generation
fast (~10 s total).

`build.py` passes `--source-dir mellea` automatically when the directory exists.
Omit `--source-dir` to skip cross-references (e.g. when Griffe is not installed).

## Important: Pipeline Is Not Idempotent

Running `decorate_api_mdx.py` on already-decorated files **corrupts them** — each
pass appends without checking for prior output. Always regenerate from scratch:

```bash
uv run poe apidocs
```

Never run `decorate_api_mdx.py` directly on files already processed by it.

## Development

```bash
uv run poe apidocs-test                                   # all tooling unit tests
uv run pytest tooling/docs-autogen/test_escape_mdx.py -v
uv run pytest tooling/docs-autogen/test_cli_reference.py -v
```

To add a new decoration step:

1. Add the function to `decorate_api_mdx.py`
2. Call it from `process_mdx_file()` in the correct order
3. Write unit tests in a `test_*.py` file
4. Update the module docstring pass list and this README

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `No module named 'mdxify'` | `uv add --dev mdxify griffe` |
| `Could not parse expression with acorn` | Unescaped `{}` — run `uv run poe apidocs` to regenerate |
| `VIRTUAL_ENV … does not match` warning | Harmless — `uv run` uses the project venv regardless |
| API Reference tab shows 404 locally | Generated artefacts are gitignored — run `uv run poe apidocs` first |
| Duplicate preamble / double dividers in MDX | Files were decorated twice — run `uv run poe apidocs` (starts from fresh generation) |
| Griffe loading wrong package with `--source-dir` | Expected — Griffe uses `try_relative_path=False` |
