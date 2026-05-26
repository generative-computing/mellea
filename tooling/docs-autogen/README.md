# API Documentation Build System

Automated system for generating, decorating, and validating Mellea API
documentation using Mintlify.

## Prerequisites

```bash
uv sync --all-extras --group dev   # Same as CI — installs all extras + dev tools
```

Node.js LTS (v22 or earlier) is required for `mintlify dev` previews.

## Quick Start

```bash
uv run poe apidocs           # Generate + decorate (version auto-read from pyproject.toml)
uv run poe apidocs-preview   # Generate fresh docs to /tmp and run quality audit
uv run poe apidocs-quality   # Audit docstring quality (public symbols, no methods)
uv run poe apidocs-orphans   # Find MDX files not referenced in docs.json navigation
uv run poe apidocs-validate  # Verify coverage + MDX syntax
uv run poe apidocs-clean     # Remove generated artefacts
uv run poe clidocs           # Generate CLI reference page from Typer metadata
uv run poe clidocs-clean     # Remove generated CLI reference page
```

The `apidocs` task runs `build.py`, which orchestrates four steps: AST generation
(`generate-ast.py`), MDX decoration (`decorate_api_mdx.py`), nav rebuild, and CLI
reference generation (`generate_cli_reference.py`). Both the `docs/docs/api/`
directory and `docs/docs/api-reference.mdx` are **fully generated artefacts** —
do not commit or edit them by hand. `docs/docs/reference/cli.md` is also generated
— edit the CLI command docstrings in `cli/` instead.

`clidocs` runs `generate_cli_reference.py` without validation flags — useful for
quick local iteration. The `apidocs` pipeline always passes `--strict`, which fails
the build if any command is missing required docstring sections or has options
without `help=` text. To test strict validation locally:

```bash
uv run python tooling/docs-autogen/generate_cli_reference.py --strict
```

## Pipeline Overview

```text
mellea/ source code
    │
    ▼
[1] generate-ast.py
    - Runs mdxify to extract classes, functions, docstrings
    - Reorganises flat output into nested folder structure
    - Strips empty files, updates frontmatter
    - Generates docs/docs/api-reference.mdx (landing page)
    - Writes API Reference tab into docs/docs/docs.json
    │
    ▼
docs/docs/api/   (fresh copy, replaces previous entirely)
    │
    ▼
[2] decorate_api_mdx.py
    - Fix GitHub source links (version tag)
    - Inject per-module preamble text
    - Inject SidebarFix Mintlify component
    - Escape { } in code blocks so MDX doesn't interpret them as JSX
    - Add cross-reference links (e.g. `Backend` → link to its definition)
    - Add CLASS/FUNC pills and visual dividers to headings
    │
    ▼
[3] generate-ast.py --nav-only
    - Re-reads decorated files to rebuild the landing page and nav
    - Ensures API Reference tab cards show full module descriptions
    │
    ▼
[4] generate_cli_reference.py  (--strict in CI / apidocs; lenient via clidocs)
    - Imports Typer app, walks Click command tree
    - Extracts flags, types, defaults, help strings
    - Parses docstring sections (Prerequisites, Output, Examples, See Also)
    - Emits docs/docs/reference/cli.md (standard Markdown)
    │
    ▼
[5] Mintlify dev server  (mintlify dev from docs/docs/)
    http://localhost:3000

── Optional tools (not part of the build pipeline) ──────────────────────────

validate.py  (uv run poe apidocs-validate)
    - GitHub source links correct?
    - API coverage ≥ threshold?
    - MDX syntax valid (no unescaped braces)?
    - Internal cross-reference links resolve?
    - No duplicate heading anchors?
```

## File Structure

```text
tooling/docs-autogen/
├── README.md               # This file
├── build.py                # Unified wrapper: calls generate-ast then decorate
├── generate-ast.py         # Step 1: MDX generation + nav + landing page
├── decorate_api_mdx.py     # Step 2: decoration, escaping, cross-references
├── validate.py             # Step 3: quality validation
├── audit_coverage.py       # Symbol coverage + quality audit
├── generate_cli_reference.py  # CLI reference page generator
├── test_cli_reference.py   # Tests for CLI reference generation
├── test_escape_mdx.py      # Tests for MDX brace escaping
├── test_cross_references.py
├── test_mintlify_anchors.py
├── test_anchor_collisions.py
└── test_validate.py

docs/docs/
├── docs.json               # Mintlify config — API Reference tab auto-generated
├── api-reference.mdx       # Auto-generated landing page
├── api/                    # Fully generated — do not edit
│   └── mellea/
├── reference/
│   └── cli.md              # Auto-generated CLI reference — do not edit
└── snippets/
    └── SidebarFix.mdx      # Mintlify sidebar component (hand-maintained)
```

## Configuration

**Version** is auto-detected from `pyproject.toml`. To override:

```bash
uv run python tooling/docs-autogen/build.py --version 0.4.0
```

**Cross-repo docs generation** — generate docs for a different checkout without
touching that repo (e.g. `../mellea-b`) by passing `--source-dir`:

```bash
# Build docs from mellea-b source, write output to /tmp/mellea-b-preview/api
uv run python tooling/docs-autogen/build.py \
    --source-dir ../mellea-b \
    --output-dir /tmp/mellea-b-preview/api

# Then audit the generated output against the same source
uv run python tooling/docs-autogen/audit_coverage.py \
    --quality --no-methods \
    --docs-dir /tmp/mellea-b-preview/api \
    --source-dir ../mellea-b/mellea
```

Both `build.py` and `audit_coverage.py` accept `--source-dir` to point at a
different repo. `generate-ast.py` also accepts `--source-dir` if you need to
call it directly.

**Coverage threshold** (`validate.py`, default 80%):

```bash
uv run poe apidocs-validate  # uses default threshold
uv run python tooling/docs-autogen/validate.py docs/docs/api --version 0.3.2 --coverage-threshold 50
```

**Docstring quality audit** (`audit_coverage.py --quality`):

```bash
uv run poe apidocs-quality          # missing, short, no Args/Returns/Raises (top-level, no methods)

# Extended options (direct invocation)
uv run python tooling/docs-autogen/audit_coverage.py \
    --docs-dir docs/docs/api --quality               # include class methods too
uv run python tooling/docs-autogen/audit_coverage.py \
    --docs-dir docs/docs/api --quality --short-threshold 15   # raise "short" threshold
uv run python tooling/docs-autogen/audit_coverage.py \
    --docs-dir docs/docs/api --quality --output report.json   # save JSON report
uv run python tooling/docs-autogen/audit_coverage.py \
    --docs-dir docs/docs/api --quality --fail-on-quality      # exit 1 if any issues (CI/pre-commit)
```

Eight issue kinds are reported:

| Kind | Flagged when |
| --- | --- |
| `missing` | No docstring at all |
| `short` | Fewer than `--short-threshold` words (default 5) |
| `no_args` | Function has parameters but no `Args:`/`Parameters:` section |
| `no_returns` | Function has a non-`None` return annotation but no `Returns:` section |
| `no_raises` | Function body contains `raise` statement but no `Raises:` section |
| `no_class_args` | Class `__init__` has typed params but no `Args:` section |
| `no_attributes` | Class has public attributes but no `Attributes:` section |
| `param_mismatch` | `Args:` section documents parameter names not in the real signature |

**`*args` / `**kwargs` forwarders** — functions whose only non-`self` parameters are
`*args` and/or `**kwargs` are exempt from both `no_args` and `param_mismatch`. These
are variadic forwarders where the concrete signature carries no named parameters; the
docstring `Args:` section conventionally describes the accepted keyword arguments rather
than the signature itself. Authors should document those kwargs freely — the audit will
not flag them as mismatches.

The quality audit is informational — it does not fail the build unless `--fail-on-quality`
is passed. Use `--output` to track trends over time or feed results into issue tracking.

**Navigation orphan audit** (`audit_coverage.py --orphans`):

```bash
uv run poe apidocs-orphans    # find MDX files not referenced in docs.json navigation
```

Reports any generated MDX files in `docs/docs/api/` that are absent from the
navigation tree in `docs/mint.json`. Also works with `--source-dir` for
cross-repo audits.

**Navigation and landing page** are auto-generated by `generate-ast.py`.
Do not edit the API Reference tab in `docs.json` or `api-reference.mdx` by hand
— both are overwritten on every `uv run poe apidocs` run.

**CLI reference generator** (`generate_cli_reference.py`):

```bash
uv run poe clidocs                   # generate (lenient — for local iteration)
uv run python tooling/docs-autogen/generate_cli_reference.py --strict        # strict mode (same as CI)
uv run python tooling/docs-autogen/generate_cli_reference.py --docs-root /tmp/preview  # custom output location
uv run python tooling/docs-autogen/generate_cli_reference.py --source-dir ../mellea-b  # cross-repo
```

`--strict` exits non-zero if any command lacks a summary, `Prerequisites:`, or
`Output:` section, or has options without `help=` text. The `apidocs` pipeline
always runs with `--strict`; `clidocs` does not.

**Skipping pipeline steps** (`build.py`):

```bash
uv run python tooling/docs-autogen/build.py --skip-cli-reference   # API docs only, no CLI reference
uv run python tooling/docs-autogen/build.py --skip-generation       # decoration + CLI reference only
uv run python tooling/docs-autogen/build.py --skip-decoration       # generation + CLI reference only
```

**Full clean** — `apidocs-clean` removes only the API artefacts; `clidocs-clean`
removes only the CLI reference. To remove everything:

```bash
uv run poe apidocs-clean && uv run poe clidocs-clean
```

## Cross-References

`decorate_api_mdx.py` uses [Griffe](https://mkdocstrings.github.io/griffe/) to
resolve type names to their source modules and emit hyperlinks. The package is
loaded **once** per run (via `build_symbol_cache()`), then the resulting
`symbol → module` dict is reused across all files — making cross-reference
generation fast (~10 s total, down from ~3 min with the old per-symbol load).

`build.py` passes `--source-dir mellea` automatically if the directory exists.
To disable cross-references (e.g. if Griffe is not installed), omit `--source-dir`.

## Important: Pipeline Is Not Idempotent

Running `decorate_api_mdx.py` on already-decorated files **corrupts them**.
Each decorator (`inject_preamble`, `decorate_mdx_body`, `add_cross_references`)
appends or wraps without checking whether its output already exists.

**Always run the full pipeline from scratch:**

```bash
uv run poe apidocs   # generate-ast.py replaces api/ entirely, then decorates fresh files
```

Never run `decorate_api_mdx.py` standalone on files that have already been
decorated.

## Development

```bash
# Run all unit tests
uv run poe apidocs-test

# Run a specific test file
uv run pytest tooling/docs-autogen/test_escape_mdx.py -v
uv run pytest tooling/docs-autogen/test_cli_reference.py -v

# Find MDX files not in docs.json navigation
uv run poe apidocs-orphans
```

To add a new decoration step:

1. Add the function to `decorate_api_mdx.py`
2. Call it from `process_mdx_file()` in the correct order
3. Write unit tests in a `test_*.py` file
4. Update this README

## Known Issues

### Runtime parsing errors in Mintlify dev server

Despite `validate.py` reporting no MDX syntax errors, the Mintlify dev server
shows parse errors in a handful of files (tracebacks, multiline JSON examples).
The enhanced blockquote/escaping logic in `escape_mdx_syntax()` addresses the
most common cases but has not been fully verified end-to-end.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `No module named 'mdxify'` | `uv add --dev mdxify griffe` |
| `Could not parse expression with acorn` | Unescaped `{}` — run `uv run poe apidocs` to regenerate |
| `VIRTUAL_ENV … does not match` warning | Harmless — `uv run` uses the project venv regardless |
| Port 3000 in use | `lsof -ti:3000 \| xargs kill -9` |
| `mint dev is not supported on node versions 25+` | Switch to Node.js LTS v22 via your version manager (nvm, volta, fnm) |
| API Reference tab shows 404 locally | Generated artefacts are gitignored — run `uv run poe apidocs` first |
| Duplicate preamble / double dividers in MDX | Files were decorated twice — run `uv run poe apidocs` (starts from fresh generation) |
| Griffe loading wrong package when using `--source-dir` | Expected — Griffe uses `try_relative_path=False` to avoid loading same-named packages from CWD |
