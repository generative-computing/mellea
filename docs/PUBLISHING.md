# Documentation Publishing Strategy

This document describes how Mellea's documentation is built, validated,
and deployed to Mintlify.

## Architecture

```text
main branch (source of truth)
├── docs/docs/           ← hand-authored MDX guides
├── mellea/, cli/        ← Python source (docstrings → API reference)
└── tooling/docs-autogen/  ← build & validation scripts
        │
        ▼  GitHub Actions (docs-publish.yml)
   ┌─────────────────────────────┐
   │  1. Install from source     │
   │  2. Generate API docs       │
   │  3. Validate everything     │
   │  4. Combine static + API    │
   └────────────┬────────────────┘
                │
        ┌───────┴───────┐
        ▼               ▼
  docs/staging     docs/production
  (from main)      (from releases)
        │               │
        ▼               ▼
     Mintlify        Mintlify
   (staging env)   (production env)
```

### Key principle

The **source of truth** is always `main` (or a feature branch targeting
`main`). The deployment branches `docs/staging` and `docs/production` are
**fully automated orphan branches** that are force-pushed on each run.

**Never edit files on the deployment branches directly.** Any manual
changes will be overwritten without warning on the next pipeline run.

## Deployment branches

| Branch | Trigger | Purpose |
| --- | --- | --- |
| `docs/staging` | Push to `main` (paths: `docs/**`, `mellea/**`, `cli/**`, `tooling/docs-autogen/**`) | Preview environment for reviewing docs before release |
| `docs/production` | GitHub release published | Production docs site served to users |
| `docs/preview` | PR with `docs-preview` label, or `workflow_dispatch` | Throwaway branch for validating the full pipeline |

All deployment branches are true orphan branches (no shared commit
history with `main`). They contain only the assembled documentation
output — static guides plus generated API reference.

## What the pipeline does

The `docs-publish.yml` workflow (`Docs` in GitHub Actions) runs these steps:

1. **Install dependencies** — `uv sync --all-extras --group dev` installs
   mellea from local source along with build tooling (griffe, mdxify).

2. **Generate API docs** — `tooling/docs-autogen/build.py` orchestrates:
   - `generate-ast.py`: runs `mdxify` against the installed package to
     produce MDX files, restructures into nested folders, updates
     frontmatter, generates the API Reference navigation in `docs.json`.
   - `decorate_api_mdx.py`: injects SidebarFix component, CLASS/FUNC
     pills, cross-reference links, and escapes JSX-sensitive syntax.
   - A final nav-rebuild pass to incorporate decorated preamble text.

3. **Validate** — multiple checks run (soft-fail by default):
   - **markdownlint** on static `.md` docs
   - **validate.py** on generated MDX: syntax, frontmatter, internal
     links, anchor collisions
   - **audit_coverage.py**: checks that generated docs cover ≥ 80% of
     public symbols

4. **Deploy** — uses `peaceiris/actions-gh-pages` to force-push the
   combined output to the target orphan branch.

### Validation strictness

By default, validation issues are **reported but do not block** the
deploy. This allows us to publish while resolving remaining issues.

To enable hard failures, use the `strict_validation` input:

- Via `workflow_dispatch`: check the "Fail the build if validation checks
  fail" checkbox.
- To make it the permanent default: change `default: false` to
  `default: true` in the workflow file.

## Local development

### Generate API docs locally

```bash
# Full pipeline: generate + decorate + nav rebuild
uv run poe apidocs

# Preview build to /tmp (doesn't touch your working tree)
uv run poe apidocs-preview

# Clean generated artifacts
uv run poe apidocs-clean
```

### Run validation locally

```bash
# Validate generated MDX (syntax, links, anchors)
uv run poe apidocs-validate

# Audit API coverage
uv run python tooling/docs-autogen/audit_coverage.py \
    --docs-dir docs/docs/api --threshold 80

# Audit docstring quality
uv run poe apidocs-quality

# Find orphan MDX files not in navigation
uv run poe apidocs-orphans
```

### Run the Mintlify dev server

```bash
cd docs/docs
npx mintlify dev
# → http://localhost:3000
```

## Testing the pipeline from a PR

### Label-based preview (recommended)

Add the **`docs-preview`** label to your PR. This triggers the full
pipeline including deployment to `docs/preview`. Remove the label when
you're done testing.

Without the label, PRs only run build + validation (no deploy).

> **Fork PRs:** The deploy step requires write access to the upstream
> repo. PRs from forks will build and validate successfully, but the
> deploy will fail with a permission error. Use manual dispatch instead
> (see below), or push the branch to the upstream repo.

### Manual dispatch

For more control (e.g. deploying to a custom branch):

1. Go to **Actions → Docs → Run workflow**.
2. Select your feature branch (must exist on the upstream repo).
3. Check **"Deploy even from a non-main context"**.
4. Optionally change the target branch (defaults to `docs/preview`).
5. Click **Run workflow**.

## Mintlify configuration

In the Mintlify dashboard (Git Configuration):

- **Staging environment** → branch: `docs/staging`, docs directory: `/` (root)
- **Production environment** → branch: `docs/production`, docs directory: `/` (root)

The deployment branches contain only documentation — `docs.json` sits at
the root, not under `docs/docs/` as it does on `main`. Set the docs
directory to `/` (root), not `docs/docs`.

## Pre-commit hooks

Two informational hooks run locally when relevant files change:

- **`docs-mdx-validate`**: validates MDX syntax/frontmatter when
  `.mdx` files or docs tooling change (requires `docs/docs/api/` to
  exist locally).
- **`docs-docstring-quality`**: audits docstring quality when Python
  source files change (requires `docs/docs/api/` to exist locally).

Both skip gracefully if generated docs are not present.

## File reference

| Path | Description |
| --- | --- |
| `.github/workflows/docs-publish.yml` | CI/CD workflow |
| `tooling/docs-autogen/build.py` | Unified build wrapper |
| `tooling/docs-autogen/generate-ast.py` | MDX generation + nav |
| `tooling/docs-autogen/decorate_api_mdx.py` | Decoration + escaping |
| `tooling/docs-autogen/validate.py` | Comprehensive validation |
| `tooling/docs-autogen/audit_coverage.py` | Coverage + quality audit |
| `tooling/docs-autogen/README.md` | Detailed tooling docs |
| `docs/docs/docs.json` | Mintlify configuration |
| `docs/docs/api/` | Generated API docs (gitignored) |
