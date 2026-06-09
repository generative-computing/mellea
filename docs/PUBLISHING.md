# Documentation Publishing Strategy

This document describes how Mellea's documentation is built, validated,
and deployed to GitHub Pages.

## Architecture

```text
main branch (source of truth)
├── docs/                    ← Docusaurus site root
│   ├── docs/                ← hand-authored MDX/MD guides
│   ├── docusaurus.config.ts ← site config, redirects, nav
│   ├── sidebars.ts          ← sidebar structure
│   └── src/                 ← custom components and CSS
├── mellea/, cli/            ← Python source (docstrings → API reference)
└── tooling/docs-autogen/    ← build & validation scripts
        │
        ▼  GitHub Actions (docs-publish.yml)
   ┌─────────────────────────────┐
   │  1. Install Python deps     │
   │  2. Generate API docs       │
   │  3. Validate everything     │
   │  4. Build Docusaurus site   │
   │  5. Deploy to gh-pages      │
   └────────────┬────────────────┘
                │
                ▼
          gh-pages branch
                │
                ▼
         GitHub Pages
      (docs.mellea.ai)
```

### Key principle

The **source of truth** is always `main`. The `gh-pages` branch is a
**fully automated output** — every deploy replaces it with a fresh build.

**Never edit files on `gh-pages` directly.** Any manual changes will be
overwritten without warning on the next pipeline run.

## When does deployment happen?

| Trigger | Build | Deploy |
| ------- | ----- | ------ |
| Push to `main` (docs/source paths) | ✅ | ✅ |
| GitHub release published | ✅ | ✅ (latest final only) |
| Pull request (docs/source paths) | ✅ (validate only) | ❌ |
| `workflow_dispatch` with `force_publish: true` | ✅ | ✅ |

PRs run the full build and validation pipeline — markdownlint, docstring
quality gate, Docusaurus build — but do **not** deploy. This surfaces
broken links and build failures in CI before merge without affecting
production.

## What the pipeline does

The `docs-publish.yml` workflow (`Docs` in GitHub Actions) runs these steps:

1. **Install dependencies** — `uv sync --all-extras --group dev` installs
   mellea from local source along with build tooling (griffe, mdxify).
   Node 22 + `npm ci` installs Docusaurus and its plugins.

2. **Generate API docs** — `tooling/docs-autogen/build.py` orchestrates:
   - `generate-ast.py`: runs `mdxify` against the installed package to
     produce MDX files, restructures into nested folders, updates
     frontmatter (`sidebar_label`, `title`, `description`).
   - `decorate_api_mdx.py`: injects CLASS/FUNC pills, cross-reference
     links, and escapes JSX-sensitive syntax. Also fixes source-link
     git refs (blob/main for dev builds, blob/vX.Y.Z for releases).
   - CLI reference: `generate_cli_reference.py` generates
     `docs/docs/reference/cli.md` from Typer command metadata.

3. **Validate** — multiple checks run:
   - **Markdownlint** on static `.md` docs (soft-fail)
   - **API validation** — `validate.py` checks anchor collisions within
     generated API pages and RST double-backtick notation in docstrings
     (soft-fail; these are gaps not caught by Docusaurus itself)
   - **API coverage audit** — warns if < 80% of public symbols are
     documented (soft-fail)
   - **Docstring quality gate** — fails if any public function with a
     `raise` statement is missing a `Raises:` entry (hard fail)
   - **CLI reference tests** — `pytest tooling/docs-autogen/test_cli_reference.py`

4. **Build** — `npm run build` in `docs/`. Runs with
   `onBrokenLinks: 'throw'` and `onBrokenAnchors: 'throw'`, so broken
   internal links and anchors are hard failures.

5. **Deploy** — `peaceiris/actions-gh-pages` force-pushes the built site
   to `gh-pages`. For release events, a `latest_check` step first confirms
   the release is the highest final semver — a backported patch to an old
   minor cannot overwrite the production site.

## Docs versioning

Docusaurus versioning is integrated into the release pipeline:

- **Pre-release / dev builds:** the site shows `main` as the only version
  — there are no snapshots yet.
- **On each final release** (`publish-release.yml` with `bump_type: final`
  or `patch-final`): the `snapshot-docs` job runs
  `docusaurus docs:version X.Y.Z`, commits `versioned_docs/version-X.Y.Z/`
  to `main`, and explicitly dispatches `docs-publish.yml` via
  `workflow_dispatch`. (A `GITHUB_TOKEN` push cannot trigger a new workflow
  run, so the dispatch is explicit rather than relying on a path filter.)
- **Version dropdown** in the navbar lets users switch between released
  versions and `main`.

No manual bootstrap is needed — the first final release run triggers
`snapshot-docs` automatically, which creates the initial snapshot and
updates `lastVersion`. Until the first final release, the site shows only
`main` in the version dropdown, which is correct.

## Local development

### Generate API docs locally

```bash
# Full pipeline: generate + decorate + nav rebuild
uv run poe apidocs

# Clean generated artifacts
uv run poe apidocs-clean
```

### Run the Docusaurus dev server

```bash
cd docs && npm ci   # first time only
npm run start
# → http://localhost:3000
```

Changes to `docs/docs/` are hot-reloaded. Generated API docs (`docs/docs/api/`) must be generated separately with `uv run poe apidocs` — they are gitignored.

### Run validation locally

```bash
# Audit API coverage (add --quality to also run docstring quality, as CI does)
uv run python tooling/docs-autogen/audit_coverage.py \
    --docs-dir docs/docs/api --threshold 80 --quality

# Audit docstring quality only (via poe alias)
uv run poe apidocs-quality

# Full Docusaurus build (catches broken links)
cd docs && npm run build
```

## Testing the pipeline from a PR

PR branches run the full build-and-validate job automatically. The
Docusaurus build enforces broken-link detection (`onBrokenLinks: 'throw'`),
so any broken internal link or anchor surfaces in CI before merge.

To preview the deployed site from a non-main branch:

1. Go to **Actions → Docs → Run workflow**.
2. Select your feature branch.
3. Check **"Deploy even from a non-main context"** (`force_publish: true`).
4. Click **Run workflow**.

> **Fork PRs:** the deploy step requires write access to the upstream repo.
> PRs from forks build and validate successfully, but the deploy will fail
> with a permission error. Push the branch to the upstream repo and use
> manual dispatch instead.

### Previewing from your own fork

To get a live preview site from your fork (useful for reviewing visual changes before submitting a PR):

1. **Enable GitHub Pages** on your fork: Settings → Pages → Source: `gh-pages` branch, root `/`.
   - GitHub requires a branch to exist before Pages can be enabled. If `gh-pages` doesn't exist yet, push any content to a temporary `docs/staging` branch, enable Pages pointing at it, then switch to `gh-pages` once the first deployment creates it.
2. **Push your branch** to your fork (`git push origin my-branch`).
3. The `docs-publish.yml` workflow runs automatically — it builds the site with `baseUrl: /mellea/` (fork-aware) and deploys to `gh-pages` on your fork.
4. Your preview is live at `https://<your-username>.github.io/mellea/`.

> The fork `baseUrl` (`/mellea/`) differs from upstream (`/`). Internal links and assets will resolve correctly on the fork preview, but absolute URLs pointing to `docs.mellea.ai` will still go to the production site.

## File reference

| Path | Description |
| --- | --- |
| `.github/workflows/docs-publish.yml` | CI/CD workflow (build, validate, deploy) |
| `.github/workflows/publish-release.yml` | Release pipeline — includes `snapshot-docs` job |
| `.github/scripts/set-last-version.mjs` | Updates `lastVersion` in `docusaurus.config.ts` on release |
| `tooling/docs-autogen/build.py` | Unified build wrapper |
| `tooling/docs-autogen/generate-ast.py` | MDX generation + frontmatter update |
| `tooling/docs-autogen/decorate_api_mdx.py` | Decoration + escaping |
| `tooling/docs-autogen/generate_cli_reference.py` | CLI reference page generator |
| `tooling/docs-autogen/audit_coverage.py` | Coverage + quality audit |
| `tooling/docs-autogen/validate.py` | Anchor / RST-syntax validation |
| `tooling/docs-autogen/README.md` | Detailed tooling docs |
| `docs/docusaurus.config.ts` | Docusaurus config (redirects, nav, plugins) |
| `docs/sidebars.ts` | Sidebar structure for docs and API reference |
| `docs/src/` | Custom CSS and MDX shim components |
| `docs/docs/api/` | Generated API docs (gitignored) |
