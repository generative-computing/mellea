# RELEASE.md

## Overview

Mellea uses a release-branch workflow. Every minor version has a long-lived
`release/vX.Y` branch that carries release candidates, the final minor release,
and any subsequent patch releases. `main` carries `.dev`-versioned work for the
next minor.

This gives each release a frozen codebase without requiring cherry-picks back
into `main`, and keeps CD resilient to concurrent merges on `main`.

## Release Cadence

Minor releases target a roughly 4-week cadence. Patch releases happen as
needed.

## Versioning

Versions follow **[PEP 440](https://peps.python.org/pep-0440/)** (which is
compatible with SemVer for final releases).

| Phase | Branch | Version example | Tag |
|-------|--------|-----------------|-----|
| Dev on main | `main` | `0.6.0.dev0` | (untagged) |
| Release branch cut | `release/v0.6` | `0.6.0rc0` | `v0.6.0rc0` if `PUBLISH_PRERELEASES`, else untagged |
| Further RCs | `release/v0.6` | `0.6.0rc1`, `rc2`, … | `v0.6.0rcN` if `PUBLISH_PRERELEASES`, else untagged |
| Final minor | `release/v0.6` | `0.6.0` | `v0.6.0` |
| Patch RC | `release/v0.6` | `0.6.1rc0` | `v0.6.1rc0` if `PUBLISH_PRERELEASES`, else untagged |
| Patch final | `release/v0.6` | `0.6.1` | `v0.6.1` |
| Next minor dev on main | `main` | `0.7.0.dev0` | (untagged) |

Invariants:

- `main` always carries `X.Y.0.devN`. `main` is tagged only when a dev
  publication runs (`publish-dev-from-main` workflow); never during routine
  commits.
- Release branches always carry `X.Y.Zrc?` or `X.Y.Z`.
- Prereleases (`rcN`, `.devN`) are tagged, uploaded to PyPI, and given a
  prerelease-marked GitHub Release only when the `PUBLISH_PRERELEASES` repo
  variable is `true` (see below). With the default (`false`), the version
  bump commits to the branch but no tag, no PyPI upload, and no Release.
  Prerelease Releases use `--prerelease` so they don't appear as the
  repo's "latest" release on GitHub.

## The `PUBLISH_PRERELEASES` flag

Repo variable `PUBLISH_PRERELEASES` (default `false`) governs whether
prereleases are tagged, uploaded to PyPI, and given a GitHub Release.

| `PUBLISH_PRERELEASES` | rc / dev | Finals |
|-----------------------|----------|--------|
| `false` (default) | version bump committed; no tag, no Release, no PyPI | tag + GitHub Release + PyPI + changelog entry + sync PR |
| `true` | tag + prerelease GitHub Release + PyPI | tag + GitHub Release + PyPI + changelog entry + sync PR |

With the default, prereleases stay branch-local — the bump commit identifies
the version but there is no immutable tag pointer. Users who need a specific
prerelease can install from a branch SHA
(`pip install git+https://github.com/generative-computing/mellea@<sha>`).

When the flag is enabled, every rc and dev produces a `--prerelease`-marked
GitHub Release. The notes are incremental — `0.6.0rc2` diffs against
`0.6.0rc1`, so testers can see "what changed in this rc" without re-reading
the full cycle. The cumulative view ("everything in 0.6") shows up on the
final's Release page, which diffs against the previous final. Prerelease
Releases never become the repo's "latest" release.

Finals always follow the full release flow regardless of the flag.

To enable prerelease publishing, a repo admin sets the variable to `true`
under **Settings → Secrets and variables → Actions → Variables**. No code
change needed.

## Workflows

| Workflow | Purpose |
|----------|---------|
| `cut-release-branch` | Cut `release/vX.Y` from `main`, publish `X.Y.0rc0`, bump `main` to next minor `.dev0` |
| `publish-release` | Publish a release (rc, final, patch-rc, patch-final, or retry a failed publish) |
| `publish-dev-from-main` | Iterate main's `.devN` counter and publish a dev release |

All three are `workflow_dispatch`-only and run from the GitHub Actions UI.

Whether any given prerelease (`rc`, `dev`) produces a PyPI artifact depends
on the `PUBLISH_PRERELEASES` flag described above.

## Cutting a minor release branch

When `main` is ready to freeze for the next minor:

1. Go to **Actions → Cut release branch → Run workflow**.
2. Optionally enter the expected minor (e.g. `0.6`) in `confirm_minor` as a
   safety check. Leave blank to trust whatever is in `pyproject.toml` on
   `main`.
3. Run.

The workflow:

- Verifies `pyproject.toml` on `main` matches `X.Y.0.devN`.
- Creates `release/vX.Y` with version set to `X.Y.0rc0`.
- Publishes `X.Y.0rc0` per the `PUBLISH_PRERELEASES` flag — by default the
  version bump is committed to the release branch with no tag and no PyPI
  upload; with the flag enabled, also tags and uploads.
- Pushes `main` with version bumped to `X.(Y+1).0.dev0`.

The `main` push requires `github-actions[bot]` to be listed as a bypass actor
in the `main` branch-protection ruleset (see **Branch protection** below).

## Publishing a release candidate

Once a release branch exists:

1. Go to **Actions → Publish release → Run workflow**.
2. Select the release branch (e.g. `release/v0.6`) from the branch picker.
3. Choose `bump_type: rc`.
4. Run.

The workflow:

- Computes the next rc (e.g. `0.6.0rc0` → `0.6.0rc1`).
- Commits the bump to the release branch.
- When `PUBLISH_PRERELEASES=true`: pushes tag `v{version}`, creates a
  `--prerelease`-marked GitHub Release with incremental notes (diffed
  against the previous rc), and uploads to PyPI.
- With the default (`false`): the bump commit is the only artifact — no
  tag, no Release, no PyPI upload.
- Either way, no changelog entry, no sync PR — those are reserved for
  finals.

## Promoting an RC to a final minor

When testing on an RC is complete:

1. **Actions → Publish release → Run workflow** against the same release branch.
2. `bump_type: final`.
3. Run.

This creates the `v0.6.0` GitHub Release (with auto-generated notes from
the previous final), uploads to PyPI, appends to `CHANGELOG.md` on the
release branch, opens a sync PR to `main` with the changelog delta, and
triggers the docs production deploy.

## Patch releases

Patches live on the original release branch. `main` is touched only when a
`patch-final` lands and opens its changelog sync PR.

### 1. Land the fix on the release branch

Open a PR targeting the release branch (e.g. `release/v0.6`). Branch
protection applies the same way as `main`: review + CI required.

If the same fix also belongs on `main`, open a separate followup PR — usually
after the release-branch PR merges, but the order can be flipped if it makes
review easier. Either direction works; the only constraint is that both
branches end up carrying the change so it doesn't regress in the next minor.

### 2. Publish a patch RC and final

1. **Publish release** against `release/v0.6` with `bump_type: patch-rc`. Produces
   e.g. `v0.6.1rc0`.
2. Test.
3. **Publish release** again with `bump_type: patch-rc` for additional rcs if needed.
4. **Publish release** with `bump_type: patch-final` to promote to `v0.6.1`.

## Publishing a dev release from main

Ad-hoc, case-by-case `.devN` bumps on `main`. Typical uses: a contributor
wants a tagged snapshot of main for debugging, an external tester needs a
specific point-in-time artifact, etc. Not intended for routine or scheduled
releases.

1. **Actions → Publish dev release from main → Run workflow** (must dispatch
   against `main`).
2. Run.

The workflow (publish-then-increment):

1. Publishes `main`'s **current** `.devN` per `PUBLISH_PRERELEASES` — skipped
   by default; tag + prerelease GitHub Release + PyPI upload if enabled.
   When tagged, the tag points at the current `main` HEAD.
2. Iterates pyproject on main: `X.Y.Z.devN → X.Y.Z.dev(N+1)`, commits, pushes.

The invariant is that `main`'s pyproject always carries "the next version
that would be published." Inspecting main tells you what the next dispatch
will produce.

With `PUBLISH_PRERELEASES=false` (default), the publish step is a no-op —
the `.devN` counter still advances, but no tag is pushed and nothing reaches
PyPI. With the flag enabled, the publish tags main HEAD, creates a
`--prerelease`-marked GitHub Release, and uploads to PyPI (installable via
`pip install --pre mellea`). Dev publishes never touch `CHANGELOG.md` and
never become the repo's "latest" release.

## Rollback and retry

`bump_type: none` re-runs CD against whatever version is currently in
`pyproject.toml`, skipping the version-bump step. Useful when a previous
run failed after the bump committed but before the publish completed.

The retry is a "skip what's done, finish what isn't" path — it does not
validate existing artifacts. If a prior run produced a tag pointing at the
wrong commit, a Release with stale notes, or a sync PR with a stale body,
delete the bad artifact (`gh release delete`, `git push --delete`,
`gh pr close`) before re-dispatching. Retry is for resuming after a partial
failure, not for fixing a corrupted prior result.

## Release branch retention

**Release branches are never deleted.** GitHub Releases pin to specific
commits on each branch, so pruning a branch would orphan those references and
break `git checkout v0.4.2` semantics. Old `release/v0.3`, `release/v0.4`, etc.
stay around indefinitely.

## Branch protection

All four write-capable workflows authenticate via `secrets.GITHUB_TOKEN`.
Each declares the scopes it needs via an inline `permissions:` block.

`github-actions[bot]` (the identity `GITHUB_TOKEN` acts as) needs to be
listed as a **bypass actor** on two rulesets:

- `main`: `cut-release-branch` pushes the `X.(Y+1).0.dev0` bump directly;
  `publish-dev-from-main` pushes the `.dev(N+1)` advance commit directly.
- `release/**`: `publish-release` pushes the version-bump commit and (when
  publishing a final) the changelog update.

Recommended ruleset for `release/**`:

- Require pull request review (bypassable by `github-actions[bot]`).
- Require status checks to pass (CI).
- No force-push, no deletion.

Docs publishing (`docs-publish.yml`) deploys to `docs/production` only when
a published GitHub Release is the latest final by semver, so older-branch
patches don't overwrite production docs.

## Docs behavior by release type

| Release type | docs/production | docs/staging |
|--------------|-----------------|--------------|
| RC (`v0.6.0rc0`) | unchanged | unchanged |
| Final minor (`v0.6.0`) | deployed | (main-push rebuilds as usual) |
| Patch on latest minor (`v0.6.1` after `v0.6.0`) | deployed | unchanged |
| Patch on older minor (`v0.5.1` after `v0.6.0`) | unchanged | unchanged |

Versioned docs (per-minor URL prefixes and a version switcher) would supersede
the latest-final-by-semver gate; not in scope here.
