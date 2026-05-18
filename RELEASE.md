# RELEASE.md

## Overview

Mellea uses a release-branch workflow. Every release has a long-lived
`release/vX.Y` branch that carries release candidates, the final release,
and any subsequent patch releases. `main` carries `.dev`-versioned work for
the next release.

This gives each release a frozen codebase and keeps release publishing
resilient to concurrent merges on `main`.

## Release Cadence

Releases target a roughly 4-week cadence. Patch releases happen as needed.

## Cutting a release

To ship `X.Y.0`:

1. Dispatch **Cut release branch** against `main` — creates `release/vX.Y` at
   `X.Y.0rc0` and bumps `main` to `X.(Y+1).0.dev0`.
2. Stabilize on `release/vX.Y` by landing fixes via PRs targeting that branch
   (open a separate followup PR to port the fix to `main` if it also belongs
   there).
3. Dispatch **Publish release** against `release/vX.Y` with `bump_type: rc` to
   produce each rc; repeat as needed.
4. Dispatch **Publish release** against `release/vX.Y` with `bump_type: final`
   to ship `X.Y.0`.

The per-step detail follows.

### 1. Cut the release branch

When `main` is ready to freeze for the next release:

1. Go to **Actions → Cut release branch → Run workflow**.
2. Optionally enter the expected X.Y (e.g. `0.6`) in `confirm_minor` as a
   safety check. Leave blank to trust whatever is in `pyproject.toml` on
   `main`.
3. Run.

The workflow:

- Verifies `pyproject.toml` on `main` matches `X.Y.0.devN`.
- Creates `release/vX.Y` with version set to `X.Y.0rc0`.
- Publishes `X.Y.0rc0` per the [`PUBLISH_PRERELEASES`](#b-the-publish_prereleases-flag)
  flag — by default the version bump is committed to the release branch with
  no tag and no PyPI upload; with the flag enabled, also tags and uploads.
- Pushes `main` with version bumped to `X.(Y+1).0.dev0`. The push goes through
  `github-actions[bot]`, which has bypass-actor permission on `main`'s ruleset
  (see [Appendix D](#d-branch-protection)).

To cut a major release (e.g. `1.0.0` from a `0.x` line), first land a regular
PR on `main` setting `pyproject.toml` to `(X+1).0.0.dev0`, then dispatch
**Cut release branch** as above. The workflow always bumps `main` by one
minor; the major bump itself is a manual step performed beforehand.

### 2. Stabilize on the release branch

Stabilization fixes land on `release/vX.Y` via normal PRs targeting that
branch. Branch protection applies the same way as `main`: review + CI
required.

If the same fix also belongs on `main`, open a separate followup PR — usually
after the release-branch PR merges, but the order can be flipped if it makes
review easier. Either direction works; the only constraint is that both
branches end up carrying the change so it doesn't regress in the next
release.

To publish each rc:

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

### 3. Promote to final

When testing on an rc is complete:

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

1. Land the fix on `release/vX.Y` via a PR targeting that branch (open a
   separate followup PR to port to `main` if the fix also belongs there).
2. **Publish release** against `release/vX.Y` with `bump_type: patch-rc`.
   Produces e.g. `v0.6.1rc0`.
3. Test.
4. Repeat **Publish release** with `bump_type: patch-rc` for additional rcs
   if needed.
5. **Publish release** with `bump_type: patch-final` to promote to `v0.6.1`.

## Dev release from main

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

`bump_type: none` re-runs `publish-release` against whatever version is
currently in `pyproject.toml`, skipping the version-bump step. Useful when
a previous run failed after the bump committed but before the publish
completed.

The retry is a "skip what's done, finish what isn't" path — it does not
validate existing artifacts. If a prior run produced a tag pointing at the
wrong commit, a Release with stale notes, or a sync PR with a stale body,
delete the bad artifact (`gh release delete`, `git push --delete`,
`gh pr close`) before re-dispatching. Retry is for resuming after a partial
failure, not for fixing a corrupted prior result.

---

# Appendix

## A. Versioning

Versions follow **[PEP 440](https://peps.python.org/pep-0440/)** (which is
compatible with SemVer for final releases).

| Phase | Branch | Version example | Tag |
|-------|--------|-----------------|-----|
| Dev on main | `main` | `0.6.0.dev0` | (untagged) |
| Release branch cut | `release/v0.6` | `0.6.0rc0` | `v0.6.0rc0` if `PUBLISH_PRERELEASES`, else untagged |
| Further RCs | `release/v0.6` | `0.6.0rc1`, `rc2`, … | `v0.6.0rcN` if `PUBLISH_PRERELEASES`, else untagged |
| Final X.Y release | `release/v0.6` | `0.6.0` | `v0.6.0` |
| Patch RC | `release/v0.6` | `0.6.1rc0` | `v0.6.1rc0` if `PUBLISH_PRERELEASES`, else untagged |
| Patch final | `release/v0.6` | `0.6.1` | `v0.6.1` |
| Next dev on main | `main` | `0.7.0.dev0` | (untagged) |

Invariants:

- `main` always carries `X.Y.0.devN`. `main` is tagged only when a dev
  publication runs (`publish-dev-from-main` workflow); never during routine
  commits.
- Release branches always carry `X.Y.Zrc?` or `X.Y.Z`.
- Prereleases (`rcN`, `.devN`) are tagged, uploaded to PyPI, and given a
  prerelease-marked GitHub Release only when the `PUBLISH_PRERELEASES` repo
  variable is `true` (see [Appendix B](#b-the-publish_prereleases-flag)).
  With the default (`false`), the version bump commits to the branch but no
  tag, no PyPI upload, and no Release. Prerelease Releases use `--prerelease`
  so they don't appear as the repo's "latest" release on GitHub.

## B. The `PUBLISH_PRERELEASES` flag

`PUBLISH_PRERELEASES` is a forward-looking feature flag. It exists so the
project can start publishing public prerelease artifacts later (likely
post-1.0) without a code change. It defaults to `false` and is expected to
stay `false` for the foreseeable future; the rest of this section documents
both states for when that changes.

The flag is a repo variable that gates whether prereleases are tagged,
uploaded to PyPI, and given a GitHub Release.

| `PUBLISH_PRERELEASES` | rc / dev | Finals |
|-----------------------|----------|--------|
| `false` (current) | version bump committed; no tag, no Release, no PyPI | tag + GitHub Release + PyPI + changelog entry + sync PR |
| `true` (future) | tag + prerelease GitHub Release + PyPI | tag + GitHub Release + PyPI + changelog entry + sync PR |

While `false`, prereleases stay branch-local — the bump commit identifies
the version but there is no immutable tag pointer. Users who need a specific
prerelease can install from a branch SHA
(`pip install git+https://github.com/generative-computing/mellea@<sha>`).

When the flag is eventually enabled, every rc and dev would produce a
`--prerelease`-marked GitHub Release. Notes would be incremental —
`0.6.0rc2` diffs against `0.6.0rc1`, so testers see "what changed in this
rc" without re-reading the full cycle. The cumulative view ("everything in
0.6") shows up on the final's Release page, which diffs against the previous
final. Prerelease Releases never become the repo's "latest" release.

Finals always follow the full release flow regardless of the flag.

## C. Workflow inventory

| Workflow | Purpose |
|----------|---------|
| `cut-release-branch` | Cut `release/vX.Y` from `main`, publish `X.Y.0rc0`, bump `main` to the next `.dev0` |
| `publish-release` | Publish a release (rc, final, patch-rc, patch-final, or retry a failed publish) |
| `publish-dev-from-main` | Iterate main's `.devN` counter and publish a dev release |

All three are `workflow_dispatch`-only and run from the GitHub Actions UI.

Whether any given prerelease (`rc`, `dev`) produces a PyPI artifact depends
on the [`PUBLISH_PRERELEASES`](#b-the-publish_prereleases-flag) flag.

## D. Branch protection

All three write-capable workflows authenticate via `secrets.GITHUB_TOKEN`,
declaring scopes via inline `permissions:` blocks. The `GITHUB_TOKEN`
identity is `github-actions[bot]`, which is configured as a bypass actor on
the `main` and `release/**` rulesets so workflows can push directly:

- `main`: `cut-release-branch` pushes the `X.(Y+1).0.dev0` bump;
  `publish-dev-from-main` pushes the `.dev(N+1)` advance commit.
- `release/**`: `publish-release` pushes the version-bump commit and (for
  finals) the changelog update.

The `release/**` ruleset otherwise mirrors `main`: PR review required,
status checks (CI) required, no force-push, no deletion.

## E. Release branch retention

**Release branches are never deleted.** GitHub Releases pin to specific
commits on each branch, so pruning a branch would orphan those references and
break `git checkout v0.4.2` semantics. Old `release/v0.3`, `release/v0.4`, etc.
stay around indefinitely.

## F. Docs behavior by release type

Docs publishing (`docs-publish.yml`) deploys to `docs/production` only when
a published GitHub Release is the latest final by semver, so older-branch
patches don't overwrite production docs.

| Release type | docs/production | docs/staging |
|--------------|-----------------|--------------|
| RC (`v0.6.0rc0`) | unchanged | unchanged |
| Final X.Y release (`v0.6.0`) | deployed | (main-push rebuilds as usual) |
| Patch on latest X.Y (`v0.6.1` after `v0.6.0`) | deployed | unchanged |
| Patch on older X.Y (`v0.5.1` after `v0.6.0`) | unchanged | unchanged |
