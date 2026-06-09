# RELEASE.md

Mellea uses a release-branch workflow. Each minor release lives on a
long-lived `release/vX.Y` branch that carries the final release and any
later patches. `main` carries `.devN`-versioned work for the next minor.
Releases target a roughly 4-week cadence; patch releases happen as needed.

- [Making a minor release](#making-a-minor-release)
- [Making a patch release](#making-a-patch-release)
- [Troubleshooting](#troubleshooting)
- [Appendix](#appendix)

## Making a minor release

1. [Cut the release branch](#1-cut-the-release-branch)
2. [Stabilize on the release branch](#2-stabilize-on-the-release-branch)
3. [Publish the final](#3-publish-the-final)
4. [Sync the changelog back to main](#4-sync-the-changelog-back-to-main)

### 1. Cut the release branch

Run [Cut release branch](https://github.com/generative-computing/mellea/actions/workflows/cut-release-branch.yml)
against `main`.

- **Optional**: enter the expected `X.Y` (e.g. `0.6`) in `confirm_minor` as
  a sanity check. Leave blank to trust whatever is in `pyproject.toml`.

After this runs: `release/vX.Y` exists at `X.Y.0rc0`, and `main` has been
bumped to `X.(Y+1).0.dev0`.

### 2. Stabilize on the release branch

Land fixes via PRs targeting `release/vX.Y` (normal review + CI; protection
mirrors `main`). If the same change also belongs on `main`, open a
follow-up PR to `main` once the release-branch PR is merged.

> RC iteration during stabilization (`bump_type: rc`) is currently a no-op:
> the rc counter advances in `pyproject.toml` but nothing is published.
> See [rc cycling](#rc-cycling) in the appendix for what changes when
> prereleases are enabled.

### 3. Publish the final

Run [Publish release](https://github.com/generative-computing/mellea/actions/workflows/publish-release.yml)
against `release/vX.Y`.

- `bump_type`: `final`

After this runs: `vX.Y.0` is tagged, the GitHub Release exists, the PyPI
upload is done, and docs versioning is updated automatically. The
`snapshot-docs` job commits `versioned_docs/version-X.Y.0/` to `main` and
sets that version as the default. This commit triggers `docs-publish.yml`,
which redeploys production. Visitors land on `vX.Y.0` docs by default;
the version dropdown lets them switch to `main` (unreleased) or any prior
snapshot. No manual docs step is required.

### 4. Sync the changelog back to main

The publish workflow opens a PR titled `docs: sync changelog for vX.Y.0`
from `chore/changelog-sync-X.Y.0` to `main`. Review and merge it.

## Making a patch release

1. [Land fixes on the release branch](#1-land-fixes-on-the-release-branch)
2. [Publish the patch](#2-publish-the-patch)
3. [Sync the changelog back to main](#3-sync-the-changelog-back-to-main)

### 1. Land fixes on the release branch

PR targeting `release/vX.Y`. If the change also belongs on `main`, open a
follow-up PR to `main` once the release-branch PR is merged.

### 2. Publish the patch

Run [Publish release](https://github.com/generative-computing/mellea/actions/workflows/publish-release.yml)
against `release/vX.Y`.

- `bump_type`: `patch-final`

As with a minor final, `snapshot-docs` runs automatically and docs are
redeployed with the patch as the new default version.

> Patch rc iteration (`bump_type: patch-rc`) is currently a no-op for the
> same reason as minor rc iteration; see [rc cycling](#rc-cycling).

### 3. Sync the changelog back to main

Same as the minor flow: review and merge the auto-opened
`chore/changelog-sync-X.Y.Z` PR.

## Troubleshooting

### Retry a failed publish

`bump_type: none` re-runs `publish-release` against whatever version is
already in `pyproject.toml`, skipping the version-bump step. Useful when a
previous run failed after the bump committed but before the publish
completed.

The retry is a "skip what's done, finish what isn't" path — it does not
validate existing artifacts. If a prior run produced a tag at the wrong
commit, a Release with stale notes, or a sync PR with a stale body, delete
the bad artifact (`gh release delete`, `git push --delete`, `gh pr close`)
before re-dispatching. Retry is for resuming after a partial failure, not
for fixing a corrupted result.

### Cutting a major release

The cut-release workflow always bumps `main` by one minor. To cut a major
(e.g. `1.0.0` from a `0.x` line):

1. Land a regular PR on `main` setting `pyproject.toml` to
   `(X+1).0.0.dev0`.
2. Dispatch [Cut release branch](https://github.com/generative-computing/mellea/actions/workflows/cut-release-branch.yml)
   as usual.

### Ad-hoc dev publish from main

Used when someone needs a tagged point-in-time artifact of `main` outside
the normal release cadence. With `PUBLISH_PRERELEASES=false` (the default)
this workflow is a no-op for publishing; the only effect is that `main`'s
`.devN` counter advances.

Run [Publish dev release from main](https://github.com/generative-computing/mellea/actions/workflows/publish-dev-from-main.yml)
against `main`. With the flag enabled, the workflow tags `main` HEAD,
creates a prerelease GitHub Release, and uploads to PyPI. Either way, it
then bumps `main` from `X.Y.Z.devN` to `X.Y.Z.dev(N+1)` and pushes.

`main`'s `pyproject.toml` always reflects the version that the next
dispatch will publish.

---

# Appendix

## Versioning

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
  publication runs; never during routine commits.
- Release branches always carry `X.Y.Zrc?` or `X.Y.Z`.
- Prereleases (`rcN`, `.devN`) are tagged, uploaded to PyPI, and given a
  prerelease-marked GitHub Release only when
  [`PUBLISH_PRERELEASES`](#the-publish_prereleases-flag) is `true`. With the
  default (`false`), the version bump commits to the branch but no tag, no
  PyPI upload, and no Release. Prerelease Releases use `--prerelease` so
  they don't appear as "latest" on GitHub.

## Workflow inventory

| Workflow | Purpose |
|----------|---------|
| [`cut-release-branch`](.github/workflows/cut-release-branch.yml) | Cut `release/vX.Y` from `main`, publish `X.Y.0rc0`, bump `main` to the next `.dev0` |
| [`publish-release`](.github/workflows/publish-release.yml) | Publish a release (rc, final, patch-rc, patch-final, or retry a failed publish) |
| [`publish-dev-from-main`](.github/workflows/publish-dev-from-main.yml) | Iterate main's `.devN` counter and (when enabled) publish a dev release |

All three are `workflow_dispatch`-only and run from the GitHub Actions UI.

Whether any given prerelease (`rc`, `dev`) produces a PyPI artifact depends
on the [`PUBLISH_PRERELEASES`](#the-publish_prereleases-flag) flag.

## Branch protection

All three write-capable workflows authenticate via `secrets.GITHUB_TOKEN`,
declaring scopes via inline `permissions:` blocks. The `GITHUB_TOKEN`
identity is `github-actions[bot]`, configured as a bypass actor on the
`main` and `release/**` rulesets so workflows can push directly:

- `main`: `cut-release-branch` pushes the `X.(Y+1).0.dev0` bump;
  `publish-dev-from-main` pushes the `.dev(N+1)` advance commit.
- `release/**`: `publish-release` pushes the version-bump commit and (for
  finals) the changelog update.

The `release/**` ruleset otherwise mirrors `main`: PR review required,
status checks (CI) required, no force-push, no deletion.

## Release branch retention

**Release branches are never deleted.** GitHub Releases pin to specific
commits on each branch, so pruning a branch would orphan those references
and break `git checkout v0.4.2` semantics. Old `release/v0.3`,
`release/v0.4`, etc. stay around indefinitely.

## Docs behavior by release type

For finals and patch-finals, the `snapshot-docs` job in `publish-release.yml`
commits a versioned snapshot of `main`'s docs to `main` immediately after the
GitHub Release is created, then explicitly dispatches `docs-publish.yml` via
`workflow_dispatch`. (A `GITHUB_TOKEN` push cannot trigger a new workflow run,
so the dispatch is explicit.) This deploys production with the new version as
the default. Visitors see the released docs by default; a version dropdown in the
navbar lets them switch to `main` (unreleased) or any prior snapshot.

`docs-publish.yml` only deploys to production when the triggering release is the
latest final by semver, so older-branch patches don't overwrite production docs.

| Release type | docs/production default version | Dropdown includes |
|--------------|----------------------------------|-------------------|
| RC (`v0.6.0rc0`) | unchanged | unchanged |
| Final X.Y release (`v0.6.0`) | `v0.6.0` | `v0.6.0`, prior snapshots, `main` |
| Patch on latest X.Y (`v0.6.1` after `v0.6.0`) | `v0.6.1` | all snapshots, `main` |
| Patch on older X.Y (`v0.5.1` after `v0.6.0`) | unchanged | unchanged |

## RC cycling

`bump_type: rc` (minor) and `bump_type: patch-rc` (patch) iterate the rc
counter on the release branch:

- **`PUBLISH_PRERELEASES=false`** (current default): the rc counter
  advances in `pyproject.toml`, but no tag is pushed, no Release is
  created, and nothing reaches PyPI. The rc step is effectively a no-op
  during stabilization, so the documented flow above goes straight from
  cut to final.
- **`PUBLISH_PRERELEASES=true`** (future): each `rc` dispatch tags
  `vX.Y.ZrcN`, creates a `--prerelease` GitHub Release with notes diffed
  against the previous rc, and uploads to PyPI.

When `PUBLISH_PRERELEASES` is enabled, the rc step is a real publish and
fits between [Stabilize](#2-stabilize-on-the-release-branch) and
[Publish the final](#3-publish-the-final): dispatch
[Publish release](https://github.com/generative-computing/mellea/actions/workflows/publish-release.yml) with
`bump_type: rc` (or `patch-rc`) as many times as needed.

## The `PUBLISH_PRERELEASES` flag

`PUBLISH_PRERELEASES` is a forward-looking feature flag. It exists so the
project can start publishing public prerelease artifacts later (likely
post-1.0) without a code change. It defaults to `false` and is expected to
stay `false` for the foreseeable future; the rest of this section
documents both states for when that changes.

The flag is a repo variable that gates whether prereleases are tagged,
uploaded to PyPI, and given a GitHub Release.

| `PUBLISH_PRERELEASES` | rc / dev | Finals |
|-----------------------|----------|--------|
| `false` (current) | version bump committed; no tag, no Release, no PyPI | tag + GitHub Release + PyPI + changelog entry + sync PR |
| `true` (future) | tag + prerelease GitHub Release + PyPI | tag + GitHub Release + PyPI + changelog entry + sync PR |

While `false`, prereleases stay branch-local — the bump commit identifies
the version but there is no immutable tag pointer. Users who need a
specific prerelease can install from a branch SHA
(`pip install git+https://github.com/generative-computing/mellea@<sha>`).

When the flag is eventually enabled, every rc and dev produces a
`--prerelease`-marked GitHub Release. Notes are incremental —
`0.6.0rc2` diffs against `0.6.0rc1`, so testers see "what changed in this
rc" without re-reading the full cycle. The cumulative view ("everything
in 0.6") shows up on the final's Release page, which diffs against the
previous final. Prerelease Releases never become the repo's "latest"
release.

Finals always follow the full release flow regardless of the flag.
