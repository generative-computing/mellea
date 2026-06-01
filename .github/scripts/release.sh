#!/bin/bash
# Publish a release from the currently checked-out branch. Uses the version
# already written to pyproject.toml; does not modify it.
#
# Env:
#   RELEASE_BRANCH (required) — branch being published (e.g. release/v0.6).
#                              Guards against accidentally releasing from main.
#   GH_TOKEN       (required) — for gh release create / gh pr create
#   GITHUB_REPOSITORY (required) — owner/repo, used for origin URL and links
#   CHGLOG_FILE    (optional) — path to changelog (default: CHANGELOG.md)
#   ALLOW_MAIN_RELEASE=1 (optional) — bypass the main-branch guard
#   PUBLISH_PRERELEASES (optional) — "true" to tag prereleases and upload to
#                                    PyPI; default "false" leaves the bump
#                                    commit as the only artifact.
#
# Prereleases (rc, .dev): when PUBLISH_PRERELEASES=true, push a git tag,
# create a prerelease GitHub Release with notes diffed against the previous
# tag (incremental), and dispatch pypi.yml. Otherwise no tag, no Release,
# no PyPI upload — the version-bump commit on the branch is the only
# artifact.
#
# Finals: create a GitHub Release (tag + Release object + generated notes
# diffed against the previous final), append to the changelog on the release
# branch, and open a sync PR to main with the changelog delta.
#
# All write steps are idempotent so `bump_type=none` retries are safe after
# a partial failure.

set -euo pipefail
set -x

if [ -z "${RELEASE_BRANCH:-}" ]; then
    >&2 echo "error: RELEASE_BRANCH env var is required"
    exit 2
fi
if [ "${RELEASE_BRANCH}" = "main" ] && [ "${ALLOW_MAIN_RELEASE:-0}" != "1" ]; then
    >&2 echo "error: refusing to release from main; dispatch against a release/v* branch"
    exit 2
fi

CHGLOG_FILE="${CHGLOG_FILE:-CHANGELOG.md}"
PUBLISH_PRERELEASES="${PUBLISH_PRERELEASES:-false}"

TARGET_VERSION=$(uvx --from=toml-cli toml get --toml-path=pyproject.toml project.version)
TARGET_TAG_NAME="v${TARGET_VERSION}"

IS_PRERELEASE=0
if [[ "${TARGET_VERSION}" == *rc* ]] || [[ "${TARGET_VERSION}" == *.dev* ]]; then
    IS_PRERELEASE=1
fi

# START_TAG drives --notes-start-tag. Prereleases and finals serve
# different audiences and use different selection rules:
#   - Prereleases (rc, dev): incremental — most recent reachable tag
#     from HEAD. Steady state: previous rc on the release branch, previous
#     dev on main. First-in-cycle (no reachable prior tag) leaves START_TAG
#     empty so gh's default fills in. Testers want "what's new in rc2
#     since rc1"; the cumulative view lives on the final's Release.
#   - Finals: cumulative — previous final regardless of intermediate
#     prereleases. Selection by version shape:
#       * Patch (Z>0): git describe excluding rc/dev (reachable on this
#         branch; excludes a parallel-branch v0.7.0).
#       * Minor (Z=0, Y>0): previous final is on a parallel release branch
#         and isn't reachable; pick highest v(M).(Y-1).* by semver.
#       * First minor (X.0.0): empty → gh's default ("latest non-prerelease").
START_TAG=""
if [ "${IS_PRERELEASE}" = "1" ]; then
    START_TAG=$(git describe --tags --abbrev=0 HEAD 2>/dev/null || true)
elif [[ "${TARGET_VERSION}" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)$ ]]; then
    TGT_MAJOR="${BASH_REMATCH[1]}"
    TGT_MINOR="${BASH_REMATCH[2]}"
    TGT_PATCH="${BASH_REMATCH[3]}"
    if [ "${TGT_PATCH}" -gt 0 ]; then
        START_TAG=$(git describe --tags --abbrev=0 \
            --match 'v*' --exclude '*rc*' --exclude '*.dev*' \
            HEAD 2>/dev/null || true)
    elif [ "${TGT_MINOR}" -gt 0 ]; then
        PREV_MINOR=$((TGT_MINOR - 1))
        START_TAG=$(git tag -l "v${TGT_MAJOR}.${PREV_MINOR}.*" \
            | grep -vE '(rc|\.dev)' \
            | sort -V \
            | tail -1)
    fi
fi

git config --global user.name 'github-actions[bot]'
git config --global user.email 'github-actions[bot]@users.noreply.github.com'

git remote set-url origin "https://x-access-token:${GH_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"

if [ "${IS_PRERELEASE}" = "1" ]; then
    if [ "${PUBLISH_PRERELEASES}" = "true" ]; then
        if ! git rev-parse "refs/tags/${TARGET_TAG_NAME}" >/dev/null 2>&1; then
            git tag "${TARGET_TAG_NAME}"
            git push origin "${TARGET_TAG_NAME}"
        fi
        if ! gh release view "${TARGET_TAG_NAME}" >/dev/null 2>&1; then
            if [ -n "${START_TAG}" ]; then
                gh release create "${TARGET_TAG_NAME}" \
                    --target "${RELEASE_BRANCH}" \
                    --prerelease \
                    --generate-notes \
                    --notes-start-tag "${START_TAG}"
            else
                gh release create "${TARGET_TAG_NAME}" \
                    --target "${RELEASE_BRANCH}" \
                    --prerelease \
                    --generate-notes
            fi
        fi
        gh workflow run pypi.yml --ref "${TARGET_TAG_NAME}"
        echo "Published prerelease ${TARGET_TAG_NAME} (tag + Release + PyPI)"
    else
        echo "Prerelease ${TARGET_TAG_NAME}: PUBLISH_PRERELEASES=false, skipping tag and PyPI upload"
    fi
    exit 0
fi

if ! gh release view "${TARGET_TAG_NAME}" >/dev/null 2>&1; then
    if [ -n "${START_TAG}" ]; then
        gh release create "${TARGET_TAG_NAME}" \
            --target "${RELEASE_BRANCH}" \
            --generate-notes \
            --notes-start-tag "${START_TAG}"
    else
        gh release create "${TARGET_TAG_NAME}" \
            --target "${RELEASE_BRANCH}" \
            --generate-notes
    fi
fi

# Dispatch follow-on workflows directly (GITHUB_TOKEN-authored release/tag
# events don't auto-trigger).
# Note: docs-publish.yml is NOT dispatched here for finals — the snapshot-docs
# job in publish-release.yml commits versioned docs to main, which triggers
# docs-publish.yml automatically via the docs/** path filter.
gh workflow run pypi.yml --ref "${TARGET_TAG_NAME}"

if [ "${RELEASE_BRANCH}" = "main" ]; then
    echo "Published ${TARGET_TAG_NAME} from main — skipping changelog sync PR"
    exit 0
fi

REL_NOTES=$(mktemp)
gh release view "${TARGET_TAG_NAME}" --json body -q ".body" >> "${REL_NOTES}"

TMP_CHGLOG=$(mktemp)
RELEASE_URL="$(gh repo view --json url -q ".url")/releases/tag/${TARGET_TAG_NAME}"
printf "## [%s](%s) - %s\n\n" "${TARGET_TAG_NAME}" "${RELEASE_URL}" "$(date -Idate)" >> "${TMP_CHGLOG}"
cat "${REL_NOTES}" >> "${TMP_CHGLOG}"
if [ -f "${CHGLOG_FILE}" ]; then
    printf "\n" | cat - "${CHGLOG_FILE}" >> "${TMP_CHGLOG}"
fi
mv "${TMP_CHGLOG}" "${CHGLOG_FILE}"

# idempotent: skip if a prior run committed the same content.
git add "${CHGLOG_FILE}"
if ! git diff --cached --quiet; then
    git commit -m "docs: update changelog for ${TARGET_TAG_NAME} [skip ci]"
    git push origin "${RELEASE_BRANCH}"
fi

# Main is never pushed to directly from this script; branch protection
# applies normally.
SYNC_BRANCH="chore/changelog-sync-${TARGET_VERSION}"
git fetch origin main
# Reuse an existing sync branch on origin so a partial prior run's commits
# aren't silently discarded.
if git rev-parse --verify "refs/remotes/origin/${SYNC_BRANCH}" >/dev/null 2>&1; then
    git checkout "${SYNC_BRANCH}"
    git reset --hard "origin/${SYNC_BRANCH}"
else
    git checkout -b "${SYNC_BRANCH}" origin/main
fi
git checkout "${RELEASE_BRANCH}" -- "${CHGLOG_FILE}"
git add "${CHGLOG_FILE}"
if ! git diff --cached --quiet; then
    git commit -m "docs: sync changelog for ${TARGET_TAG_NAME}"
    git push origin "${SYNC_BRANCH}"
fi

EXISTING_PR=$(gh pr list --head "${SYNC_BRANCH}" --base main --state open --json number --jq '.[0].number // ""')
if [ -z "${EXISTING_PR}" ]; then
    gh pr create \
        --base main \
        --head "${SYNC_BRANCH}" \
        --title "docs: sync changelog for ${TARGET_TAG_NAME}" \
        --body "Automated changelog sync from \`${RELEASE_BRANCH}\` after publishing [${TARGET_TAG_NAME}](${RELEASE_URL}).

This PR brings the release-branch CHANGELOG entry back to main so the project root CHANGELOG remains the canonical history across all branches."
fi

# GITHUB_TOKEN-authored pull_request events don't trigger workflows, so
# required status checks on main would block the sync PR forever. Dispatch
# CI explicitly against the sync branch.
gh workflow run ci.yml --ref "${SYNC_BRANCH}"
