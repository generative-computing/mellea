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
#
# Prereleases (rc, .dev): push a git tag. PyPI upload is handled by pypi.yml
# when the tag push fires, gated on PUBLISH_PRERELEASES.
#
# Finals: create a GitHub Release (tag + Release object + generated notes),
# append to the changelog on the release branch, and open a sync PR to main
# with the changelog delta.

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

# Pull the version from pyproject.toml — authoritative after bump_version.py ran.
TARGET_VERSION=$(uvx --from=toml-cli toml get --toml-path=pyproject.toml project.version)
TARGET_TAG_NAME="v${TARGET_VERSION}"

# Detect prerelease shape (rc or .dev).
IS_PRERELEASE=0
if [[ "${TARGET_VERSION}" == *rc* ]] || [[ "${TARGET_VERSION}" == *.dev* ]]; then
    IS_PRERELEASE=1
fi

git config --global user.name 'github-actions[bot]'
git config --global user.email 'github-actions[bot]@users.noreply.github.com'

# Configure the remote with the token for pushes.
git remote set-url origin "https://x-access-token:${GH_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"

# Prerelease path: tag, push, dispatch pypi.yml. (Direct dispatch because
# GITHUB_TOKEN pushes don't fire downstream push: triggers; pypi.yml gates
# the upload on PUBLISH_PRERELEASES.)
if [ "${IS_PRERELEASE}" = "1" ]; then
    git tag "${TARGET_TAG_NAME}"
    git push origin "${TARGET_TAG_NAME}"
    gh workflow run pypi.yml --ref "${TARGET_TAG_NAME}"
    echo "Tagged prerelease ${TARGET_TAG_NAME}"
    exit 0
fi

# Final path. gh release create both tags and creates the Release object
# with notes generated against the previous Release.
gh release create "${TARGET_TAG_NAME}" \
    --target "${RELEASE_BRANCH}" \
    --generate-notes

# Dispatch follow-on workflows directly (GITHUB_TOKEN-authored release/tag
# events don't auto-trigger).
gh workflow run pypi.yml --ref "${TARGET_TAG_NAME}"
gh workflow run docs-publish.yml --field "release_tag=${TARGET_TAG_NAME}"

# Changelog sync PR is release-branch → main; skip it when publishing from
# main itself.
if [ "${RELEASE_BRANCH}" = "main" ]; then
    echo "Published ${TARGET_TAG_NAME} from main — skipping changelog sync PR"
    exit 0
fi

# Pull the generated notes back locally to update the changelog.
REL_NOTES=$(mktemp)
gh release view "${TARGET_TAG_NAME}" --json body -q ".body" >> "${REL_NOTES}"

# Build the updated changelog.
TMP_CHGLOG=$(mktemp)
RELEASE_URL="$(gh repo view --json url -q ".url")/releases/tag/${TARGET_TAG_NAME}"
printf "## [%s](%s) - %s\n\n" "${TARGET_TAG_NAME}" "${RELEASE_URL}" "$(date -Idate)" >> "${TMP_CHGLOG}"
cat "${REL_NOTES}" >> "${TMP_CHGLOG}"
if [ -f "${CHGLOG_FILE}" ]; then
    printf "\n" | cat - "${CHGLOG_FILE}" >> "${TMP_CHGLOG}"
fi
mv "${TMP_CHGLOG}" "${CHGLOG_FILE}"

# Commit the changelog update to the release branch and push it.
git add "${CHGLOG_FILE}"
git commit -m "docs: update changelog for ${TARGET_TAG_NAME} [skip ci]"
git push origin "${RELEASE_BRANCH}"

# Open a PR against main syncing just the changelog delta. Main is never
# pushed to directly from this script; branch protection applies normally.
SYNC_BRANCH="chore/changelog-sync-${TARGET_VERSION}"
git fetch origin main
# If the sync branch already exists on origin (e.g. a previous run got past
# the push but failed before opening the PR), reuse it so we don't silently
# discard prior commits. Otherwise branch fresh from origin/main.
if git rev-parse --verify "refs/remotes/origin/${SYNC_BRANCH}" >/dev/null 2>&1; then
    git checkout "${SYNC_BRANCH}"
    git reset --hard "origin/${SYNC_BRANCH}"
else
    git checkout -b "${SYNC_BRANCH}" origin/main
fi
# Pick just the changelog change from the commit we just made on the release branch.
git checkout "${RELEASE_BRANCH}" -- "${CHGLOG_FILE}"
git add "${CHGLOG_FILE}"
git commit -m "docs: sync changelog for ${TARGET_TAG_NAME}"
git push origin "${SYNC_BRANCH}"

gh pr create \
    --base main \
    --head "${SYNC_BRANCH}" \
    --title "docs: sync changelog for ${TARGET_TAG_NAME}" \
    --body "Automated changelog sync from \`${RELEASE_BRANCH}\` after publishing [${TARGET_TAG_NAME}](${RELEASE_URL}).

This PR brings the release-branch CHANGELOG entry back to main so the project root CHANGELOG remains the canonical history across all branches."
