#!/bin/bash
# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Cut a release branch off main and apply matched version bumps to both branches.
#
# Expected state before running:
#   - Checked out on main with a clean working tree
#   - pyproject.toml version matches X.Y.0.devN
#   - No existing tag v{X.Y.0rc0} or branch release/vX.Y
#
# Produces:
#   - release/vX.Y branch at X.Y.0rc0, pushed to origin
#   - main bumped to X.(Y+1).0.dev0, pushed to origin
#
# Env:
#   CONFIRM_MINOR (optional) — if set, must match X.Y derived from pyproject.

set -eu
set -x

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "${CURRENT_BRANCH}" != "main" ]; then
    >&2 echo "error: must be run from main, got ${CURRENT_BRANCH}"
    exit 2
fi

if [ -n "$(git status --porcelain)" ]; then
    >&2 echo "error: working tree is not clean"
    exit 2
fi

# Read the current version from pyproject.toml.
CURRENT_VERSION=$(uvx --from=toml-cli toml get --toml-path=pyproject.toml project.version)

# Expected shape: X.Y.0.devN
if ! [[ "${CURRENT_VERSION}" =~ ^([0-9]+)\.([0-9]+)\.0\.dev([0-9]+)$ ]]; then
    >&2 echo "error: pyproject version ${CURRENT_VERSION} does not match X.Y.0.devN"
    exit 2
fi
MAJOR="${BASH_REMATCH[1]}"
MINOR="${BASH_REMATCH[2]}"

if [ -n "${CONFIRM_MINOR:-}" ]; then
    if [ "${CONFIRM_MINOR}" != "${MAJOR}.${MINOR}" ]; then
        >&2 echo "error: CONFIRM_MINOR=${CONFIRM_MINOR} does not match pyproject minor ${MAJOR}.${MINOR}"
        exit 2
    fi
fi

RELEASE_BRANCH="release/v${MAJOR}.${MINOR}"
RC_VERSION="${MAJOR}.${MINOR}.0rc0"
RC_TAG="v${RC_VERSION}"
NEXT_MINOR=$((MINOR + 1))
NEXT_DEV_VERSION="${MAJOR}.${NEXT_MINOR}.0.dev0"

# Refuse if tag or branch already exists (local or remote).
git fetch origin --tags --prune
if git rev-parse --verify "refs/tags/${RC_TAG}" >/dev/null 2>&1; then
    >&2 echo "error: tag ${RC_TAG} already exists"
    exit 2
fi
if git rev-parse --verify "refs/heads/${RELEASE_BRANCH}" >/dev/null 2>&1 \
    || git rev-parse --verify "refs/remotes/origin/${RELEASE_BRANCH}" >/dev/null 2>&1; then
    >&2 echo "error: branch ${RELEASE_BRANCH} already exists"
    exit 2
fi

git config --global user.name 'github-actions[bot]'
git config --global user.email 'github-actions[bot]@users.noreply.github.com'

# Create the release branch and set the rc version there.
git checkout -b "${RELEASE_BRANCH}"
uvx --from=toml-cli toml set --toml-path=pyproject.toml project.version "${RC_VERSION}"
UV_FROZEN=0 uv lock --upgrade-package mellea
git add pyproject.toml uv.lock
git commit -m "release: cut v${MAJOR}.${MINOR} branch at ${RC_VERSION} [skip ci]"
git push origin "${RELEASE_BRANCH}"

# Publish rc0 via release.sh — tag-only when PUBLISH_PRERELEASES is disabled,
# full prerelease flow when enabled.
RELEASE_BRANCH="${RELEASE_BRANCH}" "$(dirname "$0")/release.sh"

# Back to main and bump to the next dev version.
git checkout main
uvx --from=toml-cli toml set --toml-path=pyproject.toml project.version "${NEXT_DEV_VERSION}"
UV_FROZEN=0 uv lock --upgrade-package mellea
git add pyproject.toml uv.lock
git commit -m "chore: bump main to ${NEXT_DEV_VERSION} [skip ci]"
git push origin main

set +x
echo ""
echo "Cut ${RELEASE_BRANCH} at ${RC_VERSION}"
echo "Bumped main to ${NEXT_DEV_VERSION}"
echo ""
echo "Next step: dispatch the Publish release workflow against ${RELEASE_BRANCH} with bump_type=rc to produce the next rc (rc1)."
