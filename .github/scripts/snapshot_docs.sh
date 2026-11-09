#!/bin/bash
# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Build the docs snapshot for a final release from the current checkout (the
# release tag) and open a sync PR adding it to main.
#
# Env:
#   VERSION           (required) — released version, X.Y.Z
#   GH_TOKEN          (required) — for git push and gh
#   GITHUB_REPOSITORY (required) — owner/repo

set -euo pipefail
set -x

for var in VERSION GH_TOKEN GITHUB_REPOSITORY; do
    if [ -z "${!var:-}" ]; then
        >&2 echo "error: ${var} env var is required"
        exit 2
    fi
done

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "${REPO_ROOT}"

SNAPSHOT_DOCS="docs/versioned_docs/version-${VERSION}"
SNAPSHOT_SIDEBAR="docs/versioned_sidebars/version-${VERSION}-sidebars.json"

uv sync --frozen --all-extras --group dev
uv run python tooling/docs-autogen/build.py --version "${VERSION}"
(cd docs && npm ci && npm run docusaurus -- docs:version "${VERSION}")

FINALS=$(gh api "repos/${GITHUB_REPOSITORY}/releases" --paginate \
    --jq '.[] | select(.prerelease == false and .draft == false) | .tag_name | ltrimstr("v")')
LATEST_FINAL=$(printf '%s\n%s\n' "${FINALS}" "${VERSION}" | sort -V | tail -1)

git config user.name 'github-actions[bot]'
git config user.email 'github-actions[bot]@users.noreply.github.com'
git remote set-url origin "https://x-access-token:${GH_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"

SYNC_BRANCH="chore/docs-snapshot-${VERSION}"
SYNC_DIR=$(mktemp -d)

git fetch origin "+refs/heads/main:refs/remotes/origin/main"
if git ls-remote --exit-code --heads origin "${SYNC_BRANCH}" >/dev/null; then
    git fetch origin "+refs/heads/${SYNC_BRANCH}:refs/remotes/origin/${SYNC_BRANCH}"
    git worktree add -B "${SYNC_BRANCH}" "${SYNC_DIR}" "origin/${SYNC_BRANCH}"
else
    git worktree add -B "${SYNC_BRANCH}" "${SYNC_DIR}" origin/main
fi

rm -rf "${SYNC_DIR:?}/${SNAPSHOT_DOCS}"
mkdir -p "${SYNC_DIR}/docs/versioned_docs" "${SYNC_DIR}/docs/versioned_sidebars"
cp -R "${SNAPSHOT_DOCS}" "${SYNC_DIR}/docs/versioned_docs/"
cp "${SNAPSHOT_SIDEBAR}" "${SYNC_DIR}/docs/versioned_sidebars/"

cd "${SYNC_DIR}"
VERSIONS_TMP=$(mktemp)
jq --arg v "${VERSION}" \
    'if index($v) then . else . + [$v] end | sort_by(split(".") | map(tonumber)) | reverse' \
    docs/versions.json > "${VERSIONS_TMP}"
mv "${VERSIONS_TMP}" docs/versions.json
if [ "${LATEST_FINAL}" = "${VERSION}" ]; then
    node .github/scripts/set-last-version.mjs "${VERSION}"
fi

git add "${SNAPSHOT_DOCS}" "${SNAPSHOT_SIDEBAR}" docs/versions.json docs/docusaurus.config.ts
if ! git diff --cached --quiet; then
    git commit -m "docs: snapshot ${VERSION}"
    git push origin "${SYNC_BRANCH}"
fi

EXISTING_PR=$(gh pr list --head "${SYNC_BRANCH}" --base main --state open --json number --jq '.[0].number // ""')
if [ -z "${EXISTING_PR}" ]; then
    RELEASE_URL="https://github.com/${GITHUB_REPOSITORY}/releases/tag/v${VERSION}"
    gh pr create \
        --base main \
        --head "${SYNC_BRANCH}" \
        --title "docs: snapshot ${VERSION}" \
        --body "Automated docs snapshot for [v${VERSION}](${RELEASE_URL}), built from the \`v${VERSION}\` tag.

Adds \`versioned_docs/version-${VERSION}/\` and its sidebar to the docs version list. Merging this PR deploys the docs site."
fi
