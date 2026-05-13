#!/bin/bash
# Cherry-pick one or more commits from main onto a release branch, preserving
# original merge order via topological sort.
#
# Usage:
#   cherry_pick_to_release.sh <release-branch> <sha> [<sha> ...]
#
# Example:
#   cherry_pick_to_release.sh release/v0.6 abc1234 def5678
#
# Behavior:
#   1. Checks out the target release branch and fetches origin.
#   2. Validates every SHA is an ancestor of origin/main and not already on
#      the release branch.
#   3. Topologically sorts the provided SHAs by their position in
#      git log origin/main (oldest first), so the operator can pass SHAs in
#      any order and they apply in original merge order.
#   4. Runs git cherry-pick -x for each SHA in sorted order.
#   5. On conflict, stops and prints a resolution playbook.
#   6. On success, either pushes to origin (when AUTO_PUSH=1, set by the
#      CI workflow) or prints the push command for the operator to run.

set -eu

if [ "$#" -lt 2 ]; then
    >&2 echo "usage: $0 <release-branch> <sha> [<sha> ...]"
    exit 2
fi

RELEASE_BRANCH="$1"
shift

if ! [[ "${RELEASE_BRANCH}" =~ ^release/v ]]; then
    >&2 echo "error: target branch ${RELEASE_BRANCH} does not match release/v*"
    exit 2
fi

if [ -n "$(git status --porcelain)" ]; then
    >&2 echo "error: working tree is not clean"
    exit 2
fi

git fetch origin --tags --prune

# Ensure the release branch exists on origin.
if ! git rev-parse --verify "refs/remotes/origin/${RELEASE_BRANCH}" >/dev/null 2>&1; then
    >&2 echo "error: origin/${RELEASE_BRANCH} does not exist"
    exit 2
fi

# Checkout the release branch tracking origin.
if git rev-parse --verify "refs/heads/${RELEASE_BRANCH}" >/dev/null 2>&1; then
    git checkout "${RELEASE_BRANCH}"
    git reset --hard "origin/${RELEASE_BRANCH}"
else
    git checkout -b "${RELEASE_BRANCH}" "origin/${RELEASE_BRANCH}"
fi

# Validate each SHA:
#   - Must resolve to a commit.
#   - Must be an ancestor of origin/main (ie, merged).
#   - Must NOT be already on the release branch.
for sha in "$@"; do
    if ! git rev-parse --verify "${sha}^{commit}" >/dev/null 2>&1; then
        >&2 echo "error: ${sha} is not a commit"
        exit 2
    fi
    if ! git merge-base --is-ancestor "${sha}" origin/main; then
        >&2 echo "error: ${sha} is not an ancestor of origin/main (not yet merged?)"
        exit 2
    fi
    if git merge-base --is-ancestor "${sha}" HEAD; then
        >&2 echo "error: ${sha} is already on ${RELEASE_BRANCH}"
        exit 2
    fi
done

# Topologically sort SHAs by their position in git log origin/main (oldest first).
# git log --reverse lists commits in chronological (merge) order; we filter to
# just the SHAs we care about by streaming through the log and printing only
# matches.
SORTED_SHAS=$(
    git log --reverse --format='%H' origin/main \
        | while read -r commit; do
            for sha in "$@"; do
                short=$(git rev-parse --short "${sha}")
                full=$(git rev-parse "${sha}")
                if [ "${commit}" = "${full}" ]; then
                    echo "${full}"
                    break
                fi
            done
        done
)

if [ -z "${SORTED_SHAS}" ]; then
    >&2 echo "error: no SHAs resolved to commits on origin/main (internal error)"
    exit 2
fi

echo "Cherry-picking (in merge order):"
echo "${SORTED_SHAS}" | while read -r sha; do
    echo "  $(git log -1 --format='%h %s' "${sha}")"
done

# Apply the cherry-picks.
CONFLICTED=0
while read -r sha; do
    if ! git cherry-pick -x "${sha}"; then
        CONFLICTED=1
        break
    fi
done <<< "${SORTED_SHAS}"

if [ "${CONFLICTED}" -eq 1 ]; then
    cat >&2 <<EOF

=============================================================================
Cherry-pick hit a conflict on $(git rev-parse --short CHERRY_PICK_HEAD 2>/dev/null || echo "a commit").

To resolve locally:
  1. Clone the repo (if you are not already local) and check out ${RELEASE_BRANCH}.
  2. Re-run this script with the same SHAs to reach the same state.
  3. Resolve the conflicted files, then:
       git add <resolved-files>
       git cherry-pick --continue
  4. Push to origin (requires push access / bypass rights):
       git push origin ${RELEASE_BRANCH}

Abort with:
  git cherry-pick --abort
=============================================================================
EOF
    exit 1
fi

if [ "${AUTO_PUSH:-0}" = "1" ]; then
    git push origin "${RELEASE_BRANCH}"
    echo ""
    echo "Pushed to origin/${RELEASE_BRANCH}"
else
    echo ""
    echo "Cherry-picks applied locally on ${RELEASE_BRANCH}."
    echo "To push:  git push origin ${RELEASE_BRANCH}"
fi
