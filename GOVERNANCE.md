# Governance

This document describes how the Mellea project is governed: the roles people hold, how decisions are made, and how code gets reviewed and merged.

For related topics, see:

- [CONTRIBUTING.md](CONTRIBUTING.md) — development setup, coding standards, and PR workflow
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) — behavioral norms and enforcement
- [RELEASE.md](RELEASE.md) — release cadence, versioning, and process

## Roles

| Role | Description | Examples |
|------|-------------|----------|
| **Contributor** | Anyone who opens issues, submits pull requests, or participates in discussions. No special access required. | External collaborators, first-time contributors |
| **Committer** | Has merge/write access to the repository. Member of [`@generative-computing/mellea-contributors`](https://github.com/orgs/generative-computing/teams/mellea-contributors). | Team members with commit rights |
| **Code Owner** | Domain expert listed in [`.github/CODEOWNERS`](.github/CODEOWNERS). Automatically requested as a reviewer for PRs touching their area. | Core library owners, intrinsics team |
| **Maintainer** | Overall project stewardship. Holds release authority and can grant committer access. | Project leads |

New committers are added by invitation from existing maintainers, based on sustained, high-quality contributions.

## Code Ownership

[`.github/CODEOWNERS`](.github/CODEOWNERS) defines required reviewers per area of the codebase. GitHub automatically requests reviews from the appropriate owners when a PR touches their files.

Current ownership zones:

| Path | Owners |
|------|--------|
| `mellea/core/` | @nrfulton, @jakelorocco |
| `mellea/formatters/granite/`, `test/formatters/granite/` | @generative-computing/mellea-intrinsics |
| Everything else | @generative-computing/mellea-contributors |

## PR Review & Merge Policy

### What approval means

A GitHub approval is equivalent to an Apache-style **LGTM** and **implies ownership of the change**. By approving, you are vouching that the change is correct and appropriate. Only approve PRs in areas where you have sufficient domain context — for example, core library changes should be approved by someone with core expertise.

### Requesting reviews

If you explicitly tag someone as a reviewer, you are asking for their review specifically. All explicitly requested reviewers should approve before the PR is merged. CODEOWNERS also enforces required reviewers automatically — PRs cannot be merged until all required code-owner reviews are satisfied.

### Reviewing

- **Use "Request Changes" to block** — if you have concerns that must be addressed, use "Request Changes" rather than a comment-only review. This prevents the PR from being merged on another reviewer's approval alone.
- **Respond to all review comments** — authors should resolve or reply to every review thread before merging. Don't let feedback get lost.
- **Re-request review after significant changes** — if you push substantial updates after a review round, re-request review from the same reviewers so they can verify their feedback was addressed.

### Merging

Once a PR has approvals from all requested and required reviewers:

- **Author has commit rights** — the author may merge or enable auto-merge.
- **Author is an external contributor** — the approver is responsible for merging.

### PR scope

Keep PRs focused on one logical change. Smaller, well-scoped PRs are easier to review, faster to merge, and safer to revert if needed.

### Merge queue

All PRs merge through GitHub's [merge queue](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/managing-a-merge-queue). After approval, PRs enter the queue, which runs CI against the latest `main` before landing. This ensures that every commit on `main` has passed the full test suite.

## Decision-Making

- **Day-to-day changes** (bug fixes, small features): lazy consensus via PR review. If a PR is approved and CI passes, it can be merged.
- **Significant changes** (new core abstractions, new backends, breaking API changes): open a GitHub issue for discussion before submitting a PR. These changes should receive broader review from maintainers and relevant code owners.
- **Disputes**: escalate to maintainers. If consensus cannot be reached, the project lead makes the final call.

## Releases

Maintainers hold release authority. The full release process — including cadence, versioning, and automation — is documented in [RELEASE.md](RELEASE.md).

## Communication

- **GitHub Issues and Pull Requests** — code-related decisions and technical discussion
- **GitHub Discussions** — broader topics, questions, and ideas

All community interactions are subject to the [Code of Conduct](CODE_OF_CONDUCT.md).
