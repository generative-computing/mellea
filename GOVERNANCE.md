# Governance

This document describes how the Mellea project is governed: the roles people hold, how decisions are made, and how code gets reviewed and merged. This governance applies across the Mellea ecosystem, including the [mellea](https://github.com/generative-computing/mellea) and [mellea-contribs](https://github.com/generative-computing/mellea-contribs) repositories. Repo-specific policies are noted where they apply.

For related topics, see:

- [CONTRIBUTING.md](CONTRIBUTING.md) — development setup, coding standards, and PR workflow
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) — behavioral norms and enforcement
- [RELEASE.md](RELEASE.md) — release cadence, versioning, and process

## Roles

| Role | Description | Examples |
|------|-------------|----------|
| **Contributor** | Anyone who opens issues, submits pull requests, or participates in discussions. No special access required. | External collaborators, first-time contributors |
| **Member** | An established contributor who has demonstrated sustained interest through multiple contributions. Members are added to the GitHub organization and gain the ability to be assigned to issues and PRs, automatic CI runs on their PRs, and eligibility to review PRs. | Organization members |
| **Module Maintainer** | Responsible for the health and direction of a specific area of the codebase or a specific package. Has authority to approve and merge PRs within their area. | Core library maintainers, intrinsics team, contribs package maintainers |
| **Maintainer** | Has project-wide merge/write access and responsible for code review across the repository. Member of [`@generative-computing/mellea-maintainers`](https://github.com/orgs/generative-computing/teams/mellea-maintainers). Maintainers hold release authority and the ability to grant access to new members and module maintainers. | Team members with maintainer rights |

Current module ownership in mellea:

| Area | Owners |
|------|--------|
| `mellea/core/` | @nrfulton, @jakelorocco |
| `mellea/formatters/granite/`, `test/formatters/granite/` | @generative-computing/mellea-intrinsics |
| Everything else | @generative-computing/mellea-maintainers |

In the mellea repository, module maintainers are listed in [`.github/CODEOWNERS`](.github/CODEOWNERS) and automatically requested as reviewers for PRs touching their area.

**How to become a maintainer:**
1. **Contribute** — Submit pull requests, report bugs, review code, participate in discussions
2. **Become a member** — After multiple meaningful contributions, request membership (or be nominated). Requires sponsorship by at least one existing member or maintainer.
3. **Become a module maintainer** — After demonstrating familiarity with an area through contributions and reviews, you may be nominated by an existing maintainer. Nomination is accepted if there are no objections from other maintainers of that area within one week.
4. **Become a maintainer** — After demonstrating sustained, high-quality contributions across the project, you may be invited by existing maintainers.

## PR Review & Merge Policy

### What approval means

A GitHub approval is equivalent to an Apache-style **LGTM** and **implies ownership of the change**. By approving, you are vouching that the change is correct and appropriate. Only approve PRs in areas where you have sufficient domain context — for example, core library changes should be approved by someone with core expertise.

### Requesting reviews

If you explicitly tag someone as a reviewer, you are asking for their review specifically. All explicitly requested reviewers should approve before the PR is merged. In mellea, CODEOWNERS also enforces required reviewers automatically — PRs cannot be merged until all required code-owner reviews are satisfied.

### Reviewing

- **Use "Request Changes" to block** — if you have concerns that must be addressed, use "Request Changes" rather than a comment-only review. This prevents the PR from being merged on another reviewer's approval alone.
- **Respond to all review comments** — authors should resolve or reply to every review thread before merging. Don't let feedback get lost.
- **Re-request review after significant changes** — if you push substantial updates after a review round, re-request review from the same reviewers so they can verify their feedback was addressed.

### Merging

Once a PR has approvals from all requested and required reviewers:

- **Author has commit rights** — the author may merge or enable auto-merge.
- **Author is an external contributor** — the approver is responsible for merging.

Pull requests require the following before merging:

1. **At least one review** from a member, module maintainer, or maintainer
2. **Approval from a module maintainer** (or maintainer) for the affected area or package
3. **All CI checks pass**
4. **No unresolved "request changes" reviews**

In mellea-contribs, pull requests that span multiple packages require approval from a module maintainer of each affected package (or a maintainer).

### PR scope

Keep PRs focused on one logical change. Smaller, well-scoped PRs are easier to review, faster to merge, and safer to revert if needed.

### Contributor responsibility

When a PR is accepted, the maintainers inherit both an asset and a liability: the new functionality is (hopefully) an asset, but the code and documentation required to support it are ongoing liabilities. Reviewers invest real time reading and understanding every PR.

Contributors are responsible for every line they submit. Do not ask others to read and maintain code or documentation that you have not taken the time to read, understand, and refine yourself. This applies regardless of how the code was produced — whether written by hand or generated with AI coding assistants.

**Use of AI coding assistants:** We neither prohibit nor discourage the use of AI coding assistants. However, AI-generated code is not exempt from this standard. If you use AI coding assistants to produce code or documentation, you are expected to review, understand, and take full ownership of the output before submitting it for review. The reviewer's time is not the place to discover what the AI wrote on your behalf. For attribution conventions, see [CONTRIBUTING.md](CONTRIBUTING.md).

### Merge queue

All PRs merge through GitHub's [merge queue](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/managing-a-merge-queue). After approval, PRs enter the queue, which runs CI against the latest `main` before landing. This ensures that every commit on `main` has passed the full test suite.

## Decision-Making

- **Day-to-day changes** (bug fixes, small features): lazy consensus via PR review. If a PR is approved and CI passes, it can be merged.
- **Significant changes** (new core abstractions, new backends, breaking API changes): open a GitHub issue for discussion before submitting a PR. These changes should receive broader review from maintainers and relevant module maintainers.
- **Disputes**: escalate to maintainers. If consensus cannot be reached, decisions are made by majority vote of the maintainers.

In mellea-contribs, module maintainers use lazy consensus within their package — a proposal is considered accepted if no module maintainer objects within a reasonable timeframe.

## Releases

In mellea, maintainers hold release authority. The full release process — including cadence, versioning, and automation — is documented in [RELEASE.md](RELEASE.md).

In mellea-contribs, each package has its own release cadence, managed by its module maintainer(s). There is no project-wide release schedule — packages are released independently as their module maintainers see fit. The maintainers may coordinate releases when cross-package changes require it.

## Communication

- **GitHub Issues and Pull Requests** — code-related decisions and technical discussion
- **GitHub Discussions** — broader topics, questions, and ideas

All community interactions are subject to the [Code of Conduct](CODE_OF_CONDUCT.md).

## mellea-contribs

The [`mellea-contribs`](https://github.com/generative-computing/mellea-contribs) repository is an incubation point for contributions to the Mellea ecosystem. It contains multiple packages — framework integrations, tools, and libraries — each of which may be maintained by different individuals or teams.

### Package Requirements

Every package in the repository must meet the following standards:

- **`pyproject.toml`** — Each package is a self-contained Python project with its own `pyproject.toml` defining dependencies, metadata, and tool configuration.
- **Tests** — Each package must include a `tests/` (or `test/`) directory with at least one test module. All tests must be runnable via `pytest`. See [Testing](#testing) for details.
- **CI** — Packages do not need their own CI workflow files. The repository-level CI workflows automatically discover updated packages and run quality checks (linting, type checking, tests) against them.
- **Documentation** — Each package must include a README with usage documentation. Packages may also include a CONTRIBUTING.md if their contribution workflow has package-specific requirements.
- **License** — Each package must use a license compatible with the project (Apache 2.0 preferred).

### Package Structure

Packages live under `mellea_contribs/` and follow this general layout:

```
mellea_contribs/<package_name>/
├── pyproject.toml
├── README.md
├── src/
│   └── <import_name>/
│       └── ...
└── tests/
    ├── test_*.py
    └── external/     # optional
        └── test_*.py
```

### Package Ownership

Each package has one or more designated module maintainers who are responsible for its health and direction. Module maintainers have autonomy over technical decisions within their package, including:

- API design and implementation approach
- Dependency choices (within project-wide constraints)
- Release cadence for their package
- Internal test organization and contribution guidelines

The maintainers retain override authority on any decision and may intervene when:

- A package decision affects other packages or the project as a whole
- Project-wide standards or policies are not being met
- A package becomes unmaintained

### Package Maintenance

Module maintainers are expected to keep their packages in good working order. This means:

- **CI stays green** — Tests should be passing on the main branch. Flaky or broken tests should be addressed promptly, not left failing.
- **Dependencies stay current** — Packages must specify minimum and maximum compatible versions of Mellea core in their `pyproject.toml` (e.g., `mellea>=0.5.0,<0.7.0`). When Mellea core releases a new version, module maintainers should verify compatibility and update these bounds as needed. Packages that fall too far behind supported Mellea versions may be candidates for deprecation or removal.
- **Python version support** — Packages should support the Python versions tested in CI (currently 3.11, 3.12, and 3.13). When the project adds or drops Python versions, module maintainers should update accordingly.
- **Documentation stays accurate** — READMEs and usage docs should reflect the current state of the package. Outdated documentation is a maintenance issue.
- **Security issues are addressed** — Dependabot alerts, reported vulnerabilities, and security-related issues should be treated as high priority.

#### When a Package Falls Behind

If a package is not meeting the standards above, the following escalation path applies:

1. **The maintainers flag the issue** — by opening an issue on the package describing what needs attention.
2. **The module maintainer has a reasonable window to respond** — either by fixing the issue, providing a timeline, or explaining why the current state is acceptable.
3. **If there is no response**, the maintainers may seek a new module maintainer by posting a call for volunteers.
4. **If no new module maintainer steps up**, the maintainers may retire the package (see [Retiring Packages](#retiring-packages)).

The goal is not to be punitive — module maintainers are volunteers and life happens. The escalation path exists to ensure that packages in the repository remain usable and trustworthy for the broader community.

### Accepting New Packages

To propose a new package, open an issue describing:

1. The package's purpose and scope
2. How it fits within the Mellea ecosystem (i.e., what need it addresses that is not already covered)
3. Who will maintain it

The maintainers will review and decide on acceptance. Accepted packages must meet all the [Package Requirements](#package-requirements) before merging.

### Retiring Packages

A package may be retired by the maintainers if:

- It has no active module maintainer and no one volunteers to take over
- It is superseded by functionality in another package or in Mellea core
- It no longer meets project-wide standards and the module maintainer is unresponsive

Retired packages will be removed from the repository. The maintainers will make reasonable efforts to notify users and provide a migration path before removal.

### Promotion to Mellea Core

Packages that prove their value in `mellea-contribs` may be promoted directly into Mellea core. Promotion is not automatic — it is a recognition that a package has become essential to the ecosystem.

#### Criteria

The maintainers will consider factors such as:

- **Stability** — How long the package has been in the repository and its track record of passing CI.
- **Community adoption** — Evidence of real-world usage, such as downloads, issues filed by users, or adoption by other packages.
- **Active maintenance** — Whether the module maintainer is responsive and keeps the package up to date.
- **Test coverage and documentation** — The overall quality and completeness of the package's test suite and docs.

#### Process

1. **Open a GitHub Discussion** — Anyone (maintainer, user, or module maintainer) can propose a package for promotion by opening a discussion in the repository.
2. **Community input** — The discussion should remain open for a reasonable period to gather feedback from maintainers, users, and module maintainers.
3. **Maintainer decision** — The maintainers review the proposal against the criteria above and make a final decision.
4. **Migration** — If approved, the maintainers coordinate with the module maintainer to migrate the package into Mellea core.

### Testing

Every package is required to have its own test suite. This section describes project-wide testing conventions; individual packages may layer on additional requirements.

#### Requirements

1. **Test directory** — Every package must contain a `tests/` (or `test/`) directory with at least one test module.
2. **Test runner** — All tests must be runnable via `pytest`.
3. **CI workflow** — Every package must have a CI workflow file that calls the shared `quality-generic.yml` workflow.
4. **Tests must pass** — A package's CI checks must pass for any PR that touches that package. Failures in one package do not block PRs to other packages.

#### Conventions

- **External tests** — Tests that require external services (LLM providers, Ollama, APIs) should be placed in a `tests/external/` subdirectory or marked with `@pytest.mark.external`. External tests are excluded from CI by default (`--ignore=tests/external`).
- **Async support** — Use `asyncio_mode = "auto"` in `pyproject.toml` so async tests do not need explicit `@pytest.mark.asyncio` decorators.
- **Custom markers** — Packages that define custom pytest markers (e.g., `qualitative`, `slow`, `requires_api_key`) should document them in their `pyproject.toml` under `[tool.pytest.ini_options]`.

## Evolution of This Document

This governance model is intentionally lightweight. As the contributor community grows, we expect to add:

- Formal inactivity and emeritus policies
- A CODEOWNERS file mapping mellea-contribs packages to module maintainers
- Company diversity guidelines for module maintainer nominations
- A structured proposal process for significant cross-package changes
- Detailed voting and escalation procedures

Changes to this document require approval from the maintainers.
