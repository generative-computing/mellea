# RELEASE.md

## Overview
This document describes the release process for the project. It outlines how release candidates are prepared, how testing is performed, how the final package is published, and what conventions the project uses for versioning and communication. The goal is to ensure reliability, transparency, and predictability for users and contributors.

## Release Cadence
Releases are prepared weekly, with work occurring roughly on the following schedule:

- **Mon–Tue:** Identify and finalize high‑priority pull requests for inclusion.
- **Wed:** Merge selected pull requests, lock down changes, and cut the release.

This schedule may shift depending on team availability, urgency of fixes, and the stability of the codebase.

## Versioning
This project follows **[Semantic Versioning (semver)](https://semver.org)**. Breaking changes must be clearly documented and highlighted in release notes.

To see the current, **prospective** version that will be published by the next release action, you can run:
```
uv run --no-sync semantic-release print-version
```

## Pre‑Release Coordination
### 1. Identify Candidate Pull Requests
Early in the week, maintainers review open pull requests and determine which ones should be included in the upcoming release.

### 2. Merge or Defer Selected Pull Requests
By mid‑week, all targeted PRs should be merged or explicitly deferred. No new PRs should be merged once the release process begins.

## Release Process
### 1. Freeze `main`
Once the release process begins, no new PRs may be merged into `main`.

### 2. Run the Full Test Suite
All tests must pass before proceeding. The suite includes unit tests, integration tests, and example workflows. Issues are classified into quick fixes or larger problems requiring release delay. This should be run against python versions 3.11, 3.12, and 3.13.

### 3. Trigger Release Automation
A GitHub Actions workflow handles the CI pipeline, branch cutting, and publishing to PyPI.

## Post‑Release Steps
### 1. Announcement & Communication
When a release includes noteworthy updates—especially breaking API changes—maintainers should publish release notes and communicate updates publicly.

### 2. Documentation Synchronization
Documentation must be updated before the release is finalized. API Documentation is automatically generated.

### 3. Transparency: Test Strategy & Results
Testing methodology and coverage information may be made public where appropriate.

## Release Types
- **Fix releases** – bug fixes and stability improvements
- **Feature releases** – new capabilities, backward compatible
- **Breaking releases** – major API revisions

## Supported Use Cases & Stability
The project is in **beta**, so users should expect rapid iteration and occasional breaking changes.

## Future Improvements
Future enhancements may include:
- add package dependency tests
    - ensure each piece of mellea can run with only its necessary mellea package (ie huggingface with mellea[hf])
- release candidate branch
    - the job supports a branch selector; we should make sure it works as expected
    - this also means we need to keep track of which PRs need to be double merged (ie into both the release and main branches)
- improved PR coordination
    - this becomes less necessary with a candidate release branch
- publishing a standardized test strategy
