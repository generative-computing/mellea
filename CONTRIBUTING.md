# Contributing to Mellea

Thank you for your interest in contributing to Mellea! This guide will help you [get started](#getting-started) with developing and contributing to the project.

## Contribution Pathways

There are several ways to contribute to Mellea:

### 1. Contributing to This Repository
Contribute to the Mellea core, standard library, or fix bugs. This includes:
- Core features and bug fixes
- Standard library components (Requirements, Components, Sampling Strategies)
- Backend improvements and integrations
- Documentation and examples
- Tests and CI/CD improvements

**Process:** See the [Pull Request Process](#pull-request-process) section below for detailed steps.

### 2. Applications & Libraries
Build tools and applications using Mellea. These can be hosted in your own repository. For observability, use a `mellea-` prefix.

**Examples:**
- `github.com/my-company/mellea-legal-utils`
- `github.com/my-username/mellea-swe-agent`

### 3. Community Components
Contribute experimental or specialized components to [mellea-contribs](https://github.com/generative-computing/mellea-contribs).

**Note:** For general-purpose Components, Requirements, or Sampling Strategies, please
**open an issue** first to discuss whether they should go in the standard library (this
repository) or mellea-contribs.

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code. Please report unacceptable behavior
to melleaadmin@ibm.com.

## Getting Started

### Prerequisites

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended) or conda/mamba
- [Ollama](https://ollama.com/download) with [required models](#required-models) (for local testing) 

### Installation with `uv` (Recommended)

1. **Fork and clone the repository:**
   ```bash
   git clone ssh://git@github.com/<your-username>/mellea.git
   cd mellea/
   ```

2. **Setup virtual environment:**
   ```bash
   uv venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   # Install all dependencies (recommended for development)
   uv sync --all-extras --all-groups
   
   # Or install just the backend dependencies
   uv sync --extra backends --all-groups
   ```

4. **Install pre-commit hooks (Required):**
   ```bash
   pre-commit install
   ```
   > **Note:** Some hooks require tools in dev dependency groups to be on your PATH. Activate the virtual environment before committing to ensure they are available:
   > ```bash
   > source .venv/bin/activate
   > ```

### Installation with `conda`/`mamba`

1. **Fork and clone the repository:**
   ```bash
   git clone ssh://git@github.com/<your-username>/mellea.git
   cd mellea/
   ```

2. **Run the installation script:**
   ```bash
   conda/install.sh
   ```

This script handles environment setup, dependency installation, and pre-commit hook installation.

### Verify Installation

```bash
# Start Ollama (required for most tests)
ollama serve

# Run fast tests (skip qualitative tests, ~2 min)
uv run pytest -m "not qualitative"
```

## Directory Structure

| Path | Contents |
|------|----------|
| `mellea/core` | Core abstractions: Backend, Base, Formatter, Requirement, Sampling |
| `mellea/stdlib` | Standard library: Session, Context, Components, Requirements, Sampling, Intrinsics, Tools |
| `mellea/backends` | Backend providers: HF, OpenAI, Ollama, Watsonx, LiteLLM |
| `mellea/formatters` | Output formatters and parsers |
| `mellea/helpers` | Utilities, logging, model ID tables |
| `mellea/templates` | Jinja2 templates for prompts |
| `cli/` | CLI commands (`m serve`, `m alora`, `m decompose`, `m eval`) |
| `test/` | All tests (run from repo root) |
| `docs/` | Documentation, examples, tutorials |

## Coding Standards

### Type Annotations

**Required** on all core functions:

```python
def process_text(text: str, max_length: int = 100) -> str:
    """Process text with maximum length."""
    return text[:max_length]
```

### Docstrings

**Docstrings are prompts** - the LLM reads them, so be specific.

Use **[Google-style docstrings](https://google.github.io/styleguide/pyguide.html#381-docstrings)**:

```python
def extract_entities(text: str, entity_types: list[str]) -> dict[str, list[str]]:
    """Extract named entities from text.

    Args:
        text: The input text to analyze.
        entity_types: List of entity types to extract (e.g., ["PERSON", "ORG"]).

    Returns:
        Dictionary mapping entity types to lists of extracted entities.

    Example:
        ```python
        result = extract_entities("Alice works at IBM", ["PERSON", "ORG"])
        # {"PERSON": ["Alice"], "ORG": ["IBM"]}
        ```
    """
    ...
```

#### Code examples in docstrings

Use **triple-backtick fenced code blocks** (` ```python `) for all code examples inside
docstrings — not RST-style `Example::` notation and not `>>>` doctest prompts.

**Why:** Fenced blocks render correctly in IDEs (VS Code hover cards, PyCharm quick
docs), in the Docusaurus-based docs site, and in raw GitHub file views. Doctest-style
`>>>` prompts mislead readers into thinking output is verified, which it is not — this
project does not run doctests. RST directives (`Example::`, `.. deprecated::`,
`:param:`, `:type:`) are not valid inside Google-style docstring sections; use
Google-style section headers (`Example:`, `Raises:`, etc.) with plain Markdown instead.

````python
# Correct — triple-backtick fence inside an Example: section
def greet(name: str) -> str:
    """Return a greeting.

    Example:
        ```python
        greet("world")  # "Hello, world!"
        ```
    """

# Wrong — RST directive inside a docstring section
def greet(name: str) -> str:
    """Return a greeting.

    Example::

        greet("world")
    """

# Wrong — doctest prompts (output is not verified)
def greet(name: str) -> str:
    """Return a greeting.

    Example:
        >>> greet("world")
        'Hello, world!'
    """
````

Code references in docstrings (parameter names, variables, types, literals) use a single backtick (\`), like ``` `variable` ```— not double backticks (``` ``variable`` ```). Mellea uses Markdown-style docstrings, so a single backtick is the correct delimiter for inline code.

#### Class and `__init__` docstrings

Place `Args:` on the **class docstring only**. The `__init__` docstring should be a
single summary sentence with no `Args:` section. This keeps hover docs clean in IDEs
and ensures the docs pipeline (which skips `__init__`) publishes the full parameter
list.

```python
class MyComponent(Component[str]):
    """A component that does something useful.

    Args:
        name (str): Human-readable label for this component.
        max_tokens (int): Upper bound on generated tokens.
    """

    def __init__(self, name: str, max_tokens: int = 256) -> None:
        """Initialize MyComponent with a name and token budget."""
        self.name = name
        self.max_tokens = max_tokens
```

Add an `Attributes:` section on the class docstring **only** when a stored attribute
differs in type or behaviour from the constructor input — for example, when a `str`
argument is wrapped into a `CBlock`, or when a class-level constant is relevant to
callers. Pure-echo entries that repeat `Args:` verbatim should be omitted.

**`TypedDict` classes are a special case.** Their fields *are* the entire public
contract, so when an `Attributes:` section is present it must exactly match the
declared fields. The audit will flag:

- `typeddict_phantom` — `Attributes:` documents a field that is not declared in the `TypedDict`
- `typeddict_undocumented` — a declared field is absent from the `Attributes:` section

```python
class ConstraintResult(TypedDict):
    """Result of a constraint check.

    Attributes:
        passed: Whether the constraint was satisfied.
        reason: Human-readable explanation.
    """
    passed: bool
    reason: str
```

#### Validating docstrings

Run the coverage and quality audit to check your changes before committing:

```bash
# Build fresh API docs then audit quality (documented symbols only)
uv run python tooling/docs-autogen/generate-ast.py
uv run python tooling/docs-autogen/audit_coverage.py \
    --quality --no-methods --docs-dir docs/docs/api
```

Key checks the audit enforces:

| Check | Meaning |
|-------|---------|
| `no_class_args` | Class has typed `__init__` params but no `Args:` on the class docstring |
| `duplicate_init_args` | `Args:` appears in both the class and `__init__` docstrings (Option C violation) |
| `no_args` | Standalone function has params but no `Args:` section |
| `no_returns` | Function has a non-trivial return annotation but no `Returns:` section |
| `param_mismatch` | `Args:` documents names not present in the actual signature |
| `typeddict_phantom` | `TypedDict` `Attributes:` documents a field not declared in the class |
| `typeddict_undocumented` | `TypedDict` has a declared field absent from its `Attributes:` section |

**IDE hover verification** — open any of these existing classes in VS Code and hover
over the class name or a constructor call to confirm the hover card shows `Args:` once
with no duplication:

- `ReactInitiator` ([mellea/stdlib/components/react.py](mellea/stdlib/components/react.py)) — `Args:` + `Attributes:` (`goal: str → CBlock` transform)
- `BaseSamplingStrategy` ([mellea/stdlib/sampling/base.py](mellea/stdlib/sampling/base.py)) — `Args:` only, no `Attributes:` (pure-echo removed)
- `TokenToFloat` ([mellea/formatters/granite/intrinsics/output.py](mellea/formatters/granite/intrinsics/output.py)) — `Attributes:` for `YAML_NAME` class constant

### Code Style

- **Ruff** for linting and formatting
- Use `...` in `@generative` function bodies
- **Prefer primitives over classes** for simplicity
- Keep functions focused and single-purpose
- Avoid over-engineering

### Formatting and Linting

```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Fix auto-fixable issues
uv run ruff check --fix .

# Type check
uv run mypy .
```

## Development Workflow

### Commit Messages

Follow [Angular commit format](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit):

```
<type>: <subject>

<body>

<footer>
```

**Types:** `feat`, `fix`, `docs`, `test`, `refactor`, `release`

**Example:**
```
feat: add support for streaming responses

Implements streaming for all backend types with proper
error handling and timeout management.

Closes #123
```

### Developer Certificate of Origin (DCO)

Mellea uses the [Developer Certificate of Origin](https://developercertificate.org/)
to certify that contributors have the right to submit their work under the project's
license. By signing off on a commit, you are agreeing to the terms of the DCO (full
text below).

**Sign off every commit** using `-s` or `--signoff`:

```bash
git commit -s -m "feat: your commit message"
```

This appends a `Signed-off-by` trailer using your `user.name` and `user.email` from
git config:

```text
Signed-off-by: Jane Doe <jane@example.com>
```

Use your real name and a reachable email. PRs with unsigned commits will be blocked
by the DCO check until every commit is signed off. To retroactively sign existing
commits, use `git rebase --signoff <base>` and force-push.

The repo's pre-commit config also runs a local DCO check at `commit-msg` time, so
unsigned commits fail before they're pushed.

<details>
<summary>Developer Certificate of Origin v1.1 (full text)</summary>

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

</details>

### AI Coding Assistants

AI-assisted development is welcome. You are responsible for reviewing and understanding every change before submitting.

AI coding assistants following project guidelines add an `Assisted-by:` trailer to commit messages by default, identifying which tool was used:

```text
Assisted-by: Claude Code
Assisted-by: IBM Bob
```

Add one line per tool used, using its common name (GitHub Copilot, Cursor, etc.).

### Pre-commit Hooks

Pre-commit hooks run automatically before each commit and check:
- **Ruff** - Linting and formatting
- **mypy** - Type checking
- **uv-lock** - Dependency lock file sync
- **codespell** - Spell checking
- **license-headers** - Inserts the SPDX/copyright header on new code files

**Bypass hooks (for intermediate commits):**
```bash
git commit -n -m "wip: intermediate work"
```

**Run hooks manually:**
```bash
pre-commit run --all-files
```

⚠️ **Warning:** `pre-commit --all-files` may take several minutes. Don't cancel mid-run
as it can corrupt state.

### Pull Request Process

1. **Find or create an issue** for your change:
   - **1a. Issue already exists** — add a comment asking to be assigned the issue
   - **1b. No issue exists** — open an issue describing the change
2. **Fork the repository** (if you haven't already)
3. **Create a branch** in your fork using appropriate naming
4. **Make your changes** following coding standards
5. **Add tests** for new functionality
6. **Run the test suite** to ensure everything passes
7. **Update documentation** as needed
8. **Push to your fork** and create a pull request to the main repository
9. **Follow the automated PR workflow** instructions

### Review Assignment

When a PR is opened, a subset of the relevant CODEOWNERS are requested for review at random. A PR becomes mergeable once CI passes and it has at least one approving review from a relevant CODEOWNER.

Being one of several requested reviewers does not obligate you to review every PR; one approval satisfies the merge requirement. Review when the change touches an area you maintain or when an existing approval lacks the context you need to judge it. When a PR falls in an area another maintainer owns more than you, request them as a reviewer or defer to them.

### Review States

When submitting a PR review, pick one of GitHub's three states. Using them consistently keeps the merge signal meaningful: an `APPROVE` should mean the reviewer is willing to support the change, and a `REQUEST CHANGES` should mean something is actually blocking.

| State | Use when |
|-------|----------|
| `APPROVE` | You'd be fine if this merged as-is. |
| `REQUEST CHANGES` | This PR shouldn't merge in its current form. |
| `COMMENT` | Anything else, or a follow-up that doesn't change the prior status. |

**`APPROVE`** — the reviewer is willing to support the change.
- The PR is ready to merge as-is. Any nits or suggestions are non-blocking; the author may address or skip them with no harm. If something needs a follow-up so it isn't forgotten, that belongs under `REQUEST CHANGES` instead.
- The standard open-source "LGTM" signal: the reviewer stands behind the change and shares responsibility for maintaining that area going forward.

**`REQUEST CHANGES`** — the PR should not merge in its current form. Use when:
- There is an actual blocking or breaking issue (correctness, regression, missing tests, etc.).
- The reviewer needs to validate a concrete concern (a suspected regression, an incomplete change) before the PR merges, and is withholding approval until the next round. Wanting another pass without a specific concern isn't on its own a reason to block; coordinate with the author by comment instead.
- Important follow-up issues have not yet been opened, and the reviewer doesn't want them forgotten. The resolution is to file the issues, not to change the PR; once they're filed the reviewer re-approves to clear the block.

**`COMMENT`** — for everything else. Use when:
- Posting a follow-up review that shouldn't change the PR's status from a prior `APPROVE` or `REQUEST CHANGES` (e.g., re-reviewing after a push when nothing material changed).
- The review falls between `APPROVE` and `REQUEST CHANGES` and the reviewer doesn't want to block.
- The reviewer wants to defer the final call to other reviewers.
- The review is purely informational and the reviewer isn't gating on the PR at all.

### Merging

Merging is a maintainer action, performed by either the author or a reviewer; "merge when ready" only schedules it once requirements pass. Whoever merges confirms the PR is actually ready, not just that the queue requirements are green. CI passing and the minimum approval make a PR *mergeable*, but a concern raised in discussion that never became an explicit `REQUEST CHANGES` should still be resolved first.

The `do-not-merge/hold` label blocks merging while it is applied, enforced by CI independent of review state. It is a separate mechanism from a `REQUEST CHANGES` review and is removed once the blocking condition clears.

### Review comments

For small nits on a `COMMENT` or `APPROVE` review, prefer a GitHub suggestion block — it lets the author apply the change in one click and keeps non-blocking feedback from adding review-cycle friction.

Resolve a review conversation once its specific point is handled; prefer the reviewer who raised it to resolve it, so an unresolved thread reliably means "still open."

## Testing

### Quick Reference

See [test/README.md](test/README.md) for classification rules, authoring guide,
CI tier map, coverage, and the full local workflow reference. Essential commands:

```bash
# Install all dependencies (required for tests)
uv sync --all-extras --all-groups

# Start Ollama (required for most tests)
ollama serve

# Default: includes qualitative tests, skips slow tests
uv run pytest

# Fast tests only (no qualitative, ~2 min)
uv run pytest -m "not qualitative"

# Lint and format
uv run ruff format .
uv run ruff check .
```

### Required Models

#### Ollama

HuggingFace and cloud backends download or host models automatically. Ollama
models must be pulled locally before running the tests that need them.

**CI (unit + integration tests):**

- `granite4.1:3b` — default model for `start_session()` and most examples

**Examples (`docs/examples/`):**

- `deepseek-r1:8b` — safety / guardian examples
- `granite3-guardian:2b` — mini-researcher guardian backend
- `granite3.2-vision` — vision (Ollama chat) example
- `granite3.3:8b` — m\_decompose example
- `granite4:latest` — melp examples
- `llama3.2` — repair-with-guardian example
- `llama3.2:3b` — tutorial / mify examples (via `META_LLAMA_3_2_3B`)
- `qwen2.5vl:7b` — vision (OpenAI-via-Ollama) example

**Additional test models (`test/`):**

- `granite4:small-h` — hybrid-small tests
- `llama3.2:1b` — lightweight inference tests
- `llama3:8b` — legacy Llama 3 tests
- `llava` — multimodal tests
- `mistral:7b` — Mistral backend tests
- `smollm2:1.7b` — SmolLM tests

Pull everything:

```bash
for m in granite4.1:3b deepseek-r1:8b \
  granite3-guardian:2b granite3.2-vision granite3.3:8b granite4:latest \
  llama3.2 llama3.2:3b \
  qwen2.5vl:7b granite4:small-h llama3.2:1b llama3:8b llava mistral:7b \
  smollm2:1.7b; do ollama pull "$m"; done
```

### Test Markers

Tests use a four-tier granularity system (`unit`, `integration`, `e2e`, `qualitative`) plus backend and resource markers. See [test/README.md](test/README.md) for the full guide: classification rules, marker reference, authoring guide, CI tiers, and auto-skip logic.

### CI/CD Tests

CI runs the following checks on every pull request:
1. **Pre-commit hooks** (`pre-commit run --all-files`) — ruff, mypy, uv-lock, codespell, license-headers
2. **Test suite** — `CICD=1 uv run pytest test` on Python 3.11/3.12/3.13 with Ollama running; skips qualitative tests

To replicate CI locally:
```bash
# Pre-commit checks (same as CI)
pre-commit run --all-files

# Tests with CICD flag (skips qualitative, matches CI scope)
CICD=1 uv run pytest test
```

See [test/README.md — CI pipeline](test/README.md#ci-pipeline) for the full CI
breakdown and planned nightly/pre-release tiers.

### Timing Expectations

- Fast tests (`-m "not qualitative"`): ~2 minutes
- Default tests (qualitative, no slow): Several minutes
- Slow tests (`-m slow`): >1 minute each
- Pre-commit hooks: 1-5 minutes

⚠️ **Don't cancel mid-run** - canceling `pytest` or `pre-commit` can corrupt state.

## Common Issues & Troubleshooting

| Problem | Fix |
|---------|-----|
| `ComponentParseError` | LLM output didn't match expected type. Add examples to docstring. |
| `uv.lock` out of sync | Run `uv sync` to update lock file. |
| `Ollama refused connection` | Run `ollama serve` to start Ollama server. |
| `ConnectionRefusedError` (port 11434) | Ollama not running. Start with `ollama serve`. |
| `TypeError: missing positional argument` | First argument to `@generative` function must be session `m`. |
| Output is wrong/None | Model too small or needs better prompt. Try larger model or add `reasoning` field. |
| `error: can't find Rust compiler` | Python 3.13+ requires Rust for outlines. Install [Rust](https://www.rust-lang.org/tools/install) or use Python 3.12. |
| Tests fail on Intel Mac | Use conda: `conda install 'torchvision>=0.22.0'` then `uv pip install mellea`. |
| Pre-commit hooks fail | Run `pre-commit run --all-files` to see specific issues. Fix or use `git commit -n` to bypass. If a tool reports `command not found`, activate the virtual environment before committing: `source .venv/bin/activate`. |

### Debugging Tips

```python
# Enable debug logging
from mellea.core import MelleaLogger
MelleaLogger.get_logger().setLevel("DEBUG")

# See exact prompt sent to LLM
print(m.last_prompt())
```

### Getting Help

- Check this guide and [test/README.md](test/README.md)
- Search [existing issues](https://github.com/generative-computing/mellea/issues)
- Check out [Github Discussions](https://github.com/generative-computing/mellea/discussions)
- Open a new issue with the appropriate label

## Additional Resources

### Documentation

- **[Docs writing guide](docs/CONTRIBUTING_DOCS.md)** - Conventions, PR checklist, and review process for documentation contributions
- **[API Documentation](https://docs.mellea.ai)** - Published documentation site
- **[Test Guide](test/README.md)** - Test strategy, classification, markers, and authoring guide
- **[AGENTS.md](AGENTS.md)** - Guidelines for AI assistants working on Mellea internals
- **[AGENTS_TEMPLATE.md](docs/AGENTS_TEMPLATE.md)** - Template for projects using Mellea

### Community
- **[GitHub Issues](https://github.com/generative-computing/mellea/issues)** - Report bugs or request features
- **[GitHub Discussions](https://github.com/generative-computing/mellea/discussions)** - Ask questions and share ideas

### Related Repositories
- **[mellea-contribs](https://github.com/generative-computing/mellea-contribs)** - Community contributions

---

## Feedback Loop

Found a bug, workaround, or pattern while contributing?

- **Issue/workaround?** → Add to [Common Issues](#common-issues--troubleshooting) section
- **Usage pattern?** → Add to [docs/AGENTS_TEMPLATE.md](docs/AGENTS_TEMPLATE.md)
- **New pitfall?** → Add warning to relevant section

Help us improve this guide by opening a PR with your additions!

---

Thank you for contributing to Mellea! 🎉
