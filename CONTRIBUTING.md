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

- Python 3.10 or higher (3.13+ requires [Rust compiler](https://www.rust-lang.org/tools/install) for outlines)
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
        >>> extract_entities("Alice works at IBM", ["PERSON", "ORG"])
        {"PERSON": ["Alice"], "ORG": ["IBM"]}
    """
    ...
```

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

**Important:** Always sign off commits using `-s` or `--signoff`:
```bash
git commit -s -m "feat: your commit message"
```

### Pre-commit Hooks

Pre-commit hooks run automatically before each commit and check:
- **Ruff** - Linting and formatting
- **mypy** - Type checking
- **uv-lock** - Dependency lock file sync
- **codespell** - Spell checking

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

1. **Create an issue** describing your change (if not already exists)
2. **Fork the repository** (if you haven't already)
3. **Create a branch** in your fork using appropriate naming
4. **Make your changes** following coding standards
5. **Add tests** for new functionality
6. **Run the test suite** to ensure everything passes
7. **Update documentation** as needed
8. **Push to your fork** and create a pull request to the main repository
9. **Follow the automated PR workflow** instructions

## Testing

### Quick Reference

```bash
# Install all dependencies (required for tests)
uv sync --all-extras --all-groups

# Start Ollama (required for most tests)
ollama serve

# Default: qualitative tests, skip slow tests
uv run pytest

# Fast tests only (no qualitative, ~2 min)
uv run pytest -m "not qualitative"

# Run only slow tests (>5 min)
uv run pytest -m slow

# Run ALL tests including slow (bypass config)
uv run pytest --co -q

# Run specific backend tests
uv run pytest -m "ollama"
uv run pytest -m "openai"

# Run tests without LLM calls (unit tests only)
uv run pytest -m "not llm"

# CI/CD mode (skips qualitative tests)
CICD=1 uv run pytest

# Lint and format
uv run ruff format .
uv run ruff check .
```

### Required Models

#### Ollama
- `granite4:micro-h`
- `granite3.2-vision`
- `granite4:micro`
- `qwen2.5vl:7b`

_Note: ollama models can be obtained by running `ollama pull <model>`_

### Test Markers

Tests are categorized using pytest markers:

**Backend Markers:**
- `@pytest.mark.ollama` - Requires Ollama running (local, lightweight)
- `@pytest.mark.huggingface` - Requires HuggingFace backend (local, heavy)
- `@pytest.mark.vllm` - Requires vLLM backend (local, GPU required)
- `@pytest.mark.openai` - Requires OpenAI API (requires API key)
- `@pytest.mark.watsonx` - Requires Watsonx API (requires API key)
- `@pytest.mark.litellm` - Requires LiteLLM backend

**Capability Markers:**
- `@pytest.mark.requires_gpu` - Requires GPU
- `@pytest.mark.requires_heavy_ram` - Requires 48GB+ RAM
- `@pytest.mark.requires_api_key` - Requires external API keys
- `@pytest.mark.qualitative` - LLM output quality tests (skipped in CI via `CICD=1`)
- `@pytest.mark.llm` - Makes LLM calls (needs at least Ollama)
- `@pytest.mark.slow` - Tests taking >5 minutes (skipped via `SKIP_SLOW=1`)

**Execution Strategy Markers:**
- `@pytest.mark.requires_gpu_isolation` - Requires OS-level process isolation to clear CUDA memory (use with `--isolate-heavy` or `CICD=1`)

**Default behavior:**
- `uv run pytest` skips slow tests (>5 min) but runs qualitative tests
- Use `pytest -m "not qualitative"` for fast tests only (~2 min)
- Use `pytest -m slow` or `pytest --co -q` to include slow tests

⚠️ **Don't add `qualitative` to trivial tests** - keep the fast loop fast.
⚠️ **Mark tests taking >5 minutes with `slow`** (e.g., dataset loading, extensive evaluations).

For detailed information about test markers, resource requirements, and running specific
test categories, see [test/MARKERS_GUIDE.md](test/MARKERS_GUIDE.md).

### CI/CD Tests

CI runs the following checks on every pull request:
1. **Pre-commit hooks** (`pre-commit run --all-files`) - Ruff, mypy, uv-lock, codespell
2. **Test suite** (`CICD=1 uv run pytest`) - Skips qualitative tests for speed

To replicate CI locally:
```bash
# Run pre-commit checks (same as CI)
pre-commit run --all-files

# Run tests with CICD flag (same as CI, skips qualitative tests)
CICD=1 uv run pytest
```

### Timing Expectations

- Fast tests (`-m "not qualitative"`): ~2 minutes
- Default tests (qualitative, no slow): Several minutes
- Slow tests (`-m slow`): >5 minutes
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
| Pre-commit hooks fail | Run `pre-commit run --all-files` to see specific issues. Fix or use `git commit -n` to bypass. |

### Debugging Tips

```python
# Enable debug logging
from mellea.core import FancyLogger
FancyLogger.get_logger().setLevel("DEBUG")

# See exact prompt sent to LLM
print(m.last_prompt())
```

### Getting Help

- Check this guide and [test/MARKERS_GUIDE.md](test/MARKERS_GUIDE.md)
- Search [existing issues](https://github.com/generative-computing/mellea/issues)
- Join our [Discord](https://ibm.biz/mellea-discord)
- Open a new issue with the appropriate label

## Additional Resources

### Documentation

- **[Docs writing guide](docs/docs/guide/CONTRIBUTING.md)** - Conventions, PR checklist, and review process for documentation contributions
- **[API Documentation](https://docs.mellea.ai)** - Published documentation site
- **[Test Markers Guide](test/MARKERS_GUIDE.md)** - Detailed pytest marker documentation
- **[AGENTS.md](AGENTS.md)** - Guidelines for AI assistants working on Mellea internals
- **[AGENTS_TEMPLATE.md](docs/AGENTS_TEMPLATE.md)** - Template for projects using Mellea

### Community
- **[Discord](https://ibm.biz/mellea-discord)** - Join our community
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
