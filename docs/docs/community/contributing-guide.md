---
title: "Contributing to Mellea"
description: "Development setup, coding standards, and PR process for Mellea contributors."
# diataxis: how-to
---

**Prerequisites:** Python 3.10+, [uv](https://docs.astral.sh/uv/getting-started/installation/) installed, [Ollama](https://ollama.com/download) installed.

## Contribution pathways

Three pathways exist for contributing to Mellea:

**Core repository** — bug fixes, standard library additions (Requirements, Components, Sampling Strategies), backend improvements, documentation, and tests. Follow the [Pull request process](#pull-request-process) below.

**Applications and libraries** — build tools or applications on top of Mellea in your own repository. Use the `mellea-` prefix for discoverability (e.g., `github.com/my-company/mellea-legal-utils`).

**Community components** — contribute experimental or specialized components to [mellea-contribs](https://github.com/generative-computing/mellea-contribs). Open an issue first for general-purpose additions to decide whether they belong in the standard library or in mellea-contribs.

## Development setup

### Set up with uv (recommended)

1. Fork and clone the repository:

   ```bash
   git clone ssh://git@github.com/<your-username>/mellea.git
   cd mellea/
   ```

2. Create a virtual environment:

   ```bash
   uv venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   # Install all dependencies (recommended for development)
   uv sync --all-extras --all-groups

   # Or install only backend dependencies
   uv sync --extra backends --all-groups
   ```

4. Install pre-commit hooks (required):

   ```bash
   pre-commit install
   ```

> **Note:** Python 3.13+ requires a [Rust compiler](https://www.rust-lang.org/tools/install) for the `outlines` dependency. Use Python 3.12 if you prefer to avoid this.

### Set up with conda or mamba

1. Fork and clone the repository:

   ```bash
   git clone ssh://git@github.com/<your-username>/mellea.git
   cd mellea/
   ```

2. Run the installation script:

   ```bash
   conda/install.sh
   ```

   The script handles environment setup, dependency installation, and pre-commit hook installation.

### Verify the installation

```bash
# Start Ollama (required for most tests)
ollama serve

# Run fast tests (skip qualitative tests, ~2 min)
uv run pytest -m "not qualitative"
```

## Coding standards

### Type annotations

Type annotations are required on all core functions:

```python
def process_text(text: str, max_length: int = 100) -> str:
    """Process text with maximum length."""
    return text[:max_length]
```

### Docstrings

Docstrings serve as prompts — the LLM reads them, so be specific. Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#381-docstrings):

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

### Code style

- Use **Ruff** for linting and formatting.
- Use `...` in `@generative` function bodies.
- Prefer primitives over classes.
- Keep functions focused and single-purpose.

### Linting and formatting

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

## Development workflow

### Commit messages

Follow [Angular commit format](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit):

```text
<type>: <subject>

<body>

<footer>
```

**Types:** `feat`, `fix`, `docs`, `test`, `refactor`, `release`

**Example:**

```text
feat: add support for streaming responses

Implements streaming for all backend types with proper
error handling and timeout management.

Closes #123
```

Always sign off commits with `-s` or `--signoff`:

```bash
git commit -s -m "feat: your commit message"
```

**Branch naming:** `feat/topic`, `fix/issue-id`, `docs/topic`

### Pre-commit hooks

Pre-commit hooks run automatically before each commit and check:

- **Ruff** — linting and formatting
- **mypy** — type checking
- **uv-lock** — dependency lock file sync
- **codespell** — spell checking

Run hooks manually:

```bash
pre-commit run --all-files
```

> **Warning:** `pre-commit --all-files` may take several minutes. Do not cancel mid-run as it can corrupt state.

Use the `-n` flag to bypass hooks for intermediate work-in-progress commits:

```bash
git commit -n -m "wip: intermediate work"
```

## Testing

### Test markers

Tests are categorized using pytest markers:

| Marker | Requirement |
| ------ | ----------- |
| `@pytest.mark.ollama` | Ollama running locally (lightweight) |
| `@pytest.mark.huggingface` | HuggingFace backend (local, heavy) |
| `@pytest.mark.vllm` | vLLM backend (GPU required) |
| `@pytest.mark.openai` | OpenAI API key |
| `@pytest.mark.watsonx` | Watsonx API key |
| `@pytest.mark.litellm` | LiteLLM backend |
| `@pytest.mark.requires_gpu` | GPU available |
| `@pytest.mark.requires_heavy_ram` | 48 GB+ RAM |
| `@pytest.mark.requires_api_key` | External API key |
| `@pytest.mark.qualitative` | LLM output quality (skipped in CI via `CICD=1`) |
| `@pytest.mark.llm` | Makes LLM calls (needs at least Ollama) |
| `@pytest.mark.slow` | Tests taking more than 5 minutes |

> **Warning:** Do not add `qualitative` to trivial tests — keep the fast loop fast. Mark tests taking more than 5 minutes with `slow`.

### Running tests

```bash
# Install all dependencies (required for tests)
uv sync --all-extras --all-groups

# Start Ollama (required for most tests)
ollama serve

# Default: runs qualitative tests, skips slow tests
uv run pytest

# Fast tests only (no qualitative, ~2 min)
uv run pytest -m "not qualitative"

# Run only slow tests (>5 min)
uv run pytest -m slow

# Run specific backend tests
uv run pytest -m "ollama"
uv run pytest -m "openai"

# Run tests without LLM calls (unit tests only)
uv run pytest -m "not llm"

# CI/CD mode (skips qualitative tests)
CICD=1 uv run pytest
```

### Timing expectations

| Run | Duration |
| --- | -------- |
| Fast tests (`-m "not qualitative"`) | ~2 minutes |
| Default (qualitative, no slow) | Several minutes |
| Slow tests (`-m slow`) | More than 5 minutes |
| Pre-commit hooks | 1–5 minutes |

### Replicate CI locally

```bash
# Run pre-commit checks (same as CI)
pre-commit run --all-files

# Run tests with CICD flag (same as CI, skips qualitative tests)
CICD=1 uv run pytest
```

## Pull request process

1. Create an issue describing your change (if one does not already exist).
2. Fork the repository.
3. Create a branch in your fork using the naming convention above.
4. Make your changes following the coding standards.
5. Add tests for new functionality.
6. Run the test suite to confirm everything passes.
7. Update documentation as needed.
8. Push to your fork and open a pull request.
9. Follow the automated PR workflow instructions in the PR template.

## Troubleshooting

| Problem | Fix |
| ------- | --- |
| `ComponentParseError` | LLM output did not match expected type. Add examples to the docstring. |
| `uv.lock` out of sync | Run `uv sync` to update the lock file. |
| `Ollama refused connection` | Run `ollama serve` to start the Ollama server. |
| `ConnectionRefusedError` (port 11434) | Ollama is not running. Start with `ollama serve`. |
| `TypeError: missing positional argument` | First argument to a `@generative` function must be session `m`. |
| Output is wrong or None | Model too small or prompt insufficient. Try a larger model or add a `reasoning` field. |
| `error: can't find Rust compiler` | Python 3.13+ requires Rust for outlines. Install [Rust](https://www.rust-lang.org/tools/install) or use Python 3.12. |
| Tests fail on Intel Mac | Use conda: `conda install 'torchvision>=0.22.0'` then `uv pip install mellea`. |
| Pre-commit hooks fail | Run `pre-commit run --all-files` to see specific issues. Fix them, or use `git commit -n` to bypass. |

### Debugging tips

```python
from mellea.core import FancyLogger

# Enable debug logging
FancyLogger.get_logger().setLevel("DEBUG")

# Inspect the exact prompt sent to the LLM
print(m.last_prompt())
```

## Contributing to the docs

Documentation lives in `docs/docs/`. The writing guide at
[`docs/docs/guide/CONTRIBUTING`](../guide/CONTRIBUTING) covers conventions, the PR
checklist, and the review process for documentation contributions. Key points:

- Start body content with H2 — Mintlify renders the frontmatter `title` as the page heading.
- Omit `.md` extensions from internal links.
- Tag every fenced code block with a language.
- Run `npx markdownlint-cli2` and fix all warnings before committing.

## Getting help

- Check [existing issues](https://github.com/generative-computing/mellea/issues)
- Join the [Discord](https://ibm.biz/mellea-discord)
- Open a new issue with the appropriate label

---

**See also:** [Building Extensions](../community/building-extensions)
