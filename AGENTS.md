<!--
AGENTS.md — Instructions for AI coding assistants (Claude, Cursor, Copilot, Codex, Roo, etc.)
-->

# Agent Guidelines for Mellea Contributors

> **Which guide?** Modifying `mellea/`, `cli/`, or `test/` → this file. Writing code that imports Mellea → [`docs/AGENTS_TEMPLATE.md`](docs/AGENTS_TEMPLATE.md).

## 1. Quick Reference
```bash
pre-commit install                  # Required: install git hooks
uv sync --all-extras --all-groups   # Install all deps (required for tests)
ollama serve                        # Start Ollama (required for most tests)
uv run pytest                       # Fast tests (excludes qualitative, ~2 min)
uv run pytest -m ""                 # Full suite (includes qualitative tests)
uv run ruff format . && uv run ruff check .  # Lint & format
```
**Branches**: `feat/topic`, `fix/issue-id`, `docs/topic`

## 2. Directory Structure
| Path | Contents |
|------|----------|
| `mellea/stdlib` | Core: Sessions, Genslots, Requirements, Sampling, Context |
| `mellea/backends` | Providers: HF, OpenAI, Ollama, Watsonx, LiteLLM |
| `mellea/helpers` | Utilities, logging, model ID tables |
| `cli/` | CLI commands (`m serve`, `m alora`, `m decompose`, `m eval`) |
| `test/` | All tests (run from repo root) |
| `scratchpad/` | Experiments (git-ignored) |

## 3. Test Markers
All tests and examples use markers to indicate requirements. The test infrastructure automatically skips tests based on system capabilities.

**Backend Markers:**
- `@pytest.mark.ollama` — Requires Ollama running (local, lightweight)
- `@pytest.mark.huggingface` — Requires HuggingFace backend (local, heavy)
- `@pytest.mark.vllm` — Requires vLLM backend (local, GPU required)
- `@pytest.mark.openai` — Requires OpenAI API (requires API key)
- `@pytest.mark.watsonx` — Requires Watsonx API (requires API key)
- `@pytest.mark.litellm` — Requires LiteLLM backend

**Capability Markers:**
- `@pytest.mark.requires_gpu` — Requires GPU
- `@pytest.mark.requires_heavy_ram` — Requires 48GB+ RAM
- `@pytest.mark.requires_api_key` — Requires external API keys
- `@pytest.mark.qualitative` — LLM output quality tests (excluded by default, skipped in CI via `CICD=1`)
- `@pytest.mark.llm` — Makes LLM calls (needs at least Ollama)

**Examples in `docs/examples/`** use the same markers. Tests/examples automatically skip if system lacks required resources.

**Default behavior:** `uv run pytest` excludes qualitative tests for fast feedback (~2 min). Use `uv run pytest -m ""` to run full suite including qualitative tests.

⚠️ Don't add `qualitative` to trivial tests—keep the fast loop fast.

## 4. Coding Standards
- **Types required** on all core functions
- **Docstrings are prompts** — be specific, the LLM reads them
- **Google-style docstrings**
- **Ruff** for linting/formatting
- Use `...` in `@generative` function bodies
- Use `...` in `@generative` function bodies
- Prefer primitives over classes
- **Friendly Dependency Errors**: Wraps optional backend imports in `try/except ImportError` with a helpful message (e.g., "Please pip install mellea[hf]"). See `mellea/stdlib/session.py` for examples.

## 5. Commits & Hooks
[Angular format](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit): `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `release:`

Pre-commit runs: ruff, mypy, uv-lock, codespell

## 6. Timing
> **Don't cancel**: `pytest` (full) and `pre-commit --all-files` may take minutes. Canceling mid-run can corrupt state.

## 7. Common Issues
| Problem | Fix |
|---------|-----|
| `ComponentParseError` | Add examples to docstring |
| `uv.lock` out of sync | Run `uv sync` |
| Ollama refused | Run `ollama serve` |

## 8. Self-Review (before notifying user)
1. `uv run pytest -m "not qualitative"` passes?
2. `ruff format` and `ruff check` clean?
3. New functions typed with concise docstrings?
4. Unit tests added for new functionality?
5. Avoided over-engineering?

## 9. Writing Tests
- Place tests in `test/` mirroring source structure
- Name files `test_*.py` (required for pydocstyle)
- Use `gh_run` fixture for CI-aware tests (see `test/conftest.py`)
- Mark tests checking LLM output quality with `@pytest.mark.qualitative`
- If a test fails, fix the **code**, not the test (unless the test was wrong)

## 10. Feedback Loop
Found a bug, workaround, or pattern? Update the docs:
- **Issue/workaround?** → Add to Section 7 (Common Issues) in this file
- **Usage pattern?** → Add to [`docs/AGENTS_TEMPLATE.md`](docs/AGENTS_TEMPLATE.md)
- **New pitfall?** → Add warning near relevant section
