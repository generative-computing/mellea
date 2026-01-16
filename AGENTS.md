<!--
AGENTS.md — Instructions for AI coding assistants (Claude, Cursor, Copilot, Codex, Roo, etc.)
-->

# Agent Guidelines for Mellea Contributors

> **Which guide?** Modifying `mellea/`, `cli/`, or `test/` → this file. Writing code that imports Mellea → [`docs/AGENTS_TEMPLATE.md`](docs/AGENTS_TEMPLATE.md).

## 1. Quick Reference
```bash
pre-commit install                  # Required: install git hooks
uv sync                             # Install deps & fix lockfile
uv run pytest -m "not qualitative"  # Fast loop (unit tests only)
uv run pytest                       # Full suite (includes LLM tests)
uv run pytest -m integration        # Tests requiring API keys
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
| `test/` | All tests. Unmarked = unit tests (no network/API keys) |
| `scratchpad/` | Experiments (git-ignored) |

## 3. Test Markers
- `@pytest.mark.qualitative` — LLM output quality tests (skipped in CI)
- `@pytest.mark.integration` — Requires API keys
- **Unmarked** — Pure unit tests: no network, deterministic

⚠️ Don't add `qualitative` to trivial tests—keep the fast loop fast.

## 4. Coding Standards
- **Types required** on all core functions
- **Docstrings are prompts** — be specific, the LLM reads them
- **Google-style docstrings**
- **Ruff** for linting/formatting
- Use `...` in `@generative` function bodies
- Prefer primitives over classes

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
- Use `gh_run` fixture for CI-aware tests (see `conftest.py`)
- **No LLM calls** in unmarked tests—mock or mark `qualitative`
- If a test fails, fix the **code**, not the test (unless the test was wrong)

## 10. Feedback Loop
Found a bug, workaround, or pattern? Update the docs:
- **Issue/workaround?** → Add to Section 7 (Common Issues) in this file
- **Usage pattern?** → Add to [`docs/AGENTS_TEMPLATE.md`](docs/AGENTS_TEMPLATE.md)
- **New pitfall?** → Add warning near relevant section
