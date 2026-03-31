<!--
AGENTS.md — Instructions for AI coding assistants (Claude, Cursor, Copilot, Codex, Roo, etc.)
-->

# Agent Guidelines for Mellea Contributors

> **Which guide?** Modifying `mellea/`, `cli/`, or `test/` → this file. Writing code that imports Mellea → [`docs/AGENTS_TEMPLATE.md`](docs/AGENTS_TEMPLATE.md).

> **Code of Conduct**: This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). All contributors, including AI assistants, are expected to follow these community standards when generating code, documentation, or interacting with the project.

## 1. Quick Reference

**⚠️ Always use `uv` for Python commands** — never use system Python or `pip` directly.
- Run Python scripts: `uv run python script.py` (not `python script.py`)
- Run tools: `uv run pytest`, `uv run ruff` (not `pytest`, `ruff`)
- Install deps: `uv sync` (not `pip install`)
- The virtual environment is `.venv/` — `uv run` automatically uses it

```bash
pre-commit install                    # Required: install git hooks
uv sync --all-extras --all-groups     # Install all deps (required for tests)
uv sync --extra backends --all-groups # Install just backend deps (lighter)
ollama serve                          # Start Ollama (required for most tests)
uv run pytest                         # Default: qualitative tests, skip slow tests
uv run pytest -m "not qualitative"    # Fast tests only (~2 min)
uv run pytest -m slow                 # Run only slow tests (>5 min)
uv run pytest --co -q                 # Run ALL tests including slow (bypass config)
uv run ruff format .                  # Format code
uv run ruff check .                   # Lint code
uv run mypy .                         # Type check
```
**Branches**: `feat/topic`, `fix/issue-id`, `docs/topic`

## 2. Directory Structure
| Path | Contents |
|------|----------|
| `mellea/core/` | Core abstractions: Backend, Base, Formatter, Requirement, Sampling |
| `mellea/stdlib/` | Standard library: Sessions, Components, Context |
| `mellea/backends/` | Providers: HF, OpenAI, Ollama, Watsonx, LiteLLM |
| `mellea/formatters/` | Output formatters for different types |
| `mellea/templates/` | Jinja2 templates |
| `mellea/helpers/` | Utilities, logging, model ID tables |
| `cli/` | CLI commands (`m serve`, `m alora`, `m decompose`, `m eval`) |
| `test/` | All tests (run from repo root) |
| `docs/examples/` | Example code (run as tests via pytest) |
| `.agents/skills/` | Agent skills ([agentskills.io](https://agentskills.io) standard) |
| `scratchpad/` | Experiments (git-ignored) |

## 3. Test Markers
Tests use a four-tier granularity system (`unit`, `integration`, `e2e`, `qualitative`) plus backend and resource markers. The `unit` marker is auto-applied by conftest — never write it explicitly. The `llm` marker is deprecated; use `e2e` instead.

See **[test/MARKERS_GUIDE.md](test/MARKERS_GUIDE.md)** for the full marker reference (tier definitions, backend markers, resource gates, auto-skip logic, common patterns).

**Examples in `docs/examples/`** use comment-based markers:
```python
# pytest: e2e, ollama, qualitative
"""Example description..."""
```

⚠️ Don't add `qualitative` to trivial tests — keep the fast loop fast.
⚠️ Mark tests taking >1 minute with `slow`.

## 4. Agent Skills

Skills live in `.agents/skills/` following the [agentskills.io](https://agentskills.io) open standard. Each skill is a directory with a `SKILL.md` file (YAML frontmatter + markdown instructions).

**Tool discovery:**

| Tool              | Project skills    | Global skills       | Config needed                                                      |
| ----------------- | ----------------- | ------------------- | ------------------------------------------------------------------ |
| Claude Code       | `.agents/skills/` | `~/.claude/skills/` | `"skillLocations": [".agents/skills"]` in `.claude/settings.json`  |
| IBM Bob           | `.bob/skills/`    | `~/.bob/skills/`    | Symlink: `.bob/skills` → `.agents/skills`                          |
| VS Code / Copilot | `.agents/skills/` | —                   | None (auto-discovered)                                             |

**Bob users:** create the symlink once per clone:

```bash
mkdir -p .bob && ln -s ../.agents/skills .bob/skills
```

**Available skills:** `/audit-markers`, `/skill-author`

## 5. Coding Standards
- **Types required** on all core functions
- **Docstrings are prompts** — be specific, the LLM reads them
- **Google-style docstrings** — `Args:` on the **class docstring only**; `__init__` gets a single summary sentence. Add `Attributes:` only when a stored value differs in type/behaviour from its constructor input (type transforms, computed values, class constants). See CONTRIBUTING.md for a full example.
- **Ruff** for linting/formatting
- Use `...` in `@generative` function bodies
- Prefer primitives over classes
- **Friendly Dependency Errors**: Wraps optional backend imports in `try/except ImportError` with a helpful message (e.g., "Please pip install mellea[hf]"). See `mellea/stdlib/session.py` for examples.
- **Backend telemetry fields**: All backends must populate `mot.usage` (dict with `prompt_tokens`, `completion_tokens`, `total_tokens`), `mot.model` (str), and `mot.provider` (str) in their `post_processing()` method. Metrics are automatically recorded by `TokenMetricsPlugin` — don't add manual `record_token_usage_metrics()` calls.

## 6. Commits & Hooks
[Angular format](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit): `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `release:`

Pre-commit runs: ruff, mypy, uv-lock, codespell

## 7. Timing
> **Don't cancel**: `pytest` (full) and `pre-commit --all-files` may take minutes. Canceling mid-run can corrupt state.

## 8. Common Issues
| Problem | Fix |
|---------|-----|
| `ComponentParseError` | Add examples to docstring |
| `uv.lock` out of sync | Run `uv sync` |
| Ollama refused | Run `ollama serve` |
| Telemetry import errors | Run `uv sync` to install OpenTelemetry deps |

## 9. Self-Review (before notifying user)
1. `uv run pytest test/ -m "not qualitative"` passes?
2. `ruff format` and `ruff check` clean?
3. New functions typed with concise docstrings?
4. Unit tests added for new functionality?
5. Avoided over-engineering?

## 10. Writing Tests

- Place tests in `test/` mirroring source structure
- Name files `test_*.py` (required for pydocstyle)
- Use `gh_run` fixture for CI-aware tests (see `test/conftest.py`)
- Mark tests checking LLM output quality with `@pytest.mark.qualitative`
- If a test fails, fix the **code**, not the test (unless the test was wrong)

## 11. Writing Docs

If you are modifying or creating pages under `docs/docs/`, follow the writing
conventions in [`docs/docs/guide/CONTRIBUTING.md`](docs/docs/guide/CONTRIBUTING.md).
Key rules that differ from typical Markdown habits:

- **No H1 in the body** — Mintlify renders the frontmatter `title` automatically;
  a body `# Heading` produces a duplicate title in the published site
- **No `.md` extensions in internal links** — use `../concepts/requirements-system`,
  not `../concepts/requirements-system.md`
- **Frontmatter required** — every page needs `title` and `description`; add
  `sidebarTitle` if the title is long
- **markdownlint gate** — run `npx markdownlint-cli2 "docs/docs/**/*.md"` and fix
  all warnings before committing a doc page
- **Verified code only** — every code example must be checked against the current
  mellea source; mark forward-looking content with `> **Coming soon:**`
- **No visible TODOs** — if content is missing, open a GitHub issue instead

## 12. Feedback Loop

Found a bug, workaround, or pattern? Update the docs:

- **Issue/workaround?** → Add to Section 7 (Common Issues) in this file
- **Usage pattern?** → Add to [`docs/AGENTS_TEMPLATE.md`](docs/AGENTS_TEMPLATE.md)
- **New pitfall?** → Add warning near relevant section
