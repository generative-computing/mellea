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

**Examples in `docs/examples/`** are opt-in — unlike `test/` files (auto-collected, default `unit`), examples require an explicit `# pytest:` comment to be collected. Files without this comment are silently ignored (they won't appear in skip summaries either). This is because examples have variable dependencies and limited setup:
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
- **CLI command docstrings**: Typer command functions in `cli/` follow an enriched convention with `Prerequisites:` and `See Also:` sections — these feed the auto-generated CLI reference page. See [`docs/docs/guide/CONTRIBUTING.md`](docs/docs/guide/CONTRIBUTING.md) for the full pattern. Regenerate after changes: `uv run poe clidocs`. Test the generator: `uv run pytest tooling/docs-autogen/test_cli_reference.py -v`. Full pipeline docs: [`tooling/docs-autogen/README.md`](tooling/docs-autogen/README.md).
- **Backend telemetry fields**: All backends must populate `mot.generation.usage` (dict with `prompt_tokens`, `completion_tokens`, `total_tokens`), `mot.generation.model` (str), and `mot.generation.provider` (str) in their `post_processing()` method. These fields live on `mot.generation`, a `GenerationMetadata` dataclass. `mot.generation.streaming` (bool) and `mot.generation.ttfb_ms` (float | None) are set automatically in `astream()` — backends do not need to set them. Metrics are automatically recorded by `TokenMetricsPlugin`, `LatencyMetricsPlugin`, and `ErrorMetricsPlugin` — don't add manual `record_token_usage_metrics()`, `record_request_duration()`, or `record_error()` calls.

## 6. Commits & Hooks
[Angular format](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit): `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `release:`

Pre-commit runs: ruff, mypy, uv-lock, codespell

For AI attribution trailers, see [Section 7 (AI Attribution)](#7-ai-attribution).

## 7. AI Attribution

Commits require a Signed-off-by trailer from the human author (added by running `git commit -s`). AI agents must not add a Signed-off-by in the tool's own name — instead, always add an `Assisted-by:` trailer to the commit footer:

```text
Assisted-by: Claude Code
Assisted-by: IBM Bob
```

Use the tool's common name (e.g., GitHub Copilot, Cursor, etc.).

## 8. Timing
> **Don't cancel**: `pytest` (full) and `pre-commit --all-files` may take minutes. Canceling mid-run can corrupt state.

## 9. Common Issues
| Problem | Fix |
|---------|-----|
| `ComponentParseError` | Add examples to docstring |
| `uv.lock` out of sync | Run `uv sync` |
| Ollama refused | Run `ollama serve` |
| Telemetry import errors | Run `uv sync` to install OpenTelemetry deps |

## 10. Self-Review (before notifying user)
1. `uv run pytest test/ -m "not qualitative"` passes?
2. `ruff format` and `ruff check` clean?
3. New functions typed with concise docstrings?
4. Unit tests added for new functionality?
5. Avoided over-engineering?

## 11. Writing Tests

- Place tests in `test/` mirroring source structure
- Name files `test_*.py` (required for pydocstyle)
- Use `gh_run` fixture for CI-aware tests (see `test/conftest.py`)
- Mark tests checking LLM output quality with `@pytest.mark.qualitative`
- If a test fails, fix the **code**, not the test (unless the test was wrong)

## 12. Writing Docs

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

## 13. Feedback Loop

Found a bug, workaround, or pattern? Update the docs:

- **Issue/workaround?** → Add to [Section 9 (Common Issues)](#9-common-issues) in this file
- **Usage pattern?** → Add to [`docs/AGENTS_TEMPLATE.md`](docs/AGENTS_TEMPLATE.md)
- **New pitfall?** → Add warning near relevant section

## 13. Working with Intrinsics

Intrinsics are specialized LoRA adapters that add task-specific capabilities (RAG evaluation, safety checks, calibration, etc.) to Granite models. Mellea handles adapter loading and input formatting automatically — you just call the right function.

### Using Intrinsics in Mellea

**Prefer the high-level wrappers** in `mellea/stdlib/components/intrinsic/`. These handle adapter loading, context formatting, and output parsing for you:

| Module | Function | Description |
|--------|----------|-------------|
| `core` | `check_certainty(context, backend)` | Model certainty about its last response (0–1) |
| `core` | `requirement_check(context, backend, requirement)` | Whether text meets a requirement (0–1) |
| `core` | `find_context_attributions(response, documents, context, backend)` | Sentences that influenced the response |
| `rag` | `check_answerability(question, documents, context, backend)` | Whether documents can answer a question (0–1) |
| `rag` | `rewrite_question(question, context, backend)` | Rewrite question into a retrieval query |
| `rag` | `clarify_query(question, documents, context, backend)` | Generate clarification or return "CLEAR" |
| `rag` | `find_citations(response, documents, context, backend)` | Document sentences supporting the response |
| `rag` | `check_context_relevance(question, document, context, backend)` | Whether a document is relevant; returns a string label (e.g. `'relevant'`, `'partially relevant'`, `'irrelevant'`) |
| `rag` | `flag_hallucinated_content(response, documents, context, backend)` | Flag potentially hallucinated sentences |

```python
from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.components import Message
from mellea.stdlib.components.intrinsic import core
from mellea.stdlib.context import ChatContext

backend = LocalHFBackend(model_id="ibm-granite/granite-4.1-3b")
context = (
    ChatContext()
    .add(Message("user", "What is the square root of 4?"))
    .add(Message("assistant", "The square root of 4 is 2."))
)
score = core.check_certainty(context, backend)
```

For lower-level control (custom adapters, model options), use `mfuncs.act()` with `Intrinsic` directly — see examples in `docs/examples/intrinsics/`.

### Project Resources

- **Canonical catalog**: `mellea/backends/adapters/catalog.py` — source of truth for intrinsic names, HF repo IDs, and adapter types
- **Usage examples**: `docs/examples/intrinsics/` — working code for every intrinsic
- **Helper functions**: `mellea/stdlib/components/intrinsic/rag.py` and `core.py`

### Adding New Intrinsics

When adding support for a new intrinsic (not just using an existing one), fetch its README from Hugging Face first. Each README contains the authoritative spec for input/output format, intended use, and examples.

**Writing examples?** The HF READMEs also document intended usage patterns and example inputs — useful reference when writing code in `docs/examples/intrinsics/`.

| Repo | Purpose | Intrinsics |
|------|---------|------------|
| [`ibm-granite/granitelib-rag-r1.0`](https://huggingface.co/ibm-granite/granitelib-rag-r1.0) | RAG pipeline | answerability, citations, context_relevance, hallucination_detection, query_rewrite, query_clarification |
| [`ibm-granite/granitelib-core-r1.0`](https://huggingface.co/ibm-granite/granitelib-core-r1.0) | Core capabilities | context-attribution, requirement-check, uncertainty |
| [`ibm-granite/granitelib-guardian-r1.0`](https://huggingface.co/ibm-granite/granitelib-guardian-r1.0) | Safety & compliance | guardian-core, policy-guardrails, factuality-detection, factuality-correction |

**README URLs** — RAG intrinsics (no model subfolder):
```
https://huggingface.co/ibm-granite/granitelib-rag-r1.0/blob/main/{intrinsic_name}/README.md
```

Core and Guardian intrinsics (include model subfolder):
```
https://huggingface.co/ibm-granite/granitelib-{core,guardian,rag}-r1.0/blob/main/{intrinsic_name}/granite-4.1-{3b,8b,30b}/{lora,alora}/README.md
```
