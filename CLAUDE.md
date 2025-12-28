# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Setup

Install dependencies using `uv`:
```bash
uv venv .venv && source .venv/bin/activate
uv sync --all-extras --all-groups
```

Or with conda:
```bash
conda/install.sh
```

Install pre-commit hooks:
```bash
pre-commit install
```

## Common Commands

### Linting and Formatting
- **Run ruff formatter**: `ruff format mellea test cli docs`
- **Run ruff linter**: `ruff check mellea --fix`
- **Run mypy type checking**: `uv run mypy mellea`
- **Run all linting**: `pre-commit run --all-files`

### Testing
- **Run all tests**: `uv run pytest test/`
- **Run specific test**: `uv run pytest test/stdlib_basics/test_something.py`
- **Run single test function**: `uv run pytest test/stdlib_basics/test_something.py::test_function_name`
- **Run tests excluding qualitative tests** (for local testing): `uv run pytest test/ -m "not qualitative"`

### CLI Tools
The `m` CLI provides several commands:
- **`m serve`**: Serve a generative program as an OpenAI-compatible model endpoint
- **`m decompose`**: Decompose prompts (helps break down complex prompts)
- **`m alora train`**: Train adaptive LoRA adapters
- **`m alora upload`**: Upload trained adapters
- **`m eval`**: Test-based evaluation framework

Run `m --help` for full command details.

## Architecture Overview

### Core Components

**MelleaSession** (`mellea/stdlib/session.py`): The main interface for building generative programs. Sessions manage backend connections and context. Key methods:
- `instruct()`: Generate text based on instructions
- `chat()`: Multi-turn chat interactions
- `query()`: General LLM queries
- `transform()`: Apply transformations to inputs

**Backends** (`mellea/backends/`): Pluggable inference providers
- `OllamaModelBackend`: Local models via Ollama
- `OpenAIBackend`: OpenAI API
- `HuggingFace` (LocalHFBackend): Local HF transformers
- `WatsonX`: IBM WatsonX
- `LiteLLM`: LiteLLM wrapper supporting multiple providers
- `VLLMBackend`: vLLM backend

Each backend implements model options via `ModelOption` class in `mellea/backends/types.py` (e.g., `TEMPERATURE`, `MAX_NEW_TOKENS`, `SYSTEM_PROMPT`).

**Components and Base Classes** (`mellea/stdlib/base.py`): Core data structures
- `CBlock`: Content blocks (strings with metadata)
- `Component`: Base class for generative operations
- `Context`: Manages prompt context and caching
- `ModelOutputThunk`: Wraps model outputs; stores `.value` property for accessing the result

**Requirements and Validation** (`mellea/stdlib/requirement.py`): Constraint checking
- `Requirement`: Define requirements on model outputs
- `ValidationResult`: Track validation outcomes
- Supports both programmatic validation and LLM-as-a-judge validation

**Sampling Strategies** (`mellea/stdlib/sampling/`): Inference-time scaling
- `RejectionSamplingStrategy`: Retry until requirements met
- `MajorityVotingStrategy`: Sample multiple outputs and vote
- Custom strategies extend `SamplingStrategy` base

**Generative Slots** (`mellea/stdlib/genslot.py`): Mark functions to be executed by LLM
- Use `@generative` decorator to create LLM-powered functions
- Return types are automatically converted to Pydantic models for schema extraction

**Functional API** (`mellea/stdlib/functional.py`): Low-level operations used internally by session methods

**Prompt Templates** (`mellea/templates/`): Jinja2-based prompt templates
- Each component has associated templates
- User variables can be injected via `user_variables` parameter in session methods

### Standard Library Organization

- `stdlib/base.py`: Core classes (CBlock, Component, Context, ModelOutputThunk)
- `stdlib/chat.py`: Message and conversation classes
- `stdlib/genslot.py`: Generative slot decorator and infrastructure
- `stdlib/instruction.py`: Instruction components
- `stdlib/requirement.py`: Requirement definitions and validation
- `stdlib/sampling/`: Sampling strategies (base, rejection sampling, majority voting)
- `stdlib/intrinsics/`: Specialized components (e.g., structured output, code generation)
- `stdlib/rewards/`: Reward models and scoring functions
- `stdlib/safety/`: Safety checking utilities
- `stdlib/mify.py`: Integration with legacy code (mify decorator)
- `stdlib/mobject.py`: Generative object-oriented programming support
- `stdlib/test_based_eval.py`: LLM-as-a-judge evaluation framework
- `stdlib/tools/`: Tool calling utilities for function-based LLM interaction

### Key Patterns

**Instruct-Validate-Repair**: Core pattern for robust generation
1. Instruct: Generate output with requirements specified
2. Validate: Check if output meets requirements
3. Repair: Retry or repair if validation fails

Example:
```python
m = mellea.start_session()
result = m.instruct(
    "Write an email",
    requirements=["Include a greeting", "Keep under 100 words"],
    strategy=RejectionSamplingStrategy(loop_budget=3)
)
```

**Context Management**: Control memory efficiency and KV-cache reuse
- Components manage context for individual calls
- `Context` class enables reusing context across multiple calls
- `mify()` integrates with legacy code while managing context

**Sampling and Verification**: Improve output quality
- Use sampling strategies to retry or aggregate outputs
- Combine programmatic and LLM-based validation
- Leverage multiple verifiers for different requirement types

## Testing Strategy

- **Test structure**: Tests follow pytest conventions in `test/` directory
- **Marker-based**: Tests marked with `qualitative` require exact LLM outputs (xfail in CI)
- **Fixture-based**: `test/conftest.py` provides common fixtures for sessions and backends
- **Async tests**: Use `pytest-asyncio` for async operations; asyncio mode is auto

## Code Quality Standards

Ruff is configured with:
- **Format**: Skip magic trailing comma
- **Linting rules**: Google-style docstrings (D convention), pyflakes, isort, pandas-vet, async checks
- **Max complexity**: 20 (McCabe)
- **Type checking**: MyPy with disabled error codes for empty bodies and untyped imports

Important notes:
- Docstring convention: Google-style (specified in `pyproject.toml`)
- Ignored warnings: `RUF001` (unicode), `C408`, `E501`, `F401`, `F811`, `PL`, `RUF012`, `PD901`, `C901`
- Codespell configured with custom ignored words (`mellea`, `asai`, etc.)

## CLI Structure

The `m` CLI is built with Typer (`cli/m.py`):
- Subcommands organized in separate modules (alora, decompose, eval, serve)
- Each subcommand is a Typer app added with `cli.add_typer()`
- Single commands added with `cli.command()`

Entry point: `pyproject.toml` defines `m = "cli.m:cli"`

## Package Structure

```
mellea/
├── backends/       # Inference providers (OpenAI, Ollama, HF, etc.)
├── stdlib/         # Standard library components and utilities
├── helpers/        # Utility functions and logging
└── templates/      # Jinja2 prompt templates

cli/
├── m.py           # Main CLI entrypoint
├── serve/         # OpenAI-compatible serving
├── alora/         # LoRA training and upload
├── decompose/     # Prompt decomposition
└── eval/          # Evaluation framework

test/
├── stdlib_basics/        # Core functionality tests
├── backends/             # Backend-specific tests
├── stdlib_intrinsics/    # Specialized component tests
└── conftest.py           # Shared pytest fixtures
```

## Important Implementation Details

### Model Options Pattern
Use `ModelOption` sentinel values (wrapped with `@@@`) for backend-agnostic model configuration:
```python
from mellea.backends.types import ModelOption
model_options = {
    ModelOption.TEMPERATURE: 0.7,
    ModelOption.MAX_NEW_TOKENS: 500,
    ModelOption.SYSTEM_PROMPT: "You are helpful"
}
```

### Session Context Variable
Sessions are stored in a context variable (`_context_session`) to enable convenience functions like `instruct()` without passing the session object. Use `get_session()` to retrieve the current session within a context.

### Pre-commit Hooks
The repository has strict pre-commit checks:
- Ruff format and lint (with auto-fix on mellea/ only)
- MyPy type checking (on entire mellea/)
- UV lock file updates
- Codespell for typo detection

Hooks are applied to specific file patterns defined in `.pre-commit-config.yaml`. The `scratchpad/` directory is excluded.

### Dependencies
- **Core**: pydantic, openai, jinja2, requests, fastapi, typer, ollama, torch, transformers
- **Optional**: vllm, HuggingFace extras (for LoRA), docling, watsonx, litellm
- **Dev**: pytest, mypy, ruff, pre-commit, pylint

Use `uv sync --all-extras --all-groups` to install everything for development.

## Debugging and Introspection

- `FancyLogger` provides enhanced logging throughout the codebase
- Component execution traces are available via `GenerateLog`
- Context provides insight into prompt construction and KV-cache management
- Backend calls can be inspected for debugging model options and request/response details
