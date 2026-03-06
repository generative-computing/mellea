---
title: "FAQ"
description: "Answers to frequently asked questions about Mellea installation, backends, and generative functions."
# diataxis: reference
---

## Why does `start_session()` fail with a connection error?

Mellea's default backend is Ollama. If Ollama is not running, any call that
reaches the backend raises a connection error:

```text
ConnectionError: Could not connect to Ollama at http://localhost:11434
```

Start Ollama and try again:

```bash
ollama serve
```

Verify the server is reachable before running your script:

```bash
curl http://localhost:11434/api/version
```

If Ollama is running on a non-default host or port, pass the URL explicitly:

```python
from mellea.backends.ollama import OllamaModelBackend
from mellea import MelleaSession
from mellea.stdlib.context import SimpleContext

m = MelleaSession(
    backend=OllamaModelBackend(base_url="http://my-ollama-host:11434"),
    ctx=SimpleContext(),
)
```

## How do I use a model other than `granite4:micro`?

Pass the `model_id` parameter to `start_session()`:

```python
from mellea import start_session

with start_session(model_id="llama3.2:latest") as m:
    response = m.chat("What is 1+1?")
    print(response)
```

Pull the model with Ollama before using it:

```bash
ollama pull llama3.2:latest
```

You can also pass a backend instance directly to `MelleaSession` for full
control over backend options:

```python
from mellea import MelleaSession
from mellea.backends.ollama import OllamaModelBackend
from mellea.stdlib.context import SimpleContext

m = MelleaSession(
    backend=OllamaModelBackend("mistral:latest"),
    ctx=SimpleContext(),
)
```

## Can I use Mellea without Ollama?

Yes. Ollama is the default backend but not the only one. Mellea ships with
backends for OpenAI-compatible APIs, HuggingFace local inference, IBM WatsonX,
and LiteLLM (which itself proxies dozens of providers).

Install the backend you need:

```bash
pip install mellea[litellm]    # LiteLLM multi-provider
pip install mellea[hf]         # HuggingFace / local inference
pip install mellea[watsonx]    # IBM WatsonX
```

Then pass the backend to `start_session()` or `MelleaSession`:

```python
from mellea import start_session

# OpenAI
with start_session(backend_name="openai", model_id="gpt-4o") as m:
    print(m.chat("Hello!"))

# LiteLLM wrapping Anthropic
from mellea.backends.litellm import LiteLLMBackend
from mellea import MelleaSession
from mellea.stdlib.context import SimpleContext

m = MelleaSession(
    backend=LiteLLMBackend("anthropic/claude-3-5-sonnet-20241022"),
    ctx=SimpleContext(),
)
```

See [Common Errors](../troubleshooting/common-errors) for help installing
backend-specific dependencies.

## Why does my `@generative` function return the wrong type?

The `@generative` decorator uses the function's docstring as the prompt. If the
docstring is vague, the model may return output that cannot be parsed into the
declared return type.

Compare these two definitions:

```python
from mellea import generative

# Vague — the model may return extra explanation text
@generative
def extract_keywords(text: str) -> list[str]:
    """Extract keywords."""

# Specific — the model knows exactly what format is expected
@generative
def extract_keywords(text: str) -> list[str]:
    """Extract the five most important keywords from the text.
    Return only a Python list of strings with no extra commentary.
    Example output: ["machine learning", "neural networks", "training"]
    """
```

For stricter guarantees, add requirements:

```python
from mellea import generative
from mellea.stdlib.requirements import req

@generative
def classify(text: str) -> str:
    """Classify the sentiment of the text. Return only one word:
    positive, negative, or neutral."""

with start_session() as m:
    result = classify(
        m,
        text="This product is great!",
        requirements=[req("Must be one of: positive, negative, neutral")],
    )
```

If the function raises `ComponentParseError`, add an example to the docstring
— the model needs a concrete illustration of the expected format.

## What is the difference between `instruct()` and `@generative`?

Both call the LLM, but they differ in when you write the prompt and how you
pass variables.

`instruct()` takes a prompt string with `{{variable}}` placeholders at call
time. It is best for one-off instructions where the prompt text varies:

```python
from mellea import start_session

with start_session() as m:
    result = m.instruct(
        "Translate the following into {{language}}: {{text}}",
        user_variables={"language": "French", "text": "Hello, world!"},
    )
```

`@generative` defines the prompt once in the function's docstring. It is best
when you want a reusable, typed, unit-testable function:

```python
from mellea import generative

@generative
def translate(text: str, language: str) -> str:
    """Translate text into the specified language.
    Return only the translated text, with no explanation.
    """

with start_session() as m:
    result = translate(m, text="Hello, world!", language="French")
```

`@generative` functions also participate in Mellea's lazy evaluation graph,
which means you can feed a thunk from one generative call into another before
either has been evaluated.

## Why do requirements keep failing?

When the model keeps retrying but the output looks correct, one of the following
is usually the cause:

- **The requirement is too strict.** A requirement like "Must be exactly 17
  syllables" is difficult for a model to satisfy reliably. Relax the constraint
  or provide the model with more context.
- **The default budget is too low.** `instruct()` defaults to `loop_budget=2`.
  Increase it:

  ```python
  from mellea import start_session
  from mellea.stdlib.requirements import req
  from mellea.stdlib.sampling import RejectionSamplingStrategy

  with start_session() as m:
      result = m.instruct(
          "Write a haiku about autumn.",
          requirements=[req("Must be exactly 17 syllables")],
          strategy=RejectionSamplingStrategy(loop_budget=5),
      )
  ```

- **The validation function is wrong.** If you are using a custom verifier,
  check it returns `True` for valid output. Use `return_sampling_results=True`
  to inspect each attempt:

  ```python
  result = m.instruct(
      "Write a haiku about autumn.",
      requirements=[req("Must be exactly 17 syllables")],
      return_sampling_results=True,
  )
  print(f"Success: {result.success}")
  for attempt, (gen, vals) in enumerate(
      zip(result.sample_generations, result.sample_validations), 1
  ):
      print(f"Attempt {attempt}: {gen.value!r}")
      for requirement, validation in vals:
          print(f"  {requirement.description}: {validation._result}")
  ```

## How do I see what the model is actually receiving?

Use `GenerateLog` to capture the rendered prompt. Enable application tracing or
backend tracing and check the `response` and `gen_ai.usage.input_tokens`
attributes on the spans.

For a quick local inspection without a trace backend, enable console tracing:

```bash
export MELLEA_TRACE_BACKEND=true
export MELLEA_TRACE_CONSOLE=true
python your_script.py
```

Each backend span prints the operation name, model ID, and token counts.

Alternatively, inspect the `GenerateLog` objects returned with sampling results:

```python
from mellea import start_session
from mellea.stdlib.requirements import req

with start_session() as m:
    result = m.instruct(
        "Summarise in one sentence: {{text}}",
        user_variables={"text": "Long article content here."},
        return_sampling_results=True,
    )
    for log in result.generate_logs:
        print(log.prompt)
        print(log.backend)
```

For the full telemetry setup, see
[OpenTelemetry Tracing](../evaluation-and-observability/opentelemetry-tracing).

## Does Mellea support async?

Yes. Every synchronous method has an async counterpart:

| Sync | Async |
| ---- | ----- |
| `m.chat()` | `await m.achat()` |
| `m.instruct()` | `await m.ainstruct()` |
| `m.act()` | `await m.aact()` |
| `mfuncs.act()` | `await mfuncs.aact()` |

`@generative` functions work in async context when you await them:

```python
import asyncio
from mellea import generative, start_session

@generative
async def summarise(text: str) -> str:
    """Summarise the text in one sentence."""

async def main() -> None:
    with start_session() as m:
        result = await summarise(m, text="Long article content here.")
        print(result)

asyncio.run(main())
```

> **Note:** If you are inside a Jupyter notebook, the event loop is already
> running. Use `await` directly or install `nest_asyncio` to allow nested loops.

## How do I contribute?

Read the contributing guide first:

```bash
cat docs/docs/guide/CONTRIBUTING.md
```

The short version:

1. Fork the repository and clone it.
2. Install dependencies: `uv sync --all-extras --all-groups`
3. Install pre-commit hooks: `pre-commit install`
4. Create a branch: `git checkout -b feat/your-feature`
5. Run tests: `uv run pytest -m "not qualitative"`
6. Open a pull request.

All commits use Angular format (`feat:`, `fix:`, `docs:`, `refactor:`). Pre-commit
runs ruff, mypy, and codespell automatically.

## Where can I get help?

- **GitHub Issues:** Report bugs and request features at the project's GitHub
  Issues page.
- **GitHub Discussions:** Ask questions and share ideas in the Discussions tab.
- **Examples:** The `docs/examples/` directory contains runnable examples
  covering every major feature.
- **Common Errors:** See [Common Errors](../troubleshooting/common-errors) for
  a reference table of known error messages and fixes.

---

## See also

- [Common Errors](../troubleshooting/common-errors) — a reference table of
  error messages, diagnostic steps, and fixes.
- [Quick Start](../getting-started/quickstart) — install Mellea and run your
  first generative function.
