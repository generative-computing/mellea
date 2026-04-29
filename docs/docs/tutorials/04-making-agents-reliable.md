---
canonical: "https://docs.mellea.ai/tutorials/04-making-agents-reliable"
title: "Tutorial: Making Agents Reliable"
description: "Add requirements validation and Guardian safety checks to a ReACT tool-using agent."
# diataxis: tutorial
---

This tutorial shows how to build a tool-using agent with Mellea and progressively
add reliability layers: output requirements, retry budgets, and Guardian safety
checks that detect harmful or off-topic responses before they reach your users.

By the end you will have covered:

- Building a tool-using agent with `instruct()` and `ModelOption.TOOLS`
- Enforcing structured output with requirements and a retry budget
- Inspecting `SamplingResult` to understand failures
- Detecting harmful outputs with `GuardianCheck`
- Grounding safety checks against retrieved context

**Prerequisites:** [Tutorial 02](./02-streaming-and-async) and
[Tutorial 03](./03-using-generative-stubs) complete,
`pip install mellea`, Ollama running locally with `granite4:micro` downloaded.

---

## Step 1: A simple tool-using agent

Start with two tools — a search stub and a calculator — and wire them into an
`instruct()` call:

```python
# Requires: mellea
# Returns: str
import mellea
from mellea.backends import ModelOption, tool

@tool
def web_search(query: str) -> str:
    """Search the web for information about a topic.

    Args:
        query: The search query.
    """
    # Stub — replace with a real search client in production.
    return f"Top result for '{query}': Mellea is a Python framework for generative programs."

@tool(name="calculator")
def calculate(expression: str) -> str:
    """Evaluate a safe arithmetic expression and return the result as a string.

    Args:
        expression: An arithmetic expression, e.g. '12 * 7 + 3'.
    """
    allowed = set("0123456789 +-*/(). ")
    if not all(c in allowed for c in expression):
        return "Error: expression contains disallowed characters."
    return str(eval(expression))  # noqa: S307 — only safe characters pass the guard above

m = mellea.start_session()

response = m.instruct(
    "What is Mellea, and how many characters are in the word 'Mellea'?",
    model_options={ModelOption.TOOLS: [web_search, calculate]},
)
print(str(response))
# Output will vary — LLM responses depend on model and temperature.
```

The model can call either or both tools during its response. With no requirements
attached, the output format is up to the model.

---

## Step 2: Adding output requirements

Require the agent to format its answer as a short structured response:

```python
# Requires: mellea
# Returns: str
import mellea
from mellea.backends import ModelOption, tool
from mellea.stdlib.requirements import req, simple_validate

@tool
def web_search(query: str) -> str:
    """Search the web for information about a topic.

    Args:
        query: The search query.
    """
    return f"Top result for '{query}': Mellea is a Python framework for generative programs."

@tool(name="calculator")
def calculate(expression: str) -> str:
    """Evaluate a safe arithmetic expression.

    Args:
        expression: An arithmetic expression.
    """
    allowed = set("0123456789 +-*/(). ")
    if not all(c in allowed for c in expression):
        return "Error: expression contains disallowed characters."
    return str(eval(expression))  # noqa: S307

m = mellea.start_session()

response = m.instruct(
    "What is Mellea, and how many characters are in the word 'Mellea'?",
    model_options={ModelOption.TOOLS: [web_search, calculate]},
    requirements=[
        req("The response must answer both questions."),
        req(
            "The response must be 50 words or fewer.",
            validation_fn=simple_validate(
                lambda x: (
                    len(x.split()) <= 50,
                    f"Response is {len(x.split())} words; must be 50 or fewer.",
                )
            ),
        ),
    ],
)
print(str(response))
# Output will vary — LLM responses depend on model and temperature.
```

The word-count requirement runs deterministically. The "answer both questions"
requirement falls back to LLM-as-a-judge. If either fails, Mellea retries with
the failure reason embedded in the repair request.

---

## Step 3: Inspecting failures and handling a retry budget

Use `RejectionSamplingStrategy` with `return_sampling_results=True` to observe
what happens when requirements fail:

```python
# Requires: mellea
# Returns: None
import mellea
from mellea.backends import ModelOption, tool
from mellea.stdlib.requirements import req, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy

@tool
def web_search(query: str) -> str:
    """Search the web for information about a topic.

    Args:
        query: The search query.
    """
    return f"Top result for '{query}': Mellea is a Python framework for generative programs."

@tool(name="calculator")
def calculate(expression: str) -> str:
    """Evaluate a safe arithmetic expression.

    Args:
        expression: An arithmetic expression.
    """
    allowed = set("0123456789 +-*/(). ")
    if not all(c in allowed for c in expression):
        return "Error: expression contains disallowed characters."
    return str(eval(expression))  # noqa: S307

m = mellea.start_session()

result = m.instruct(
    "What is Mellea, and how many characters are in the word 'Mellea'?",
    model_options={ModelOption.TOOLS: [web_search, calculate]},
    requirements=[
        req("The response must answer both questions."),
        req(
            "The response must be 50 words or fewer.",
            validation_fn=simple_validate(
                lambda x: (
                    len(x.split()) <= 50,
                    f"Response is {len(x.split())} words; must be 50 or fewer.",
                )
            ),
        ),
    ],
    strategy=RejectionSamplingStrategy(loop_budget=3),
    return_sampling_results=True,
)

if result.success:
    print("Passed:", str(result.result))
else:
    print(f"All {len(result.sample_generations)} attempts failed.")
    for i, attempt in enumerate(result.sample_generations):
        print(f"  Attempt {i + 1}: {str(attempt.value)[:80]}...")
```

`result.success` is `True` when at least one attempt satisfied all requirements.
`result.sample_generations` gives you every attempt in order — useful for
debugging or for choosing the best available output when the budget runs out.

---

## Step 4: Adding Guardian harm detection

[`GuardianCheck`](../reference/glossary#guardiancheck) wraps a `MelleaSession` call and evaluates the output against a
set of [`GuardianRisk`](../reference/glossary#guardianrisk) category. Run it after your agent responds to flag outputs before
they reach downstream code.

```python
# Requires: mellea
# Returns: None
import mellea
from mellea.backends import ModelOption, tool
from mellea.stdlib.requirements import req, simple_validate
from mellea.stdlib.requirements.safety.guardian import GuardianCheck, GuardianRisk
from mellea.stdlib.sampling import RejectionSamplingStrategy

@tool
def web_search(query: str) -> str:
    """Search the web for information about a topic.

    Args:
        query: The search query.
    """
    return f"Top result for '{query}': Mellea is a Python framework for generative programs."

@tool(name="calculator")
def calculate(expression: str) -> str:
    """Evaluate a safe arithmetic expression.

    Args:
        expression: An arithmetic expression.
    """
    allowed = set("0123456789 +-*/(). ")
    if not all(c in allowed for c in expression):
        return "Error: expression contains disallowed characters."
    return str(eval(expression))  # noqa: S307

m = mellea.start_session()

response = m.instruct(
    "What is Mellea, and how many characters are in the word 'Mellea'?",
    model_options={ModelOption.TOOLS: [web_search, calculate]},
    requirements=[
        req("The response must answer both questions."),
        req(
            "The response must be 50 words or fewer.",
            validation_fn=simple_validate(
                lambda x: (
                    len(x.split()) <= 50,
                    f"Response is {len(x.split())} words; must be 50 or fewer.",
                )
            ),
        ),
    ],
    strategy=RejectionSamplingStrategy(loop_budget=3),
)

output_text = str(response)

# Run Guardian checks on the agent output.
harm_check = GuardianCheck(
    GuardianRisk.HARM,
    backend_type="ollama",
    ollama_url="http://localhost:11434",
)
jailbreak_check = GuardianCheck(
    GuardianRisk.JAILBREAK,
    backend_type="ollama",
    ollama_url="http://localhost:11434",
)

# session.validate() returns a list of ValidationResult objects.
validation_results = m.validate([harm_check, jailbreak_check])

safe = all(r._result for r in validation_results)
if safe:
    print("Output passed safety checks:", output_text)
else:
    for check_result in validation_results:
        if not check_result._result:
            print(f"Safety check failed — {check_result._reason}")
```

> **Note:** `m.validate()` evaluates the checks against the most recent session
> output. Run it immediately after the `instruct()` call before any other session
> activity modifies the context.

Each `GuardianCheck` runs as an independent inference call against your local
Ollama instance. The results are `ValidationResult` objects with `._result`
(bool) and `._reason` (str).

---

## Step 5: Sharing a backend across Guardian checks

When you run multiple `GuardianCheck` instances, each one loads or contacts the
model separately by default. Pass `backend=shared_backend` to reuse a single
loaded backend and avoid the overhead of repeated initialisation:

```python
# Requires: mellea
# Returns: None
import mellea
from mellea.backends import ModelOption, model_ids, tool
from mellea.backends.ollama import OllamaModelBackend
from mellea.stdlib.requirements.safety.guardian import GuardianCheck, GuardianRisk

@tool
def web_search(query: str) -> str:
    """Search the web for information about a topic.

    Args:
        query: The search query.
    """
    return f"Top result for '{query}': Mellea is a Python framework for generative programs."

m = mellea.start_session()

response = m.instruct(
    "What is Mellea?",
    model_options={ModelOption.TOOLS: [web_search]},
)

# Create a single Guardian backend and reuse it across all checks.
# Pull the model first: ollama pull granite3-guardian:2b
guardian_backend = OllamaModelBackend(model_ids.IBM_GRANITE_GUARDIAN_3_0_2B.ollama_name)

checks = [
    GuardianCheck(GuardianRisk.HARM, backend=guardian_backend),
    GuardianCheck(GuardianRisk.PROFANITY, backend=guardian_backend),
    GuardianCheck(GuardianRisk.ANSWER_RELEVANCE, backend=guardian_backend),
    GuardianCheck(GuardianRisk.JAILBREAK, backend=guardian_backend),
]

results = m.validate(checks)

for risk, result in zip(checks, results):
    status = "PASS" if result._result else "FAIL"
    print(f"[{status}] {risk}: {result._reason or 'ok'}")
```

The full list of `GuardianRisk` values you can check:
`HARM`, `GROUNDEDNESS`, `PROFANITY`, `ANSWER_RELEVANCE`, `JAILBREAK`,
`FUNCTION_CALL`, `SOCIAL_BIAS`, `VIOLENCE`, `SEXUAL_CONTENT`,
`UNETHICAL_BEHAVIOR`.

---

## Step 6: Groundedness checks with retrieved context

When your agent retrieves documents before answering, add a `GROUNDEDNESS` check
to confirm the response is grounded in what was retrieved rather than
hallucinated:

```python
# Requires: mellea
# Returns: None
import mellea
from mellea.backends import ModelOption, tool
from mellea.stdlib.requirements.safety.guardian import GuardianCheck, GuardianRisk

RETRIEVED_CONTEXT = (
    "Mellea is an open-source Python framework for building generative programs. "
    "It provides instruct(), @generative, and @mify as its core primitives. "
    "Mellea is backend-agnostic and supports Ollama, OpenAI, and custom backends."
)

@tool
def retrieve_docs(topic: str) -> str:
    """Retrieve documentation about a topic.

    Args:
        topic: The topic to retrieve documentation for.
    """
    # In production, call your vector store or search index here.
    return RETRIEVED_CONTEXT

m = mellea.start_session()

response = m.instruct(
    "Using the retrieved documentation, describe what Mellea is.",
    model_options={ModelOption.TOOLS: [retrieve_docs]},
    grounding_context={"docs": RETRIEVED_CONTEXT},
)

output_text = str(response)

# Check the response is grounded in the retrieved context.
groundedness_check = GuardianCheck(
    GuardianRisk.GROUNDEDNESS,
    backend_type="ollama",
    ollama_url="http://localhost:11434",
    context_text=RETRIEVED_CONTEXT,
)

results = m.validate([groundedness_check])
grounded = results[0]._result

if grounded:
    print("Grounded response:", output_text)
else:
    print("Response may contain hallucinated content.")
    print("Reason:", results[0]._reason)
```

> **Tip:** Pass the same text you supplied as `grounding_context` to
> `context_text` in `GuardianCheck`. This ensures the groundedness model
> evaluates the response against exactly what the agent was given.

---

## Step 7: A ReACT agent with Guardian checks

For goal-driven agentic loops, combine `react()` with Guardian validation. The
`react()` function is an async built-in that runs the Reason-Act loop until the
goal is reached or the step budget is exhausted:

```python
# Requires: mellea
# Returns: None
import asyncio
import mellea
from mellea.backends import tool
from mellea.stdlib.context import ChatContext
from mellea.stdlib.frameworks.react import react
from mellea.stdlib.requirements.safety.guardian import GuardianCheck, GuardianRisk

@tool
def web_search(query: str) -> str:
    """Search the web for information about a topic.

    Args:
        query: The search query.
    """
    return f"Search result for '{query}': Mellea is a Python framework."

@tool(name="calculator")
def calculate(expression: str) -> str:
    """Evaluate a safe arithmetic expression.

    Args:
        expression: An arithmetic expression.
    """
    allowed = set("0123456789 +-*/(). ")
    if not all(c in allowed for c in expression):
        return "Error: expression contains disallowed characters."
    return str(eval(expression))  # noqa: S307

m = mellea.start_session()

async def run_agent(goal: str) -> str:
    result, _ = await react(
        goal=goal,
        context=ChatContext(),
        backend=m.backend,
        tools=[web_search, calculate],
    )
    return str(result)

output = asyncio.run(run_agent(
    "Find out what Mellea is, then calculate how many characters are in 'Mellea'."
))

# Validate the agent's final output.
harm_check = GuardianCheck(
    GuardianRisk.HARM,
    backend_type="ollama",
    ollama_url="http://localhost:11434",
)
results = m.validate([harm_check])

if results[0]._result:
    print("Agent output (safe):", output)
else:
    print("Agent output flagged:", results[0]._reason)
# Output will vary — LLM responses depend on model and temperature.
```

> **Advanced:** `react()` implements the Reason + Act loop: the LLM alternates
> between producing a reasoning step ("Thought") and invoking a tool ("Action")
> until it determines the goal is satisfied or the step budget runs out. You can
> inspect the intermediate steps via the second return value (the trace list).
> For fine-grained control over each reasoning step, build a custom loop using
> `m.instruct()` with `ModelOption.TOOLS` directly.

---

## What you built

A progression from a basic tool-using agent to a safety-validated, grounded
agentic system:

| Layer | What it adds |
| --- | --- |
| `instruct()` + `ModelOption.TOOLS` | LLM can call Python tools |
| `requirements` + `simple_validate` | Deterministic and LLM-judged output constraints |
| `RejectionSamplingStrategy` | Explicit retry budget |
| `return_sampling_results=True` | Inspect every attempt for debugging |
| `GuardianCheck` | Post-generation safety risk detection |
| Shared `backend` | Amortise model loading across multiple checks |
| `GuardianRisk.GROUNDEDNESS` + `context_text` | Detect hallucination relative to retrieved context |
| `react()` | Goal-driven multi-step agentic loop |

---

**See also:** [The Requirements System](../concepts/requirements-system) | [Security and Taint Tracking](../advanced/security-and-taint-tracking) | [Tools and Agents](../how-to/tools-and-agents)
