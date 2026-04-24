---
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
- Detecting harmful outputs with Guardian Intrinsics
- Grounding safety checks against retrieved context

**Prerequisites:** [Tutorial 02](./02-streaming-and-async) and
[Tutorial 03](./03-using-generative-stubs) complete,
`pip install mellea`, Ollama running locally with `granite4:micro` downloaded.
Steps 4–7 additionally require `pip install "mellea[hf]"` — the Guardian Intrinsics
use a `LocalHFBackend` for evaluation (approximately 3 GB download on first run).

---

## Step 1: A simple tool-using agent

Start with two tools — a search stub and a calculator — and wire them into an
`instruct()` call:

```python
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

## Step 4: Safety checks with Guardian Intrinsics

Guardian Intrinsics use a HuggingFace backend loaded with Granite Guardian adapters
to evaluate LLM outputs. Each call returns a float score: `0.0` means safe, `1.0`
means risk detected. Use `0.5` as the threshold.

Pass `ctx=ChatContext()` to `start_session()` so the session context is available for
evaluation directly after generation:

```python
import mellea
from mellea.backends import ModelOption, tool
from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.components import Message
from mellea.stdlib.components.intrinsic import guardian
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements import req, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy
from typing import cast

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

m = mellea.start_session(ctx=ChatContext())

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

# Evaluate the session context — it already contains the full conversation.
guardian_backend = LocalHFBackend(model_id="ibm-granite/granite-4.0-micro")
eval_ctx = cast(ChatContext, m.ctx)

harm_score = guardian.guardian_check(eval_ctx, guardian_backend, criteria="harm")
jailbreak_score = guardian.guardian_check(eval_ctx, guardian_backend, criteria="jailbreak")

print(f"Harm:      {harm_score:.4f}  ({'RISK' if harm_score >= 0.5 else 'SAFE'})")
print(f"Jailbreak: {jailbreak_score:.4f}  ({'RISK' if jailbreak_score >= 0.5 else 'SAFE'})")

if harm_score < 0.5 and jailbreak_score < 0.5:
    print("Output passed safety checks:", str(response))
# Example output:
# Harm:      0.0018  (SAFE)
# Jailbreak: 0.0011  (SAFE)
# Output passed safety checks: Mellea is a Python framework for building ...
```

`guardian_check()` evaluates the last message from `target_role` in the context.
The default `target_role="assistant"` checks the most recent assistant response.

---

## Step 5: Checking multiple risk criteria

`CRITERIA_BANK` contains 10 pre-baked criteria keys. Reuse a single
`guardian_backend` across all checks to avoid repeated model initialisation:

```python
import mellea
from mellea.backends import ModelOption, tool
from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.components.intrinsic import guardian
from mellea.stdlib.context import ChatContext
from typing import cast

@tool
def web_search(query: str) -> str:
    """Search the web for information about a topic.

    Args:
        query: The search query.
    """
    return f"Top result for '{query}': Mellea is a Python framework for generative programs."

m = mellea.start_session(ctx=ChatContext())
m.instruct("What is Mellea?", model_options={ModelOption.TOOLS: [web_search]})

guardian_backend = LocalHFBackend(model_id="ibm-granite/granite-4.0-micro")
eval_ctx = cast(ChatContext, m.ctx)

for criteria in ["harm", "jailbreak", "profanity", "social_bias", "answer_relevance"]:
    score = guardian.guardian_check(eval_ctx, guardian_backend, criteria=criteria)
    status = "RISK" if score >= 0.5 else "SAFE"
    print(f"[{status}] {criteria}: {score:.4f}")
# Example output:
# [SAFE] harm:             0.0021
# [SAFE] jailbreak:        0.0013
# [SAFE] profanity:        0.0009
# [SAFE] social_bias:      0.0031
# [SAFE] answer_relevance: 0.0044
```

> **Advanced:** Pass a free-text string as `criteria` for domain-specific checks
> not covered by `CRITERIA_BANK` — for example, PII detection or domain-specific
> compliance rules. See [Safety Guardrails](../how-to/safety-guardrails#custom-criteria).

---

## Step 6: Groundedness checks with retrieved context

When your agent retrieves documents before answering, use `guardian_check()` with
`criteria="groundedness"` to detect hallucinations relative to what was retrieved.
Include the source document in the evaluation context as a user message:

```python
import mellea
from mellea.backends import ModelOption, tool
from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.components import Message
from mellea.stdlib.components.intrinsic import guardian
from mellea.stdlib.context import ChatContext

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
)

output_text = str(response)

# Build an evaluation context with the source document prepended.
# guardian_check compares the assistant response against the provided document.
eval_ctx = (
    ChatContext()
    .add(Message("user", f"Document: {RETRIEVED_CONTEXT}"))
    .add(Message("assistant", output_text))
)

guardian_backend = LocalHFBackend(model_id="ibm-granite/granite-4.0-micro")
score = guardian.guardian_check(eval_ctx, guardian_backend, criteria="groundedness")

if score < 0.5:
    print(f"Grounded response (score: {score:.4f}):", output_text)
else:
    print(f"Groundedness risk detected (score: {score:.4f}) — response may be hallucinated.")
# Example output:
# Grounded response (score: 0.0034): Mellea is an open-source Python framework ...
```

> **Tip:** Pass the same text you supplied as `grounding_context` to
> `"Document:"` in the evaluation context. This ensures the guardian evaluates
> the response against exactly what the agent was given.

---

## Step 7: A ReACT agent with Guardian Intrinsics

For goal-driven agentic loops, combine `react()` with Guardian Intrinsics. The
`react()` function is an async built-in that runs the Reason-Act loop until the
goal is reached or the step budget is exhausted:

```python
import asyncio
import mellea
from mellea.backends import tool
from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.components import Message
from mellea.stdlib.components.intrinsic import guardian
from mellea.stdlib.context import ChatContext
from mellea.stdlib.frameworks.react import react

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

# Evaluate the agent's final output against harm criteria.
guardian_backend = LocalHFBackend(model_id="ibm-granite/granite-4.0-micro")
eval_ctx = ChatContext().add(Message("assistant", output))
harm_score = guardian.guardian_check(eval_ctx, guardian_backend, criteria="harm")

if harm_score < 0.5:
    print(f"Agent output (safe, score: {harm_score:.4f}):", output)
else:
    print(f"Agent output flagged — harm score: {harm_score:.4f}")
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
| `guardian_check()` | Post-generation safety risk score (0.0–1.0) |
| Reused `guardian_backend` | Single model load amortised across all criteria checks |
| `criteria="groundedness"` + document context | Detect hallucination relative to retrieved context |
| `react()` | Goal-driven multi-step agentic loop |

---

**See also:** [The Requirements System](../concepts/requirements-system) | [Safety Guardrails](../how-to/safety-guardrails) | [Tools and Agents](../how-to/tools-and-agents)
