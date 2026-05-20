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
- Detecting harmful outputs with `guardian.guardian_check`
- Grounding safety checks against retrieved context

**Prerequisites:** [Tutorial 02](./02-streaming-and-async) and
[Tutorial 03](./03-using-generative-stubs) complete,
`pip install mellea`, Ollama running locally with `granite4.1:3b` downloaded.

---

## Step 1: A simple tool-using agent

Start with two tools — a search stub and a calculator — and wire them into an
`instruct()` call:

```python
# Requires: mellea
# Returns: str
import mellea
from mellea.backends import ModelOption, tool
from mellea.stdlib.components import ToolMessage
from mellea.stdlib.context import ChatContext

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

m = mellea.start_session(ctx=ChatContext())

result = m.instruct(
    "What is Mellea, and how many characters are in the word 'Mellea'?",
    model_options={ModelOption.TOOLS: [web_search, calculate]},
    tool_calls=True,
)

if result.tool_calls:
    for name, tc in result.tool_calls.items():
        output = tc.call_func()
        m.ctx = m.ctx.add(ToolMessage("tool", str(output), output, name, tc.args, tc))
    print(str(m.instruct("Answer the original question using the tool results.", tool_calls=False)))
else:
    print(str(result))
```

```text Sample output
Mellea is a Python framework designed for creating and managing
generative programs. It provides tools and libraries that allow
developers to build applications which can generate content or
behaviors over time in an algorithmic manner. The framework likely
includes features such as event handling, state management, and
scheduling capabilities, making it easier to work with complex
procedural logic.

The word 'Mellea' consists of 6 characters.
```

> **Note:** LLM output is non-deterministic. The model may call one or both tools; the answer will reference whichever tool results it received.

`ChatContext` is required here so tool result messages persist between the first
`instruct` call and the follow-up. Without it, the session uses `SimpleContext`,
which discards history between calls and the model answers as if no tools ran.

---

## Step 2: Adding output requirements

Require the agent to format its answer as a short structured response:

```python
# Requires: mellea
# Returns: str
import mellea
from mellea.backends import ModelOption, tool
from mellea.stdlib.components import ToolMessage
from mellea.stdlib.context import ChatContext
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

word_count_req = req(
    "The response must be 50 words or fewer.",
    validation_fn=simple_validate(
        lambda x: (
            len(x.split()) <= 50,
            f"Response is {len(x.split())} words; must be 50 or fewer.",
        )
    ),
)

m = mellea.start_session(ctx=ChatContext())

result = m.instruct(
    "What is Mellea, and how many characters are in the word 'Mellea'?",
    model_options={ModelOption.TOOLS: [web_search, calculate]},
    tool_calls=True,
)

if result.tool_calls:
    for name, tc in result.tool_calls.items():
        output = tc.call_func()
        m.ctx = m.ctx.add(ToolMessage("tool", str(output), output, name, tc.args, tc))
    print(str(m.instruct(
        "Answer the original question using the tool results.",
        requirements=[
            req("The response must answer both questions."),
            word_count_req,
        ],
        tool_calls=False,
    )))
else:
    print(str(result))
```

```text Sample output
Mellea is a Python framework designed for creating and managing
generative programs, facilitating the development of complex
algorithms and simulations with ease. The word 'Mellea' consists
of 6 characters.
```

> **Note:** With the 50-word requirement enforced, the response will be concise and answer both questions. Wording varies by model and temperature.

Requirements are placed on the synthesis call (the follow-up `instruct`) rather
than the tool-call step — at the tool-call step the model returns structured
tool invocations, not the final answer, so applying requirements there would
validate the wrong output. The word-count check runs deterministically. The
"answer both questions" requirement falls back to LLM-as-a-judge. If either
fails, Mellea retries with the failure reason embedded in the repair request.

---

## Step 3: Inspecting failures and handling a retry budget

Use `RejectionSamplingStrategy` with `return_sampling_results=True` to observe
what happens when requirements fail:

```python
# Requires: mellea
# Returns: None
import mellea
from mellea.backends import ModelOption, tool
from mellea.stdlib.components import ToolMessage
from mellea.stdlib.context import ChatContext
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

answer_requirements = [
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
]

m = mellea.start_session(ctx=ChatContext())

# Step 1: run the tool-call step to collect tool results into the context.
tool_result = m.instruct(
    "What is Mellea, and how many characters are in the word 'Mellea'?",
    model_options={ModelOption.TOOLS: [web_search, calculate]},
    tool_calls=True,
)

if tool_result.tool_calls:
    for name, tc in tool_result.tool_calls.items():
        output = tc.call_func()
        m.ctx = m.ctx.add(ToolMessage("tool", str(output), output, name, tc.args, tc))

# Step 2: synthesize with requirements and a retry budget, capturing all attempts.
sampling = m.instruct(
    "Answer the original question using the tool results.",
    requirements=answer_requirements,
    strategy=RejectionSamplingStrategy(loop_budget=3),
    return_sampling_results=True,
    tool_calls=False,
)

if sampling.success:
    print("Passed:", str(sampling.result))
else:
    print(f"All {len(sampling.sample_generations)} attempts failed.")
    for i, attempt in enumerate(sampling.sample_generations):
        print(f"  Attempt {i + 1}: {str(attempt.value)[:80]}...")
```

On a successful run (the most common case):

```text Sample output
Passed: Mellea is a Python framework designed for creating and managing
generative programs, facilitating complex program generation tasks
through its modular architecture. The word "Mellea" consists of 6
characters.
```

If all attempts exceed the word limit:

```text Sample output
All 3 attempts failed.
  Attempt 1: Mellea is an open-source Python framework designed to help
             developers build generative...
  Attempt 2: Based on the web search, Mellea is a Python framework for
             generative programs. The wor...
  Attempt 3: Mellea is a Python framework. According to a web search,
             it helps build generative pro...
```

`sampling.success` is `True` when at least one attempt satisfied all requirements.
`sampling.sample_generations` gives you every attempt in order — useful for
debugging or for choosing the best available output when the budget runs out.

---

## Step 4: Adding Guardian harm detection

Guardian intrinsics evaluate the output against specific risk criteria. Run them
after your agent responds to flag outputs before they reach downstream code.

```python
# Requires: mellea
# Returns: None
import mellea
from mellea.backends import ModelOption, tool
from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.components import Message, ToolMessage
from mellea.stdlib.components.intrinsic import guardian
from mellea.stdlib.context import ChatContext
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

m = mellea.start_session(ctx=ChatContext())

question = "What is Mellea, and how many characters are in the word 'Mellea'?"

tool_result = m.instruct(
    question,
    model_options={ModelOption.TOOLS: [web_search, calculate]},
    tool_calls=True,
)

if tool_result.tool_calls:
    for name, tc in tool_result.tool_calls.items():
        output = tc.call_func()
        m.ctx = m.ctx.add(ToolMessage("tool", str(output), output, name, tc.args, tc))

response = m.instruct(
    "Answer the original question using the tool results.",
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
    tool_calls=False,
)

output_text = str(response)

# Guardian intrinsics require a LocalHFBackend — they load LoRA adapters
# that are not supported by OllamaModelBackend.
guardian_backend = LocalHFBackend(model_id="ibm-granite/granite-4.0-micro")

# Build a context containing the exchange to check.
check_ctx = (
    ChatContext()
    .add(Message("user", question))
    .add(Message("assistant", output_text))
)

harm_score = guardian.guardian_check(check_ctx, guardian_backend, criteria="harm")
jailbreak_score = guardian.guardian_check(check_ctx, guardian_backend, criteria="jailbreak")

# Scores >= 0.5 indicate a detected risk.
safe = harm_score < 0.5 and jailbreak_score < 0.5
if safe:
    print("Output passed safety checks:", output_text)
else:
    if harm_score >= 0.5:
        print(f"Harm risk detected (score: {harm_score:.2f})")
    if jailbreak_score >= 0.5:
        print(f"Jailbreak risk detected (score: {jailbreak_score:.2f})")
```

```text Sample output
Output passed safety checks: Mellea is a Python framework designed
for creating generative programs, enabling users to build complex
and dynamic applications with ease. The word "Mellea" consists of
6 characters.
```

> **Note:** Guardian intrinsics load LoRA adapters and require `LocalHFBackend`.
> They cannot run against `OllamaModelBackend`. The main agent and the Guardian
> checks can use different backends — only the Guardian calls need `LocalHFBackend`.

Scores are floats between 0.0 (safe) and 1.0 (risk detected); 0.5 is the
threshold. The available criteria are: `"harm"`, `"jailbreak"`, `"social_bias"`,
`"profanity"`, `"violence"`, `"sexual_content"`, `"unethical_behavior"`, `"groundedness"`,
`"answer_relevance"`, `"context_relevance"`, and `"function_call"`.

---

## Step 5: Running multiple Guardian checks with a shared backend

When running several Guardian criteria checks, instantiate `LocalHFBackend` once
and pass it to each call to avoid reloading the model weights for every check:

```python
# Requires: mellea
# Returns: None
import mellea
from mellea.backends import ModelOption, tool
from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.components import Message, ToolMessage
from mellea.stdlib.components.intrinsic import guardian
from mellea.stdlib.context import ChatContext

@tool
def web_search(query: str) -> str:
    """Search the web for information about a topic.

    Args:
        query: The search query.
    """
    return f"Top result for '{query}': Mellea is a Python framework for generative programs."

m = mellea.start_session(ctx=ChatContext())

question = "What is Mellea?"

result = m.instruct(
    question,
    model_options={ModelOption.TOOLS: [web_search]},
    tool_calls=True,
)

if result.tool_calls:
    for name, tc in result.tool_calls.items():
        output = tc.call_func()
        m.ctx = m.ctx.add(ToolMessage("tool", str(output), output, name, tc.args, tc))
    response = m.instruct("Answer the original question using the tool results.", tool_calls=False)
else:
    response = result

output_text = str(response)

# Load once, reuse across all criteria checks.
guardian_backend = LocalHFBackend(model_id="ibm-granite/granite-4.0-micro")

check_ctx = (
    ChatContext()
    .add(Message("user", question))
    .add(Message("assistant", output_text))
)

criteria = ["harm", "profanity", "answer_relevance", "jailbreak"]
for criterion in criteria:
    score = guardian.guardian_check(check_ctx, guardian_backend, criteria=criterion)
    status = "FAIL" if score >= 0.5 else "PASS"
    print(f"[{status}] {criterion}: {score:.2f}")
```

```text Sample output
[PASS] harm: 0.00
[PASS] profanity: 0.00
[PASS] answer_relevance: 0.07
[PASS] jailbreak: 0.00
```

> **Note:** Guardian scores are deterministic for a given response string, but the
> agent's answer varies by model and temperature, so exact scores will differ across
> runs.

The available criteria are: `"harm"`, `"jailbreak"`, `"social_bias"`,
`"profanity"`, `"violence"`, `"sexual_content"`, `"unethical_behavior"`, `"groundedness"`,
`"answer_relevance"`, `"context_relevance"`, and `"function_call"`.

---

## Step 6: Groundedness checks with retrieved context

When your agent retrieves documents before answering, use
`rag.flag_hallucinated_content` to confirm the response is faithful to what was
retrieved rather than hallucinated:

```python
# Requires: mellea
# Returns: None
import mellea
from mellea.backends import ModelOption, tool
from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.components import Document, Message, ToolMessage
from mellea.stdlib.components.intrinsic import rag
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

m = mellea.start_session(ctx=ChatContext())

question = "Using the retrieved documentation, describe what Mellea is."

result = m.instruct(
    question,
    model_options={ModelOption.TOOLS: [retrieve_docs]},
    tool_calls=True,
)

if result.tool_calls:
    for name, tc in result.tool_calls.items():
        output = tc.call_func()
        m.ctx = m.ctx.add(ToolMessage("tool", str(output), output, name, tc.args, tc))
    response = m.instruct(
        "Answer the original question using the tool results.",
        grounding_context={"docs": RETRIEVED_CONTEXT},
        tool_calls=False,
    )
else:
    response = result

output_text = str(response)

# Check the response is faithful to the retrieved document.
guardian_backend = LocalHFBackend(model_id="ibm-granite/granite-4.0-micro")
doc = Document(text=RETRIEVED_CONTEXT, title="Mellea docs")
check_ctx = ChatContext().add(Message("user", question))
hallucination_result = rag.flag_hallucinated_content(output_text, [doc], check_ctx, guardian_backend)

flagged = [s for s in hallucination_result if s.get("faithfulness") == "No"]
if not flagged:
    print("Grounded response:", output_text)
else:
    print("Response may contain hallucinated content.")
    for sentence in flagged:
        print(f"  Flagged: {sentence['response_text']}")
```

```text Sample output
Grounded response: Mellea is an open-source Python framework designed
for building generative programs. It offers three primary
functionalities: instruct(), @generative, and @mify as its core
primitives. The unique aspect of Mellea lies in its backend-agnostic
nature, meaning it can support various platforms such as Ollama,
OpenAI, and also accommodate custom backends. This versatility makes
it a powerful tool for developers looking to create flexible and
adaptable generative applications.
```

> **Tip:** Pass the same text as both `grounding_context` to `instruct()` and as
> the `Document` to `flag_hallucinated_content`. This ensures the faithfulness
> check evaluates the response against exactly what the agent was given.

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
    return f"Search result for '{query}': Mellea is a Python framework for generative programs."

m = mellea.start_session()

goal = "Use web_search to find out what Mellea is and report the definition."

async def run_agent() -> str:
    result, _ = await react(
        goal=goal,
        context=ChatContext(),
        backend=m.backend,
        tools=[web_search],
    )
    return str(result)

output = asyncio.run(run_agent())

# Validate the agent's final output.
guardian_backend = LocalHFBackend(model_id="ibm-granite/granite-4.0-micro")
check_ctx = (
    ChatContext()
    .add(Message("user", goal))
    .add(Message("assistant", output))
)
harm_score = guardian.guardian_check(check_ctx, guardian_backend, criteria="harm")

if harm_score < 0.5:
    print("Agent output (safe):", output)
else:
    print(f"Agent output flagged (harm score: {harm_score:.2f})")
```

```text Sample output
Agent output (safe): Mellea appears to be a Python framework
specifically designed for creating generative programs. It provides
tools and utilities that allow developers to build software systems
where behavior or outcomes are determined by random or probabilistic
processes, rather than fixed rules.
```

> **Note:** LLM output is non-deterministic. The `react()` loop runs sequentially —
> each turn awaits the model before proceeding, so `ChatContext` is safe to use here
> despite the async execution. The agent calls `web_search` to ground its answer, then
> calls the built-in `final_answer` tool to signal completion. The Guardian harm check
> runs after the loop using `LocalHFBackend` with the `guardian-core` LoRA adapter.
> Your exact wording will vary.

---

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
| `requirements` + `simple_validate` on synthesis call | Deterministic and LLM-judged output constraints |
| `RejectionSamplingStrategy` | Explicit retry budget |
| `return_sampling_results=True` | Inspect every attempt for debugging |
| `guardian.guardian_check` | Post-generation safety risk detection |
| Shared `LocalHFBackend` | Amortise model loading across multiple checks |
| `rag.flag_hallucinated_content` | Detect hallucination relative to retrieved context |
| `react()` | Goal-driven multi-step agentic loop |

---

**See also:** [The Requirements System](../concepts/requirements-system) | [Security and Taint Tracking](../advanced/security-and-taint-tracking) | [Tools and Agents](../how-to/tools-and-agents)
