---
title: "Mellea vs Orchestration Frameworks"
description: "What makes Mellea different from LangChain, smolagents, and other agent frameworks — and how they work together."
# diataxis: explanation
---

# Mellea vs Orchestration Frameworks

Mellea is not an orchestration framework. This distinction shapes how you design
systems with it.

**Orchestration frameworks** — LangChain, smolagents, CrewAI, LlamaIndex — decide
*what* to call and *when*. They provide planning loops, routing logic, graph
execution, agent memory, and multi-agent coordination. Their job is the horizontal
structure of a program: which step runs next, which tool gets selected, how subtasks
are divided among agents.

**Mellea** decides *how well* a single call or tightly coupled group of calls
performs. It is the vertical reliability layer: given that you are calling an LLM,
Mellea ensures the output meets your requirements before it is returned to the caller.
Its job is the local execution quality of each node in the graph, not the graph itself.

The two are complementary. An orchestrator that delegates to Mellea-instrumented
functions gains reliability guarantees at each step without changing the orchestration
logic.

## What each layer handles

| Concern | Orchestration framework | Mellea |
| ------- | ----------------------- | ------ |
| Which tool to call next | ✓ | — |
| Multi-agent routing | ✓ | — |
| Workflow graphs | ✓ | — |
| Output meets requirements | — | ✓ |
| Instruct–validate–repair | — | ✓ |
| Structured type enforcement | — | ✓ |
| Per-call sampling strategy | — | ✓ |
| Context window management | — | ✓ |

This is not a comprehensive feature comparison — both ecosystems are large. The point
is the different level of abstraction: orchestrators operate at the program level,
Mellea at the call level.

## Using Mellea inside an orchestrator

A `@generative` function or an `instruct()` call is just a Python function. Any
framework that calls Python functions can use Mellea as a tool.

### smolagents

```python
from mellea import generative, start_session
from mellea.stdlib.requirements import req, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy
from mellea.backends.tools import MelleaTool


@generative
def summarize(text: str, max_words: int) -> str:
    """Summarize the text in at most max_words words."""


# Wrap the Mellea function as a smolagents tool
# (the decorator gives it a docstring and type signature smolagents can read)
from smolagents import tool as smolagents_tool

@smolagents_tool
def reliable_summarize(text: str, max_words: int = 50) -> str:
    """Summarize text with guaranteed word limit, using Mellea.

    Args:
        text: The text to summarize.
        max_words: Maximum number of words in the summary.
    """
    m = start_session()
    result = summarize(
        m,
        text=text,
        max_words=max_words,
        requirements=[
            req(
                f"Fewer than {max_words} words.",
                validation_fn=simple_validate(
                    lambda x: (len(x.split()) <= max_words,
                               f"Summary has {len(x.split())} words; limit is {max_words}.")
                ),
            )
        ],
        strategy=RejectionSamplingStrategy(loop_budget=3),
    )
    return str(result)
```

The smolagents agent calls `reliable_summarize` as a tool. From its perspective, it
is an opaque Python function. Inside, Mellea ensures the word-count requirement is
enforced before the result is returned.

### LangChain

```python
from langchain.tools import StructuredTool
from mellea import start_session
from mellea.stdlib.requirements import req, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy


def extract_entities(text: str) -> str:
    """Extract named entities from text, returning comma-separated names."""
    m = start_session()
    result = m.instruct(
        "Extract all named entities (people, organisations, places) from: {{text}}",
        requirements=[
            "List entities as a comma-separated string with no extra text.",
            req("Include only entities that appear explicitly in the text.",
                validation_fn=simple_validate(lambda x: "," in x or len(x.split()) <= 5)),
        ],
        strategy=RejectionSamplingStrategy(loop_budget=3),
        user_variables={"text": text},
    )
    return str(result)


entity_tool = StructuredTool.from_function(
    func=extract_entities,
    name="entity_extractor",
    description="Extract named entities from text.",
)
```

The LangChain agent can include `entity_tool` in its toolbox without knowing Mellea
is involved.

## Building agents with Mellea

Mellea also supports building agentic programs directly, without an external
orchestrator:

- **ReACT loops** — implement thought/action/observation cycles using `m.chat()`
  with `ChatContext` and the `@tool` decorator. See
  [Tools and Agents](../guide/tools-and-agents.md).
- **Guarded agents** — combine the ReACT pattern with `requirements` and
  `GuardianCheck` to enforce safety constraints at every step. See
  [Security and Taint Tracking](../advanced/security-and-taint-tracking.md).
- **Structured outputs** — use `@generative` with Pydantic models or `Literal` types
  to enforce type-safe structured output at each step. See
  [Generative Functions](../guide/generative-functions.md).

For programs where the control flow is fixed in Python — a pipeline, an extraction
workflow, a classification step — there is no need for a separate orchestrator.
Use one when you need the model itself to decide what to do next; skip it when you
already know the structure.

## Adoption paths

### Greenfield

Build directly with Mellea from the start:

```python
import mellea

m = mellea.start_session()
result = m.instruct("Analyse customer feedback.", requirements=["..."])
```

This is the simplest path. You get full control over the prompts, requirements, and
sampling strategies.

### Leaf-node injection

Add Mellea to an existing system by wrapping individual calls:

```python
# Before: raw LLM call in an existing pipeline
def classify(text: str) -> str:
    return llm.call(f"Classify: {text}")

# After: drop-in Mellea replacement with reliability
from mellea import generative, start_session
from typing import Literal

@generative
def classify(text: str) -> Literal["positive", "negative", "neutral"]:
    """Classify the sentiment of the text."""

def classify_wrapper(text: str) -> str:
    m = start_session()
    return str(classify(m, text=text))
```

The surrounding system does not change. Only the leaf node — the LLM call —
is instrumented with Mellea. This is often the fastest path to reliability gains in
an existing codebase.

### Tool enrichment

Add Mellea to an existing orchestrator by replacing unreliable tool implementations:

Replace a tool function that directly calls an LLM with a Mellea-instrumented version
that validates its output before returning. The orchestrator's routing logic is
unchanged; the tool just becomes more reliable.

## When you need an orchestrator

Mellea does not provide:

- Agent planning and reasoning about which tool to use next
- Multi-agent coordination (spawning sub-agents, passing results between agents)
- Long-running workflow state across sessions
- Automatic tool selection from a registry

If your program needs any of these, pair Mellea with an orchestration framework.
Build your Mellea instrumented functions, then wire them into the orchestrator as
tools or steps.

---

**Previous:** [The Requirements System](./requirements-system.md) |
**Next:** [Context and Sessions](./context-and-sessions.md)

**See also:** [Tools and Agents](../guide/tools-and-agents.md) |
[Security and Taint Tracking](../advanced/security-and-taint-tracking.md)
