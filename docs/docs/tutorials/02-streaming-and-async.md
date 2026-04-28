---
title: "Tutorial: Streaming and Async"
description: "Make LLM calls non-blocking, stream tokens as they arrive, and process batches concurrently."
# diataxis: tutorial
---

In this tutorial you take the feedback analysis pipeline from Tutorial 01 and
make it production-ready: non-blocking async calls, token-by-token streaming to
a UI, and concurrent batch processing.

By the end you will have covered:

- `ainstruct()` and the async session method naming convention
- `ModelOption.STREAM` and `mot.astream()` for incremental output
- `wait_for_all_mots` for fan-out concurrent generation
- Context behaviour with concurrent async calls

**Prerequisites:** [Tutorial 01](./01-your-first-generative-program) complete,
`pip install mellea`, Ollama running locally with `granite4:micro` downloaded.

---

## Step 1: Your first async call

Every sync method on `MelleaSession` has an `a`-prefixed async counterpart with
the same signature and return type. Replace `instruct()` with `ainstruct()` and
wrap the call in `async def`:

```python
# Requires: mellea
# Returns: str
import asyncio
import mellea

async def main():
    m = mellea.start_session()
    result = await m.ainstruct(
        "Summarise this customer feedback in one sentence: "
        "The onboarding was confusing and took far too long. "
        "Support was helpful once I got through."
    )
    print(str(result))

asyncio.run(main())
```

```text
The customer experienced confusion during the initial setup process which was time-consuming, but support was beneficial once connected.
```

> **Note:** LLM output is non-deterministic. Your result will vary in wording, but should be a single sentence.

`ainstruct()` returns a [`ModelOutputThunk`](../reference/glossary#modeloutputthunk). `await`-ing it starts generation
immediately; `str(result)` resolves the value when it is ready. Every other
method follows the same pattern: `achat()`, `aact()`, `aquery()`,
`atransform()`, `avalidate()`.

---

## Step 2: Streaming tokens

Enable streaming by passing `ModelOption.STREAM: True` in `model_options`.
Consume chunks with `mot.astream()` as they arrive — useful for displaying
output progressively rather than waiting for the full response:

```python
# Requires: mellea
# Returns: str
import asyncio
import mellea
from mellea.backends import ModelOption

async def stream_summary(feedback: str) -> str:
    m = mellea.start_session()
    mot = await m.ainstruct(
        "Summarise this customer feedback in one sentence: {{text}}",
        user_variables={"text": feedback},
        model_options={ModelOption.STREAM: True},
        strategy=None,
    )

    chunks = []
    while not mot.is_computed():
        chunk = await mot.astream()
        print(chunk, end="", flush=True)
        chunks.append(chunk)
    print()  # newline after streaming completes

    return "".join(chunks)

asyncio.run(stream_summary(
    "The onboarding was confusing and took far too long. "
    "Support was helpful once I got through."
))
```

```text
The customer found the onboarding process confusing and lengthy but received good support assistance afterwards.
```

> **Note:** With streaming enabled, tokens print incrementally as they arrive. The example above shows the completed output. Your terminal will display each token as it is generated. LLM output is non-deterministic. Your result will vary in wording, but should be a single sentence.

How `astream()` works:

- Each call returns only the **new content** since the previous call.
- When generation is complete, `is_computed()` returns `True` and the final
  `astream()` call returns the remaining content.
- `strategy=None` is required to get a lazy thunk — without it, `ainstruct()` returns
  a pre-computed thunk and `is_computed()` is already `True` before the loop runs.
- Do not call `astream()` from multiple coroutines on the same thunk simultaneously.

---

## Step 3: Concurrent batch processing

The pipeline from Tutorial 01 processes one feedback item at a time, and each
call blocks until the previous one completes. With `ainstruct()` you can fire
all calls immediately and resolve them together.

Use `wait_for_all_mots` to await a list of thunks concurrently:

```python
# Requires: mellea
# Returns: list[str]
import asyncio
import mellea
from mellea.helpers.async_helpers import wait_for_all_mots

FEEDBACK_BATCH = [
    "The onboarding was confusing and took far too long. Support was helpful once I got through.",
    "Product works great but the mobile app crashes frequently. No response from support.",
    "Fast delivery, exactly as described. Will order again.",
    "Billing charged me twice. Still waiting for a refund after two weeks.",
]

async def summarise_batch(items: list[str]) -> list[str]:
    m = mellea.start_session()

    # Fire all summarisation calls immediately — none waits for the others.
    thunks = []
    for item in items:
        thunk = await m.ainstruct(
            "Summarise this customer feedback in one sentence: {{text}}",
            user_variables={"text": item},
        )
        thunks.append(thunk)

    # None are resolved yet — all are generating in parallel.
    await wait_for_all_mots(thunks)

    # All thunks are now resolved.
    return [t.value for t in thunks]

summaries = asyncio.run(summarise_batch(FEEDBACK_BATCH))
for summary in summaries:
    print(summary)
```

```text
The initial user experience was challenging with lengthy onboarding process but the support team proved to be very helpful post-onboarding.
The product functions well; however, there are frequent issues with the mobile app crashing, and user inquiries to the support team have not been addressed.
The customer received fast, high-quality deliveries that matched their expectations, indicating satisfaction with the service and willingness to reorder.
The customer has been billed twice and is still awaiting a refund despite the two-week duration.
```

> **Note:** LLM output is non-deterministic. You will see four summary lines, one per input, in the same order as `FEEDBACK_BATCH`.

The four requests are in flight simultaneously. Total wall-clock time is
roughly the latency of the slowest single call, rather than the sum of all four.

---

## Step 4: Mixing parallel and sequential steps

Some pipeline steps are independent; others depend on earlier results. You can
resolve dependencies explicitly without blocking unrelated work.

In the Tutorial 01 pipeline, `extract_issues` is independent of `summarize` —
both take the raw feedback. Run them in parallel, then feed the resolved summary
into `classify_sentiment`:

```python
# Requires: mellea
# Returns: None
import asyncio
from typing import Literal

import mellea
from mellea import generative
from mellea.helpers.async_helpers import wait_for_all_mots


@generative
def classify_sentiment(summary: str) -> Literal["positive", "negative", "mixed"]:
    """Classify the overall sentiment of the customer feedback summary."""


async def analyze_feedback(feedback: str) -> None:
    m = mellea.start_session()

    # Fire summarise and extract_issues in parallel — both take raw feedback.
    summary_thunk = await m.ainstruct(
        "Summarise this customer feedback in one sentence: {{text}}",
        user_variables={"text": feedback},
    )
    issues_thunk = await m.ainstruct(
        "Extract JSON with main_complaint, positive_aspect, and urgency from: {{text}}",
        user_variables={"text": feedback},
    )

    await wait_for_all_mots([summary_thunk, issues_thunk])

    summary = summary_thunk.value

    # classify_sentiment depends on the resolved summary — run it after.
    sentiment = classify_sentiment(m, summary=summary)

    print(f"Summary:   {summary}")
    print(f"Sentiment: {str(sentiment)}")
    print(f"Issues:    {issues_thunk.value}")


asyncio.run(analyze_feedback(
    "The onboarding was confusing and took far too long. "
    "Support was helpful once I got through."
))
```

```text
Summary:   The customer experienced confusion during the onboarding process which was lengthy, but support was beneficial upon successful navigation.
Sentiment: mixed
Issues:    {
  "main_complaint": "The onboarding process was confusing and time-consuming",
  "positive_aspect": "Support provided assistance once the user was able to contact them",
  "urgency": "Medium"
}
```

> **Note:** LLM output is non-deterministic, your output may vary based on model and temperature.

---

## Step 5: Context and concurrency

By default [`start_session()`](../reference/glossary#melleasession) uses [`SimpleContext`](../reference/glossary#context), which is safe for concurrent
async calls. If you switch to [`ChatContext`](../reference/glossary#context), Mellea logs a warning because
concurrent writes can corrupt the context state:

```text
WARNING: Not using a SimpleContext with asynchronous requests could cause
unexpected results due to stale contexts. Ensure you await between requests.
```

> **Note:** This warning appears whenever `ChatContext` is used with async methods,
> even if you `await` each call sequentially. It is safe to ignore when you ensure
> each call is fully resolved before starting the next.

If you need `ChatContext` (for multi-turn conversation), await each call before
starting the next:

```python
# Requires: mellea
# Returns: None
import asyncio
import mellea
from mellea.stdlib.context import ChatContext

async def sequential_chat():
    m = mellea.start_session(ctx=ChatContext())
    r1 = await m.achat("Hello.")
    r2 = await m.achat("Tell me more.")  # safe — r1 is fully resolved
    print(str(r2))

asyncio.run(sequential_chat())
```

```text
Of course! If you have any specific topics or questions in mind, feel free to share them, and I'll do my best to provide detailed information on the subject. You might be interested in a variety of areas such as technology advancements, health tips, travel destinations, historical events, scientific discoveries, or even everyday life advice. Just let me know what you're curious about!
```

> **Note:** LLM output is non-deterministic. Your result will vary in wording.

For parallel generation, keep the default `SimpleContext`.

---

## What you built

| Pattern | What it gives you |
| --- | --- |
| `ainstruct()` / `achat()` / `aact()` | Non-blocking LLM calls |
| `ModelOption.STREAM` + `astream()` | Token-by-token output for responsive UIs |
| `wait_for_all_mots` | Fan-out: all thunks resolve concurrently |
| Explicit dependency ordering | Sequential where needed, parallel everywhere else |
| `SimpleContext` (default) | Safe concurrent access with no state corruption |

---

**See also:** [Async and Streaming](../how-to/use-async-and-streaming) (full API reference) |
[Tutorial 03: Using Generative Stubs](./03-using-generative-stubs)
