---
title: "LLMON (LLM Object Notation)"
description: "How Mellea talks to an LLMON-trained model: LLMONFormatter sends each part of a component in the message fields the model's chat template turns into LLMON."
# diataxis: explanation
sidebar_label: "LLMON"
---

LLMON (LLM Object Notation, [arXiv:2603.22519](https://arxiv.org/abs/2603.22519))
is an LLM-native markup for conveying the *structure* and *semantic metadata* of
text to a language model — for example, distinguishing instructions from data to
reduce prompt-injection risk. Mellea can talk to a model that has been trained on
LLMON without the developer hand-building any markup: `LLMONFormatter` sends each
part of a Mellea component in the message fields the model's chat template turns
into LLMON.

## Where the LLMONizing happens

The granite LLMON preview models (`ibm-granite/granite-4.1-3b-llmon-preview` and
`ibm-granite/granite-4.1-8b-llmon-preview`) LLMONize in their **chat template**. It
writes the blocks, their ids, and the `exec` block that tells the model what to act
on, all from the fields of each chat message. `LLMONFormatter` fills those fields
and writes no LLMON itself.

The fields reach the template intact only on `LocalHFBackend`, which hands it each
message as Mellea built it, so LLMON is supported on that backend alone.
OpenAI-compatible servers such as vLLM, and Ollama, rebuild every message from its
standard keys before applying the template, which drops the fields. So every other
backend raises `ValueError` when it is given an `LLMONFormatter` or a registered
LLMON model, rather than silently sending prompts without their LLMON structure.

## Using `LLMONFormatter`

Pass the formatter to the backend:

```python
from mellea.backends.huggingface import LocalHFBackend
from mellea.formatters import LLMONFormatter
from mellea.stdlib.context import ChatContext
from mellea.stdlib.sampling import MultiTurnStrategy
from mellea.stdlib.session import MelleaSession

model_id = "ibm-granite/granite-4.1-3b-llmon-preview"
m = MelleaSession(
    LocalHFBackend(model_id=model_id, formatter=LLMONFormatter(model_id)),
    ctx=ChatContext(),
)
result = m.instruct(
    "What is the warehouse code for Lyon?",
    grounding_context={"codes": "The Lyon warehouse code is WH-7Q4Z-LY."},
    requirements=["Answer with the code only."],
    strategy=MultiTurnStrategy(loop_budget=3),  # See "Requirement validation".
)
print(result.value)  # Plain text, never LLMON markup.
# Output will vary — LLM responses depend on model and temperature.
```

`LocalHFBackend` chooses `LLMONFormatter` automatically for a registered LLMON
model, which both granite LLMON preview models are, so `formatter=` above can be
left out. A model that is neither registered nor given an `LLMONFormatter` is
unaffected: it gets the usual `TemplateFormatter` prose.

## What the model receives

`LLMONFormatter` builds each message from the same `format_for_llm()` args that
`TemplateFormatter` renders through Jinja2 templates, putting each part in the
field the chat template expects for it. The template turns a user message's content
and `instruction` items into `instr` blocks, its `data` items into `data` blocks, its
`requirement` items into `req` blocks, and its `documents` into `doc` blocks, and it
wires every item into the turn's `exec` block (each is sent with `"exec": True`).

- **Instruction:** the prefix and description, joined, are the content, which the
  template takes as the instruction. A repair (the feedback `RepairTemplateStrategy`
  adds after a failed check) is a further `instruction` item, each requirement a
  `requirement` item, and each in-context example and grounding-context value a
  `data` item. A `Document` in the grounding context is a `data` item too, sent as
  its text alone, without its title. Each grounding-context value is labeled with
  its key, as `key: value`, so the model can tell the values apart. An
  instruction with an `output_prefix` raises `ValueError`, since the template cannot
  start the reply with one.
- **Message:** its content, plus a user message's documents as `documents` items.
- **Content block:** its text, as the instruction.
- **Requirement check:** the LLM-as-a-judge turn that requirement validation sends
  is an instruction to judge, worded as in Mellea's `Requirement` template, with
  `Output: …` and `Requirement: …` as `data` items. In a `requirement` item, the
  requirement would constrain the judge's own reply instead.

Any other component raises `TypeError`, and so does a component inside an
instruction's args, other than a `Document` or a model output, which is sent as its
text.

The fields travel in each message's `provider_fields`, declared for the HuggingFace
backend. `print` returns a turn's text alone, so it raises for a turn with fields:
raw generation, which sends one string per action, cannot carry them.

System, assistant, and tool turns are sent as their plain text, because the
model's chat template wraps each of them in its own block. A model output in the
history is sent as the model's reply text. An assistant's tool calls travel as the
message's `tool_calls`, which the chat template writes out and wires into the next
`exec` block; calls the model wrote in its reply text, as the HuggingFace backend
leaves them, are moved there once the backend has parsed every one. A tool result
is sent as its text for the template to mark as a tool response.

For example, this instruction:

```python
from mellea.stdlib.components import Instruction

Instruction(
    "Summarise the review in one sentence.",
    requirements=["Mention the shipping."],
    grounding_context={"review": "Loved it, but shipping was slow."},
)
```

becomes a user message whose content is `Summarise the review in one sentence.`,
with these fields:

```python
{
    "data": [{"value": "review: Loved it, but shipping was slow.", "exec": True}],
    "requirement": [{"value": "Mention the shipping.", "exec": True}],
}
```

The granite LLMON preview models' chat template renders that message as follows
(the template draws each `ID` at random; newlines added for readability):

```text
<|open|>instr<|:|>ID<|close|>Summarise the review in one sentence.<|open_end|>
<|open|>data<|:|>ID<|close|>review: Loved it, but shipping was slow.<|open_end|>
<|open|>req<|:|>ID<|close|>Mention the shipping.<|open_end|>
<|open|>exec<|:|>ID<|close|>
<|open|>exec<|.|>instruction<|close|>instr<|:|>ID<|open_end|>
<|open|>exec<|.|>input<|close|>data<|:|>ID<|open_end|>
<|open|>exec<|.|>requirement<|close|>req<|:|>ID<|open_end|>
<|open|>exec<|.|>output<|close|>assistant<|:|>ID<|open_end|>
<|open_end|>
<|open|>assistant<|:|>ID<|close|>
```

## Special tokens in content

Content can contain LLMON's special tokens itself: a retrieved web page might
include `<|open_end|>`, for example. The chat template inserts content as it is, so
that token would close the page's `data` block, and whatever followed it could open
an `instr` block of its own. So `LLMONFormatter` sanitizes content with the `llmon`
library's `LLMONConverters.sanitize`, which replaces each LLMON special token with a
look-alike that is not one (`<|open_end|>` becomes `<!open_end!>`). That covers the
text of every turn and of every field item, so the model reads the text but never as
structure.

Only what passes through the formatter is sanitized, and only for LLMON's tokens. A
system prompt set with `ModelOption.SYSTEM_PROMPT`, the tool definitions, and the
arguments of tool calls reach the chat template without passing through it, and
other special tokens, such as end-of-text, are left as they are.

## What comes back

The chat template's prompt ends by opening the `assistant` block, so the model's
reply is its text, then the block's closer and end-of-text. `LocalHFBackend` decodes
a generation without special tokens, LLMON's among them, so the output's `value` is
the reply text itself, with no markup to extract. A tool call the model wrote stays
part of the reply, where the backend finds it.

Structured output works as usual: `format=` is passed to the backend unchanged. A
`@generative` function is a component of its own, so `LLMONFormatter` raises for it.

## Requirement validation

An instruction's requirements are checked with the LLM-as-a-judge turn described
above, and a sampling strategy repairs an output that fails a check. The strategy
matters on `LocalHFBackend`, which decodes greedily:

- **`RejectionSamplingStrategy`**, the default for `m.instruct`, sends the same
  prompt again, which the model answers the same way, so it cannot fix a miss.
- **`MultiTurnStrategy`** keeps the failed answer in the conversation and adds a
  message naming the requirements it missed.
- **`RepairTemplateStrategy`** sends the instruction again with the failed
  requirements as a further `instruction` item. A second try that fails the same
  requirements sends the same prompt again.

Use `MultiTurnStrategy` or `RepairTemplateStrategy` for an LLMON model. The judge is
unreliable on requirements it has to count (an exact number of words, how often a
word appears), so check those with a validation function instead.

## The `llmon` library dependency

`LLMONFormatter` uses the `llmon` converter library (`llmon.lang`) to sanitize
content. It is an optional dependency:

```bash
pip install "mellea[llmon]"
```

Constructing an `LLMONFormatter` without it raises an `ImportError` that names the
extra. Models that are not LLMON models never import it.

---

**See also:** [Requirements System](requirements-system.md) — how requirements are
validated; [Context and Sessions](context-and-sessions.md) — how `m.instruct`
reaches the backend that carries the message to the LLMON model.
