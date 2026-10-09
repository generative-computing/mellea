---
title: "LLMON (LLM Object Notation)"
description: "How Mellea talks to an LLMON-trained model: LLMONFormatter renders every part of a component as LLMON, ending each turn in an exec block, and extracts the model's reply into the output's value."
# diataxis: explanation
sidebar_label: "LLMON"
---

LLMON (LLM Object Notation, [arXiv:2603.22519](https://arxiv.org/abs/2603.22519))
is an LLM-native markup for conveying the *structure* and *semantic metadata* of
text to a language model — for example, distinguishing instructions from data to
reduce prompt-injection risk. Mellea can talk to a model that has been trained on
LLMON without the developer hand-building any markup: `LLMONFormatter` renders
Mellea's components as LLMON on the way in and extracts the model's reply on the
way out.

## Where the LLMONizing happens

LLMON can, in principle, be applied in two places: by the **server's chat
template** when it renders the prompt, or by the **model's own fine-tuned
weights**. These are not interchangeable, and which one applies is a property of
the deployment — so check your model rather than assuming.

The tested deployment (`llmon95:latest`, a fine-tuned Granite served on Ollama)
LLMONizes **in the weights**. Its chat template is a bare passthrough
(`{{ .Prompt }}`) with no system prompt and no `chat_template` in the GGUF
metadata: the server adds no LLMON structure at all, so Mellea owns the LLMON work
on the client side. Mellea's ordinary chat backends carry it; only the formatter
changes.

## Using `LLMONFormatter`

Backends choose `LLMONFormatter` automatically for a registered LLMON model
(`llmon95:latest` today), so a plain `m.instruct(...)` needs no extra setup:

```python
from mellea.backends.openai import OpenAIBackend
from mellea.stdlib.session import MelleaSession

m = MelleaSession(
    backend=OpenAIBackend(
        model_id="llmon95:latest",
        base_url="http://localhost:11434/v1",
        api_key="EMPTY",
    )
)
result = m.instruct(
    "Summarise this feedback in one sentence: {{text}}",
    user_variables={"text": "loved it, shipping slow"},
)
print(result.value)  # Plain text, never LLMON markup.
# Output will vary — LLM responses depend on model and temperature.
```

To use it with any other model, pass it to the backend yourself:

```python
from mellea.backends.openai import OpenAIBackend
from mellea.formatters import LLMONFormatter

backend = OpenAIBackend(
    model_id="my-llmon-model",
    formatter=LLMONFormatter("my-llmon-model"),
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)
```

A model that is not registered and not given an `LLMONFormatter` is unaffected: it
gets the usual `TemplateFormatter` prose.

## What the model receives

`LLMONFormatter` builds the prompt from the same `format_for_llm()` args that
`TemplateFormatter` renders through Jinja2 templates. Every non-empty arg is
output, and a sub-component inside an arg (a `Document` in an instruction's
grounding context, say) is lowered the same way, so its own parts are LLMON too.
The resulting tree is converted by the `llmon` library, which flattens the nesting
into dotted tag names such as `data<|.|>review`.

Instructions and messages use the tag names LLMON models were trained on:

- **Instruction:** the description becomes `instr`, the requirements `req`, and the
  grounding context `data` (keyed by its labels). Its other args keep their own
  names: `icl_examples`, `prefix`, `output_prefix` and `repair` (the
  rejection-sampling feedback).
- **Message:** a user message's content becomes `instr`, and its documents become
  `doc` blocks.
- **Document:** a `doc` block with `text` and `title`.

Any other component nests its args under a tag named after its template: a
`Requirement` being judged becomes a `requirement` block with `description` and
`output`, for example.

System, assistant, and tool turns are sent as their plain text, because the
model's chat template wraps each of them in its own block. A model output in the
history is sent as the model's reply text. An assistant's tool calls travel as the
message's `tool_calls`, which the chat template writes out and wires into the next
`exec` block; calls the model wrote in its reply text, as the HuggingFace backend
leaves them, are moved there once the backend has parsed every one. A tool result
is sent as its text for the template to mark as a tool response.

Every user turn ends with an `exec` block that tells the model what to act on: its
instruction blocks (`instr`, `prefix`, `repair` and `output_prefix`, or the
component's own block), its inputs (`data`, `doc`, `icl_examples`), its
requirements (`req`), and `assistant` as the output. For example, this instruction:

```python
from mellea.stdlib.components import Instruction

Instruction(
    "Summarise the review in one sentence.",
    requirements=["Mention the shipping."],
    grounding_context={"review": "Loved it, but shipping was slow."},
)
```

is sent to `llmon95:latest` as:

```text
<|open|>instr<|close|>Summarise the review in one sentence.<|open_end|><|open|>req<|:|>0<|close|>Mention the shipping.<|open_end|><|open|>data<|.|>review<|close|>Loved it, but shipping was slow.<|open_end|><|open|>exec<|.|>instruction<|close|>instr<|open_end|><|open|>exec<|.|>input<|close|>data<|open_end|><|open|>exec<|.|>requirement<|close|>req<|open_end|><|open|>exec<|.|>output<|close|>assistant<|open_end|>
```

The `exec` block references blocks by tag, without instance ids: the `llmon`
library has no way to assign ids yet short of writing the `<|:|>` token into tag
names by hand.

## Special tokens in content

Content can contain LLMON's special tokens itself: a retrieved web page might
include `<|open_end|>`, for example. Sent as it is, that token would close the
page's `data` block, and whatever followed it could open an `instr` block of its
own. So `LLMONFormatter` sanitizes content with the `llmon` library's
`LLMONConverters.sanitize`, which replaces each LLMON special token with a
look-alike that is not one (`<|open_end|>` becomes `<!open_end!>`). That covers
every string value and key in a user turn and the text of system, assistant, and
tool turns, so the model reads the text but never as structure.

Only what passes through the formatter is sanitized, and only for LLMON's tokens. A
system prompt set with `ModelOption.SYSTEM_PROMPT`, the tool definitions, and the
arguments of tool calls reach the chat template without passing through it, and
other special tokens, such as end-of-text, are left as they are.

## What comes back

When a generation completes, the backend has `LLMONFormatter.parse_output`
extract the reply. The chat template's prompt ends by opening the `assistant`
block, so a reply is normally the model's text followed by the block's closer (and
end-of-text), which is dropped. A reply that opens its own `assistant` block yields
that block's text — the last complete one, if a served model starts mid-structure
or echoes context before answering. Text without markup is kept as it is, and a
tool call the model wrote stays part of the reply, where the backend finds it.

The reply becomes the output's `value`, and the `Message` a chat turn is parsed
into carries the same text, so `print(result)`, validation functions, LLM-as-a-judge
checks, the component's own parsing, and the next turn's history all see plain text.
The text the model generated is kept on `result.raw.text` whenever extraction
changed it.
Streamed deltas from `astream()` are slices of the generated text, so they can
contain markup until the output is complete.

Structured output works as usual: `format=` (and so `@generative`) is passed to the
backend unchanged.

## LLMON dialects

By default `LLMONFormatter` uses the `llmon` library's own converter defaults,
which are the LLMON format recommended for Mellea: flat blocks with no `object` or
`list` wrappers (list items become `tag<|:|>index` blocks), each closed by a bare
`<|open_end|>`. A model trained on a different variant can be given an
`LLMONDialect` that sets the converter options it needs; options it leaves unset
keep the library's defaults. A registered model uses its registered dialect. For
example, the converter's earlier defaults, with `object` and `list` blocks and
closers that repeat their tag name:

```python
from mellea.formatters import LLMONDialect, LLMONFormatter

formatter = LLMONFormatter(
    "my-llmon-model",
    dialect=LLMONDialect(no_composite_tags=False, no_openend_tagname=False),
)
```

## The `llmon` library dependency

The markup work is delegated to the `llmon` converter library (distribution name
`llmon`, **import root `lang`**), an optional dependency:

```bash
pip install "mellea[llmon]"
```

Constructing an `LLMONFormatter` without it raises an `ImportError` that names the
extra. Models that are not LLMON models never import it.

---

**See also:** [Requirements System](requirements-system.md) — how requirements are
validated, including the LLM-as-a-judge check, which sees the output's plain-text
`value`; [Context and Sessions](context-and-sessions.md) — how `m.instruct` reaches
the backend that carries the message to the LLMON model.
