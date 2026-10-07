---
title: "LLMON (LLM Object Notation)"
description: "How Mellea talks to an LLMON-trained model over a chat-completions endpoint: LLMONInstruction parses the reply back to plain text and can optionally encode structured input, with no new backend."
# diataxis: explanation
sidebar_label: "LLMON"
---

LLMON (LLM Object Notation, [arXiv:2603.22519](https://arxiv.org/abs/2603.22519))
is an LLM-native markup for conveying the *structure* and *semantic metadata* of
text to a language model — for example, distinguishing instructions from data to
reduce prompt-injection risk. Mellea can talk to a model that has been trained on
LLMON without the developer hand-building any markup.

## Where the LLMONizing happens

LLMON can, in principle, be applied in two places: by the **server's chat
template** when it renders the prompt, or by the **model's own fine-tuned
weights**. These are not interchangeable, and which one applies is a property of
the deployment — so check your model rather than assuming.

The tested deployment (`llmon95:latest`, a fine-tuned Granite served on Ollama)
LLMONizes **in the weights**. Its chat template is a bare passthrough
(`{{ .Prompt }}`) with no system prompt and no `chat_template` in the GGUF
metadata: the server adds no LLMON structure at all. The model nonetheless emits
Machine LLMON in its replies and will infer `instr`/`exec`/`data` structure even
from a plain prompt. (An earlier version of this page claimed the server owns
serialization; that is false for this deployment and has been corrected.)

The practical consequence: Mellea's ordinary chat backends suffice (no new
backend, no formatter swap), and `LLMONInstruction` owns whatever LLMON work is
wanted on the Mellea side — always parsing the reply, and optionally encoding the
input.

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
answer = m.instruct("Summarise this feedback in one sentence: {{text}}",
                    user_variables={"text": "loved it, shipping slow"})
```

## `LLMONInstruction` — reply parsing and optional input encoding

Opt in via `instruction_cls`. The returned value is always **plain text** — an
`LLMONResult` holding the detokenized answer, never LLMON markup:

```python
from mellea.stdlib.components.llmon_instruction import LLMONInstruction

result = m.instruct(
    "Summarise this feedback in one sentence: {{text}}",
    user_variables={"text": "loved it, shipping slow"},
    instruction_cls=LLMONInstruction,
)
```

### Reply-in (always on)

`_parse` runs `parse_llmon_reply` on the reply text:

- If the reply contains the Machine-LLMON open token, the assistant block's text
  is extracted and returned as plain prose. The parser is robust to the
  partial/garbled fragments a served model actually returns (a reply that begins
  mid-stream, carries unescaped control characters, or echoes instructions as a
  preamble): it falls back to scanning for the last well-formed assistant block.
- If the reply is already plain prose, it is returned unchanged.

### Out-direction (opt-in via `llmonize_input=True`)

By default the instruction is sent as plain prose — which round-trips, since the
model infers structure from its weights. Setting `llmonize_input=True` instead
encodes the instruction (and a single grounding document, when exactly one is
present, as the `data` member) into a Machine-LLMON `instr`/`data` object, giving
the model explicit instruction/data boundaries:

```python
instr = LLMONInstruction("Summarise this feedback.", llmonize_input=True)
# format_for_llm() now emits:
#   <|open|>object<|close|><|open|>instr<|close|>Summarise this feedback.…
```

Whether structured input improves a given model's behavior is deployment-dependent
— it is off by default so the plain path is unchanged.

### Constrained decoding (soft vs. hard, opt-in via `constrain_output=True`)

By default generation is **soft**: the model is prompted but not enforced. Setting
`constrain_output=True` is **hard** mode — `output_grammar()` returns the LLMON
reply-object grammar (`llmon_grammar()`), which a grammar-aware backend uses to
mask decoding to valid Machine LLMON:

```python
instr = LLMONInstruction("Summarise this feedback.", constrain_output=True)
# or through the normal API:
m.instruct(
    "Summarise this feedback: {{text}}",
    user_variables={"text": "loved it, shipping slow"},
    instruction_cls=LLMONInstruction,
    instruction_kwargs={"constrain_output": True},
)
```

Enforcement needs a backend with a grammar hook:

- **HuggingFace** (`LocalHFBackend`): enforced via the llguidance logits processor.
- **vLLM-backed OpenAI servers**: plumbed as `extra_body["guided_grammar"]`. The
  grammar is GBNF, which vLLM consumes natively, so it is passed through unchanged
  (experimental — this path is plumbed and mock-tested but not enforcement-tested,
  since vLLM is not available in the test environment).
- **Ollama and official OpenAI**: no grammar hook, so `constrain_output=True` is a
  silent no-op (soft fallback) — exactly how `format=` degrades on backends without
  structured output. The tested `llmon95:latest` is on Ollama, so hard mode cannot
  be enforced against that specific model.

`constrain_output` is orthogonal to `llmonize_input` and may be combined.

### `format=` is rejected

`LLMONInstruction` rejects `format=`: LLMON has its own grammar, so constraining
the same generation to a Pydantic schema is contradictory. Use `constrain_output`
(above) for grammar enforcement instead.

## The `llmon` library dependency

The grammar work is delegated to the sanctioned `llmon` converter library
(distribution name `llmon`, **import root `lang`**), an optional dependency:

```bash
pip install "mellea[llmon]"
```

The primitives in `mellea.stdlib.components.llmon` are
`parse_llmon_reply(text: str) -> str` (reply-in),
`encode_llmon_input(instruction: str, data: str | None) -> str` (out-direction),
and `llmon_grammar() -> str` (the GBNF grammar for hard mode). All are built from
the Machine-LLMON terminals in `lang.common.common` /
`lang.json.llmon_converters.LLMONConverters`.

## Related

- [Requirements System](requirements-system.md) — a `requirements=[...]` instruct
  validates each requirement; `LLMONResult` renders back through the shared message
  template so validation sees the plain text, not raw tokens.
- [Context and Sessions](context-and-sessions.md) — how `m.instruct` reaches the
  backend that carries the message to the LLMON model.
