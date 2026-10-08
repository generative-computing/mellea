---
title: "LLMON (LLM Object Notation)"
description: "How Mellea talks to an LLMON-trained model over a chat-completions endpoint: input encoding and reply parsing happen automatically, gated by the served model id, with no new backend."
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
from a plain prompt.

The practical consequence: Mellea's ordinary chat backends suffice (no new
backend, no formatter swap), and Mellea owns the LLMON work on the client side.

## Automatic, model-driven round-trip

The LLMON round-trip is **automatic** and keyed off the served model id. An
allowlist (`mellea.stdlib.components.llmon_models.is_llmon_model`, seeded with
`llmon95:latest`) gates two core seams:

- **Inbound to the model (encode):** `Backend.generate_from_context` — the one
  shared chokepoint every backend funnels through — encodes an `Instruction` into
  a Machine-LLMON `instr`/`data`/`req` object before it is sent, so the model
  receives explicit instruction/data/requirement boundaries.
- **Outbound from the model (strip):** `ModelOutputThunk._set_parsed_repr` strips
  any raw Machine LLMON out of the reply, so the thunk's `parsed_repr` is always
  plain text — never markup.

So a plain `m.instruct(prompt)` against an allowlisted model does the right thing
with no subclass and no per-call flag:

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
# result's thunk carries the detokenized plain text in parsed_repr.
```

A model that is **not** in the allowlist is a pure passthrough: the input is sent
as ordinary prose and the reply is not run through the LLMON stripper, exactly like
any other instruct.

### Reply parsing details

The strip delegates to `parse_llmon_reply`:

- If the reply contains the Machine-LLMON open token, the assistant block's text
  is extracted and returned as plain prose. The parser is robust to the
  partial/garbled fragments a served model actually returns (a reply that begins
  mid-stream, carries unescaped control characters, or echoes instructions as a
  preamble): it falls back to scanning for the last well-formed assistant block.
- If the reply is already plain prose, it is returned unchanged.

Only `parsed_repr` holds the stripped text; the thunk's raw `value` is preserved as
the original record.

### Input encoding details

The encode delegates to `encode_instruction_for_llmon`, which maps the
instruction's fields onto a Machine-LLMON object: the description becomes `instr`;
grounding context becomes `data` (a scalar for a single document, a keyed object
for several); and the requirement descriptions become a `req` list. Any pending
rejection-sampling repair feedback is folded into `instr`, so repair is not dropped
on an LLMON model.

## Hard-mode output grammar (`LLMONInstruction`)

The automatic encode/strip need no subclass. `LLMONInstruction` remains only to opt
into **hard-mode** constrained decoding: `constrain_output=True` makes
`output_grammar()` return the LLMON reply-object grammar (`llmon_grammar()`), which
a grammar-aware backend uses to mask decoding to valid Machine LLMON. The automatic
encode still applies (an `LLMONInstruction` is an `Instruction`), so grammar and
encode compose.

```python
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

### `format=` is rejected against an LLMON model

LLMON has its own grammar, so constraining the same generation to a Pydantic schema
is contradictory. Supplying `format=` against an allowlisted model raises
`ValueError` at the backend chokepoint, and `LLMONInstruction` rejects `format=` in
its constructor. Use `constrain_output` (above) for grammar enforcement instead.

## The `llmon` library dependency

The grammar work is delegated to the sanctioned `llmon` converter library
(distribution name `llmon`, **import root `lang`**), an optional dependency:

```bash
pip install "mellea[llmon]"
```

The allowlist module `mellea.stdlib.components.llmon_models` is deliberately
dependency-free so the core seams can gate on the model id without the extra
installed; the encode/parse primitives in `mellea.stdlib.components.llmon`
(`parse_llmon_reply`, `encode_llmon_input`, `encode_instruction_for_llmon`,
`llmon_grammar`) are imported lazily, only once an allowlisted model has matched.
All are built from the Machine-LLMON terminals in `lang.common.common` /
`lang.json.llmon_converters.LLMONConverters`.

## Related

- [Requirements System](requirements-system.md) — a `requirements=[...]` instruct
  validates each requirement; the judge sees the thunk's `parsed_repr` (the
  detokenized plain text for an LLMON model), not raw tokens.
- [Context and Sessions](context-and-sessions.md) — how `m.instruct` reaches the
  backend that carries the message to the LLMON model.
