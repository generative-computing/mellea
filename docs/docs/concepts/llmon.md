---
title: "LLMON (LLM Object Notation)"
description: "How Mellea works with an LLMON model served behind a chat-completions endpoint: the server owns serialization, Mellea sends plain messages, and LLMONInstruction parses any raw-token reply back into a Component."
# diataxis: explanation
sidebar_label: "LLMON"
---

LLMON (LLM Object Notation, [arXiv:2603.22519](https://arxiv.org/abs/2603.22519))
is an LLM-native markup for conveying the *structure* and *semantic metadata* of
text to a language model — for example, distinguishing instructions from data to
reduce prompt-injection risk. Mellea can talk to a model that has been trained on
LLMON without the developer hand-building any markup.

## The served path — the server owns serialization

The supported way to use LLMON with Mellea is against a model **served behind a
chat-completions endpoint** (vLLM, Ollama, RITS, …) with its
`chat_template.jinja` installed **server-side**. On that path the *server* does
the llmonizing: it draws the random block ids, emits the `instr`/`data`/`req`
blocks, and wires the `exec` graph when it applies the chat template. It also
detokenizes the reply.

Because of that, Mellea's ordinary chat backends already suffice. They send plain
`{role, content}` dicts (`to_chat_messages` → `chat.completions.create`), so a
plain `m.instruct` against the served model round-trips with **no new backend and
no formatter swap**:

> **Coming soon:** the outgoing message shape is still being confirmed. Some LLMON
> checkpoints expect the message dict to carry structured `instruction`/`data`
> fields (each with an `exec` flag) that the chat template reads at render time,
> rather than a single `content` string. If your served model works that way, the
> plain `{role, content}` path below may need an out-direction field mapping; check
> the checkpoint's `chat_template.jinja`.

```python
from mellea.backends.openai import OpenAIBackend
from mellea.stdlib.session import MelleaSession

m = MelleaSession(
    backend=OpenAIBackend(
        model_id="my-llmon-model",
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
    )
)
answer = m.instruct("Summarise this feedback in one sentence: {{text}}",
                    user_variables={"text": "loved it, shipping slow"})
```

## `LLMONInstruction` — robust reply parsing

Some servers return an already-detokenized reply (plain prose); others return the
assistant message as **raw Machine LLMON**
(`<|open|>assistant<|:|>…<|close|> … <|open_end|>…`). `LLMONInstruction` makes the
round-trip robust to both. Opt in via `instruction_cls`:

```python
from mellea.stdlib.components.llmon_instruction import LLMONInstruction

result = m.instruct(
    "Summarise this feedback in one sentence: {{text}}",
    user_variables={"text": "loved it, shipping slow"},
    instruction_cls=LLMONInstruction,
)
```

Its `_parse` runs `parse_llmon_reply` on the reply text:

- If the reply contains the Machine-LLMON open token, it is parsed with the
  `llmon` library and the assistant block's text is extracted.
- If the reply is plain prose (the server already detokenized), or if parsing
  fails, the reply is returned unchanged.

`LLMONInstruction` also rejects `format=`: LLMON has its own grammar, so
constraining the same generation to a Pydantic schema is contradictory.

## The `llmon` library dependency

Reply parsing is delegated to the sanctioned `llmon` converter library
(distribution name `llmon`, **import root `converters`**), an optional dependency:

```bash
pip install "mellea[llmon]"
```

The single primitive is `parse_llmon_reply(text: str) -> str` in
`mellea.stdlib.components.llmon`, which the library backs via
`converters.JSON.llmon_converters.LLMONConverters`. Everything else on the way out
is the server's responsibility.

## Related

- [Requirements System](requirements-system.md) — requirements are the sub-parts
  the server renders as id-tagged `req` blocks in an llmonized instruction.
- [Context and Sessions](context-and-sessions.md) — how `m.instruct` reaches the
  backend that carries the message to the served LLMON model.
