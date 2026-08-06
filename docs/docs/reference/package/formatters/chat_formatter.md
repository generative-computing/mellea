---
id: chat_formatter
title: "mellea.formatters.chat_formatter"
sidebar_label: "chat_formatter"
sidebar_position: 1
description: "`ChatFormatter` for converting context histories to chat-message lists."
# diataxis: reference
---

Source: [`mellea/formatters/chat_formatter.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/chat_formatter.py) at commit `a535fc6345a0`.

`ChatFormatter` for converting context histories to chat-message lists.

## `ChatFormatter`

*class* — [line 18](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/chat_formatter.py#L18) (`Formatter`)

Formatter used by Legacy backends to format Contexts as Messages.

Methods (defined on this class; inherited members not listed):

- `to_chat_messages(cs: list[Span]) -> list[Message]` — [line 21](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/chat_formatter.py#L21)
  Convert a linearized chat history into a list of chat messages.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
