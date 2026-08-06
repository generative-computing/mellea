---
id: utils
title: "mellea.backends.utils"
sidebar_label: "utils"
sidebar_position: 15
description: "Shared utility functions used across formatter-based backend implementations."
# diataxis: reference
---

Source: [`mellea/backends/utils.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/utils.py) at commit `a535fc6345a0`.

Shared utility functions used across formatter-based backend implementations.

## `get_value()`

*function* — [line 30](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/utils.py#L30)

`get_value(obj: Any, key: str) -> Any`

Get value from dict or object attribute.

## `populate_response_metadata_openai_shape()`

*function* — [line 45](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/utils.py#L45)

`populate_response_metadata_openai_shape(mot: ModelOutputThunk, response: Any) -> None`

Populate response-side fields on `mot.generation` from an OpenAI-shaped response.

## `to_chat()`

*function* — [line 73](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/utils.py#L73)

`to_chat(action: Span, ctx: Context, formatter: ChatFormatter, system_prompt: str | None) -> list[Chat]`

Converts a context and an action into a series of dicts to be passed to apply_chat_template.

## `to_tool_calls()`

*function* — [line 125](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/utils.py#L125)

`to_tool_calls(tools: dict[str, AbstractMelleaTool], decoded_result: str) -> list[ModelToolCall] | None`

Parse a tool call string.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
