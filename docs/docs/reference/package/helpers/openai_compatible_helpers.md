---
id: openai_compatible_helpers
title: "mellea.helpers.openai_compatible_helpers"
sidebar_label: "openai_compatible_helpers"
sidebar_position: 3
description: "A file for helper functions that deal with OpenAI API compatible helpers."
# diataxis: reference
---

Source: [`mellea/helpers/openai_compatible_helpers.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/helpers/openai_compatible_helpers.py) at commit `a535fc6345a0`.

A file for helper functions that deal with OpenAI API compatible helpers.

## `ToolCallFunction`

*class* — [line 22](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/helpers/openai_compatible_helpers.py#L22) (`TypedDict`)

Function details in a tool call.

## `ToolCallDict`

*class* — [line 29](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/helpers/openai_compatible_helpers.py#L29) (`TypedDict`)

OpenAI-compatible tool call dictionary with ID and function.

## `CompletionUsage`

*class* — [line 37](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/helpers/openai_compatible_helpers.py#L37) (`BaseModel`)

Token usage statistics for a completion request.

## `extract_model_tool_requests()`

*function* — [line 50](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/helpers/openai_compatible_helpers.py#L50)

`extract_model_tool_requests(tools: dict[str, AbstractMelleaTool], response: dict[str, Any]) -> list[ModelToolCall] | None`

Extract tool calls from the dict representation of an OpenAI-like chat response object.

## `chat_completion_delta_merge()`

*function* — [line 116](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/helpers/openai_compatible_helpers.py#L116)

`chat_completion_delta_merge(chunks: list[dict], force_all_tool_calls_separate: bool = False) -> dict`

Merge a list of deltas from `ChatCompletionChunk`s into a single dict representing the `ChatCompletion` choice.

## `should_replay_reasoning()`

*function* — [line 210](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/helpers/openai_compatible_helpers.py#L210)

`should_replay_reasoning(messages: list[Message], provider: str | None) -> list[bool]`

Decide, per message, whether its reasoning trace should be replayed to the provider.

## `message_to_openai_message()`

*function* — [line 246](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/helpers/openai_compatible_helpers.py#L246)

`message_to_openai_message(msg: Message, formatter: Formatter | None = None, *, replay_reasoning: bool = False) -> dict`

Serialise a Mellea `Message` to the format required by OpenAI-compatible API providers.

## `messages_to_docs()`

*function* — [line 333](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/helpers/openai_compatible_helpers.py#L333)

`messages_to_docs(msgs: list[Message]) -> list[dict[str, str]]`

Extract all `Document` objects from a list of `Message` objects.

## `build_completion_usage()`

*function* — [line 359](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/helpers/openai_compatible_helpers.py#L359)

`build_completion_usage(output: ModelOutputThunk) -> CompletionUsage | None`

Build a normalized usage object from a model output, if available.

## `has_tool_calls()`

*function* — [line 385](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/helpers/openai_compatible_helpers.py#L385)

`has_tool_calls(output: ModelOutputThunk) -> bool`

Check if a model output has tool calls.

## `build_tool_calls()`

*function* — [line 402](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/helpers/openai_compatible_helpers.py#L402)

`build_tool_calls(output: ModelOutputThunk) -> list[ToolCallDict] | None`

Build OpenAI-compatible tool calls from a model output, if available.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
