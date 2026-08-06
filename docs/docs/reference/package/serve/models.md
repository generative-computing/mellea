---
id: models
title: "mellea.serve.models"
sidebar_label: "models"
sidebar_position: 1
description: "User-facing types for `m serve`."
# diataxis: reference
---

Source: [`mellea/serve/models.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/serve/models.py) at commit `a535fc6345a0`.

User-facing types for `m serve`.

## `TextContent`

*class* — [line 13](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/serve/models.py#L13) (`BaseModel`)

Text content in a multimodal message.

## `ImageUrlContent`

*class* — [line 20](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/serve/models.py#L20) (`BaseModel`)

Image URL content in a multimodal message.

## `InputAudioData`

*class* — [line 31](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/serve/models.py#L31) (`BaseModel`)

Audio data payload for an `input_audio` content part.

## `InputAudioContent`

*class* — [line 40](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/serve/models.py#L40) (`BaseModel`)

Audio content part in an OpenAI-compatible multimodal message.

## `ChatMessage`

*class* — [line 55](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/serve/models.py#L55) (`BaseModel`)

Chat message with support for text-only or multimodal content.

Methods (defined on this class; inherited members not listed):

- `get_text_content() -> str` — [line 70](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/serve/models.py#L70)
  Extract text content from message, handling both string and multimodal formats.
- `get_image_urls() -> list[str]` — [line 85](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/serve/models.py#L85)
  Extract image URLs from message content.
- `get_image_blocks() -> list[ImageBlock | ImageUrlBlock]` — [line 102](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/serve/models.py#L102)
  Extract image blocks from message content.
- `get_audio_blocks() -> list[AudioBlock]` — [line 135](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/serve/models.py#L135)
  Extract audio blocks from message content.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
