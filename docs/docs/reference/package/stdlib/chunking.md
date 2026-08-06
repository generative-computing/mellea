---
id: chunking
title: "mellea.stdlib.chunking"
sidebar_label: "chunking"
sidebar_position: 1
description: "ChunkingStrategy ABC and built-in implementations for streaming validation."
# diataxis: reference
---

Source: [`mellea/stdlib/chunking.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/chunking.py) at commit `a535fc6345a0`.

ChunkingStrategy ABC and built-in implementations for streaming validation.

Declared exports (`__all__`): `ChunkingStrategy`, `ParagraphChunker`, `SentenceChunker`, `WordChunker`

## `ChunkingStrategy`

*class* — [line 12](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/chunking.py#L12) (`ABC`)

Abstract base class for text chunking strategies used in streaming validation.

Methods (defined on this class; inherited members not listed):

- `split(accumulated_text: str) -> list[str]` — [line 41](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/chunking.py#L41)
  Return complete chunks from accumulated_text, excluding any trailing fragment.
- `flush(accumulated_text: str) -> list[str]` — [line 59](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/chunking.py#L59)
  Return any trailing fragment that `split` withheld.

## `SentenceChunker`

*class* — [line 95](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/chunking.py#L95) (`ChunkingStrategy`)

Splits accumulated text on sentence boundaries.

Methods (defined on this class; inherited members not listed):

- `split(accumulated_text: str) -> list[str]` — [line 108](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/chunking.py#L108)
  Return complete sentences from accumulated_text.
- `flush(accumulated_text: str) -> list[str]` — [line 139](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/chunking.py#L139)
  Return the trailing sentence fragment (if any) as a final chunk.

## `WordChunker`

*class* — [line 170](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/chunking.py#L170) (`ChunkingStrategy`)

Splits accumulated text on whitespace boundaries.

Methods (defined on this class; inherited members not listed):

- `split(accumulated_text: str) -> list[str]` — [line 177](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/chunking.py#L177)
  Return complete words from accumulated_text.
- `flush(accumulated_text: str) -> list[str]` — [line 209](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/chunking.py#L209)
  Return the trailing word fragment (if any) as a final chunk.

## `ParagraphChunker`

*class* — [line 236](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/chunking.py#L236) (`ChunkingStrategy`)

Splits accumulated text on double-newline paragraph boundaries.

Methods (defined on this class; inherited members not listed):

- `split(accumulated_text: str) -> list[str]` — [line 247](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/chunking.py#L247)
  Return complete paragraphs from accumulated_text.
- `flush(accumulated_text: str) -> list[str]` — [line 270](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/chunking.py#L270)
  Return the trailing paragraph fragment (if any) as a final chunk.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
