---
id: index
title: "mellea.stdlib"
sidebar_label: "Overview"
sidebar_position: 0
description: "The mellea standard library of components, sessions, and sampling strategies."
# diataxis: reference
---

Source: [`mellea/stdlib/__init__.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/__init__.py) at commit `a535fc6345a0`.

The mellea standard library of components, sessions, and sampling strategies.

Declared exports (`__all__`): `ChunkEvent`, `ChunkingStrategy`, `CompletedEvent`, `ErrorEvent`, `FullValidationEvent`, `ParagraphChunker`, `QuickCheckEvent`, `SentenceChunker`, `StreamChunkingResult`, `StreamEvent`, `StreamingDoneEvent`, `WordChunker`, `stream_with_chunking`

## Modules

- [`mellea.stdlib.chunking`](chunking.md) — ChunkingStrategy ABC and built-in implementations for streaming validation.
- [`mellea.stdlib.components`](components.md) — Module for Components.
- [`mellea.stdlib.context`](context.md) — Concrete `Context` implementations and the `Compactor` protocol.
- [`mellea.stdlib.frameworks`](frameworks.md) — Problem solving frameworks.
- [`mellea.stdlib.functional`](functional.md) — Low-level primitives for Mellea operations: Instruct, Chat, and friends.
- [`mellea.stdlib.requirements`](requirements.md) — Module for working with Requirements.
- [`mellea.stdlib.sampling`](sampling.md) — sampling methods go here.
- [`mellea.stdlib.session`](session.md) — `MelleaSession`: the primary entry point for running generative programs.
- [`mellea.stdlib.start_backend`](start_backend.md) — Typed `start_backend` with overloaded return types.
- [`mellea.stdlib.streaming`](streaming.md) — Streaming generation with per-chunk validation.
- [`mellea.stdlib.tools`](tools.md) — Implementations of tools.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
