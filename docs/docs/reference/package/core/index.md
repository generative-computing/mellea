---
id: index
title: "mellea.core"
sidebar_label: "Overview"
sidebar_position: 0
description: "Core abstractions for the mellea library."
# diataxis: reference
---

Source: [`mellea/core/__init__.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/core/__init__.py) at commit `a535fc6345a0`.

Core abstractions for the mellea library.

Declared exports (`__all__`): `AudioBlock`, `AudioUrlBlock`, `Backend`, `BaseModelSubclass`, `C`, `CBlock`, `Component`, `ComponentParseError`, `ComputedModelOutputThunk`, `Context`, `ContextTurn`, `Formatter`, `GenerateLog`, `GenerateType`, `GenerationMetadata`, `ImageBlock`, `ImageUrlBlock`, `MelleaLogger`, `ModelOutputThunk`, `ModelToolCall`, `PartialValidationResult`, `RawProviderResponse`, `Requirement`, `S`, `SampleActionType`, `SamplingResult`, `SamplingStrategy`, `Span`, `TemplateRepresentation`, `ValidationResult`, `blockify`, `clear_log_context`, `default_output_to_bool`, `generate_walk`, `get_audio_from_component`, `get_images_from_component`, `log_context`, `make_image_block`, `set_log_context`

## Modules

- [`mellea.core.backend`](backend.md) — Abstract `Backend` interface and generation-walk utilities.
- [`mellea.core.base`](base.md) — Foundational data structures for mellea's generative programming model.
- [`mellea.core.formatter`](formatter.md) — Abstract `Formatter` interface for rendering components to strings.
- [`mellea.core.requirement`](requirement.md) — `Requirement` interface for constrained and validated generation.
- [`mellea.core.sampling`](sampling.md) — Abstract interfaces for sampling strategies and their results.
- [`mellea.core.utils`](utils.md) — Logging utilities for the mellea core library.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
