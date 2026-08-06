---
id: litellm
title: "mellea.backends.litellm"
sidebar_label: "litellm"
sidebar_position: 9
description: "A generic LiteLLM compatible backend that wraps around the openai python sdk."
# diataxis: reference
---

Source: [`mellea/backends/litellm.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/litellm.py) at commit `a535fc6345a0`.

A generic LiteLLM compatible backend that wraps around the openai python sdk.

## `LiteLLMBackend`

*class* — [line 65](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/litellm.py#L65) (`FormatterBackend`)

A generic LiteLLM compatible backend.

Constructor: `LiteLLMBackend(model_id: str = 'ollama_chat/' + str(model_ids.IBM_GRANITE_4_1_3B.ollama_name), formatter: ChatFormatter | None = None, base_url: str | None = None, model_options: dict | None = None)`

Methods (defined on this class; inherited members not listed):

- `processing(mot: ModelOutputThunk, chunk: litellm.ModelResponse | litellm.ModelResponseStream)` *(async)* — [line 506](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/litellm.py#L506)
  Accumulate content and thinking tokens from a single LiteLLM response chunk.
  *Annotation gaps in source: return type unannotated.*
- `post_processing(mot: ModelOutputThunk, conversation: list[dict], tools: dict[str, AbstractMelleaTool], thinking, _format)` *(async)* — [line 589](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/litellm.py#L589)
  Finalize the model output thunk after LiteLLM generation completes.
  *Annotation gaps in source: params `thinking`, `_format` unannotated; return type unannotated.*

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
