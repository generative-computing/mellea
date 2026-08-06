---
id: ollama
title: "mellea.backends.ollama"
sidebar_label: "ollama"
sidebar_position: 12
description: "A model backend wrapping the Ollama Python SDK."
# diataxis: reference
---

Source: [`mellea/backends/ollama.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/ollama.py) at commit `a535fc6345a0`.

A model backend wrapping the Ollama Python SDK.

## `OllamaModelBackend`

*class* — [line 71](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/ollama.py#L71) (`FormatterBackend`)

A model that uses the Ollama Python SDK for local inference.

Constructor: `OllamaModelBackend(model_id: str | ModelIdentifier = model_ids.IBM_GRANITE_4_1_3B, formatter: ChatFormatter | None = None, base_url: str | None = None, model_options: dict | None = None, timeout: float | None = 300.0)`

Methods (defined on this class; inherited members not listed):

- `is_model_available(model_name)` — [line 192](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/ollama.py#L192)
  Checks if a specific Ollama model is available locally.
  *Annotation gaps in source: params `model_name` unannotated; return type unannotated.*
- `generate_from_chat_context(action: Component[C] | CBlock | ModelOutputThunk, ctx: Context, *, _format: type[BaseModelSubclass] | None = None, model_options: dict | None = None, tool_calls: bool = False) -> ModelOutputThunk[C]` *(async)* — [line 366](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/ollama.py#L366)
  Generate a new completion from the provided context using this backend's formatter.
- `processing(mot: ModelOutputThunk, chunk: ollama.ChatResponse, tools: dict[str, AbstractMelleaTool])` *(async)* — [line 735](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/ollama.py#L735)
  Accumulate text and tool calls from a single Ollama ChatResponse chunk.
  *Annotation gaps in source: return type unannotated.*
- `post_processing(mot: ModelOutputThunk, conversation: list[dict], tools: dict[str, AbstractMelleaTool], _format)` *(async)* — [line 777](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/ollama.py#L777)
  Finalize the output thunk after Ollama generation completes.
  *Annotation gaps in source: params `_format` unannotated; return type unannotated.*

## `chat_response_delta_merge()`

*function* — [line 850](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/ollama.py#L850)

`chat_response_delta_merge(mot: ModelOutputThunk, delta: ollama.ChatResponse)`

Merges the individual ChatResponse chunks from a streaming response into a single ChatResponse.

*Annotation gaps in source: return type unannotated.*

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
