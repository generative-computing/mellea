---
id: openai
title: "mellea.backends.openai"
sidebar_label: "openai"
sidebar_position: 13
description: "A generic OpenAI compatible backend that wraps around the openai python sdk."
# diataxis: reference
---

Source: [`mellea/backends/openai.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/openai.py) at commit `a535fc6345a0`.

A generic OpenAI compatible backend that wraps around the openai python sdk.

## `OpenAIBackend`

*class* — [line 71](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/openai.py#L71) (`FormatterBackend`, `AdapterMixin`)

A generic OpenAI compatible backend.

Constructor: `OpenAIBackend(model_id: str | ModelIdentifier = model_ids.OPENAI_GPT_5_1, formatter: ChatFormatter | None = None, base_url: str | None = None, model_options: dict | None = None, *, default_to_constraint_checking_alora: bool = True, load_embedded_adapters: bool = False, adapter_source: str | None = None, api_key: str | None = None, **kwargs)`

Properties:

- `base_model_name` *(type not annotated in source)* — Returns the base_model_id of the model used by the backend. For example, `granite-3.3-8b-instruct` for `ibm-granite/granite-3.3-8b-instruct`.

Methods (defined on this class; inherited members not listed):

- `add_adapter(adapter: AdapterInput) -> None` — [line 256](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/openai.py#L256)
  Register an adapter with this backend.
- `render_controls(adapter_qualified_name: str, active: bool) -> None` — [line 278](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/openai.py#L278)
  No-op for embedded adapters — weights are baked into the model.
- `list_adapters() -> list[str]` — [line 293](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/openai.py#L293)
  Return qualified names of all registered adapters.
- `register_embedded_adapter_model(source: str, *, revision: str = 'main', cache_dir: str | None = None) -> list[str]` — [line 305](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/openai.py#L305)
  Register all embedded adapters from an Embedded Adapter model.
- `filter_openai_client_kwargs(**kwargs) -> dict` *(staticmethod)* — [line 345](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/openai.py#L345)
  Filter kwargs to only include valid OpenAI client constructor parameters.
- `filter_chat_completions_kwargs(model_options: dict) -> dict` — [line 358](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/openai.py#L358)
  Filter model options to only include valid OpenAI chat completions parameters.
- `filter_completions_kwargs(model_options: dict) -> dict` — [line 376](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/openai.py#L376)
  Filter model options to only include valid OpenAI completions parameters.
- `generate_from_chat_context(action: Component[C] | CBlock | ModelOutputThunk, ctx: Context, *, _format: type[BaseModelSubclass] | None = None, model_options: dict | None = None, tool_calls: bool = False) -> tuple[ModelOutputThunk[C], Context]` *(async)* — [line 839](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/openai.py#L839)
  Generate a new completion from the provided Context using this backend's `Formatter`.
- `processing(mot: ModelOutputThunk, chunk: ChatCompletion | ChatCompletionChunk)` *(async)* — [line 1060](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/openai.py#L1060)
  Accumulate content from a single OpenAI response object into the output thunk.
  *Annotation gaps in source: return type unannotated.*
- `post_processing(mot: ModelOutputThunk, tools: dict[str, AbstractMelleaTool], conversation: list[dict], thinking, seed, _format)` *(async)* — [line 1120](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/openai.py#L1120)
  Finalize the output thunk after OpenAI generation completes.
  *Annotation gaps in source: params `thinking`, `seed`, `_format` unannotated; return type unannotated.*

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
