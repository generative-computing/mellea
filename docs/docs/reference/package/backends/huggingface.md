---
id: huggingface
title: "mellea.backends.huggingface"
sidebar_label: "huggingface"
sidebar_position: 7
description: "A backend that uses the Hugging Face Transformers library."
# diataxis: reference
---

Source: [`mellea/backends/huggingface.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/huggingface.py) at commit `a535fc6345a0`.

A backend that uses the Hugging Face Transformers library.

## `HFAloraCacheInfo`

*class* — [line 137](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/huggingface.py#L137) 

A dataclass for holding a KV cache and associated generation metadata.

## `LocalHFBackend`

*class* — [line 310](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/huggingface.py#L310) (`FormatterBackend`, `AdapterMixin`)

The LocalHFBackend uses Hugging Face's transformers library for inference, and uses a Formatter to convert `Component`s into prompts. This backend also supports [aLoRA adapters](https://arxiv.org/pdf/2504.12397).

Constructor: `LocalHFBackend(model_id: str | ModelIdentifier, formatter: ChatFormatter | None = None, *, use_caches: bool = True, cache: Cache | None = None, custom_config: TransformersTorchConfig | None = None, default_to_constraint_checking_alora: bool = True, model_options: dict | None = None)`

Properties:

- `base_model_name` *(type not annotated in source)* — Returns the base_model_id of the model used by the backend. For example, `granite-3.3-8b-instruct` for `ibm-granite/granite-3.3-8b-instruct`.

Methods (defined on this class; inherited members not listed):

- `processing(mot: ModelOutputThunk, chunk: str | GenerateDecoderOnlyOutput, input_ids)` *(async)* — [line 1375](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/huggingface.py#L1375)
  Accumulate decoded text from a streaming chunk or full generation output.
  *Annotation gaps in source: params `input_ids` unannotated; return type unannotated.*
- `post_processing(mot: ModelOutputThunk, conversation: list[dict], _format: type[BaseModelSubclass] | None, tool_calls: bool, tools: dict[str, AbstractMelleaTool], seed, input_ids)` *(async)* — [line 1457](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/huggingface.py#L1457)
  Finalize the output thunk after Hugging Face generation completes.
  *Annotation gaps in source: params `seed`, `input_ids` unannotated; return type unannotated.*
- `cache_get(id: str | int) -> HFAloraCacheInfo | None` — [line 1806](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/huggingface.py#L1806)
  Retrieve a cached `HFAloraCacheInfo` entry by its key.
- `cache_put(id: str | int, v: HFAloraCacheInfo)` — [line 1819](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/huggingface.py#L1819)
  Store an `HFAloraCacheInfo` entry in the cache under the given key.
  *Annotation gaps in source: return type unannotated.*
- `add_adapter(adapter: AdapterInput) -> None` — [line 1990](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/huggingface.py#L1990)
  Register a LoRA/aLoRA adapter with this backend so it can be loaded later.
- `load_peft_adapter(adapter_qualified_name: str) -> None` — [line 2035](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/huggingface.py#L2035)
  Load a previously registered adapter into the underlying Hugging Face model.
- `unload_peft_adapter(adapter_qualified_name: str) -> None` — [line 2078](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/huggingface.py#L2078)
  Unload a previously loaded adapter from the underlying Hugging Face model.
- `list_adapters() -> list[str]` — [line 2101](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/huggingface.py#L2101)
  List the qualified names of all adapters registered with this backend.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
