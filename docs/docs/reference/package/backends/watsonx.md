---
id: watsonx
title: "mellea.backends.watsonx"
sidebar_label: "watsonx"
sidebar_position: 16
description: "A generic WatsonX.ai compatible backend that wraps around the watson_machine_learning library."
# diataxis: reference
---

Source: [`mellea/backends/watsonx.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/watsonx.py) at commit `a535fc6345a0`.

A generic WatsonX.ai compatible backend that wraps around the watson_machine_learning library.

## `WatsonxAIBackend`

*class* — [line 67](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/watsonx.py#L67) (`FormatterBackend`)

A generic backend class for watsonx SDK.

Constructor: `WatsonxAIBackend(model_id: str | ModelIdentifier = model_ids.IBM_GRANITE_4_HYBRID_SMALL, formatter: ChatFormatter | None = None, base_url: str | None = None, model_options: dict | None = None, *, api_key: str | None = None, project_id: str | None = None, **kwargs)`

Methods (defined on this class; inherited members not listed):

- `filter_chat_completions_kwargs(model_options: dict) -> dict` — [line 220](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/watsonx.py#L220)
  Filter kwargs to only include valid watsonx chat.completions.create parameters.
- `generate_from_chat_context(action: Component[C] | CBlock | ModelOutputThunk, ctx: Context, *, _format: type[BaseModelSubclass] | None = None, model_options: dict | None = None, tool_calls: bool = False) -> ModelOutputThunk[C]` *(async)* — [line 349](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/watsonx.py#L349)
  Generate a new completion from the provided context using this backend's formatter.
- `processing(mot: ModelOutputThunk, chunk: dict)` *(async)* — [line 530](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/watsonx.py#L530)
  Accumulate content from a single WatsonX response dict into the output thunk.
  *Annotation gaps in source: return type unannotated.*
- `post_processing(mot: ModelOutputThunk, conversation: list[dict], tools: dict[str, AbstractMelleaTool], seed, _format)` *(async)* — [line 580](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/watsonx.py#L580)
  Finalize the output thunk after WatsonX generation completes.
  *Annotation gaps in source: params `seed`, `_format` unannotated; return type unannotated.*

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
