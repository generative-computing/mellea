---
id: granite
title: "mellea.formatters.granite"
sidebar_label: "granite"
sidebar_position: 2
description: "Input and output processing code for Granite models and for Granite intrinsics."
# diataxis: reference
---

Source: [`mellea/formatters/granite/__init__.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/__init__.py) at commit `a535fc6345a0`.

Input and output processing code for Granite models and for Granite intrinsics.

Declared exports (`__all__`): `AssistantMessage`, `ChatCompletion`, `ChatCompletionResponse`, `DocumentMessage`, `GraniteChatCompletion`, `IntrinsicsResultProcessor`, `IntrinsicsRewriter`, `UserMessage`, `VLLMExtraBody`

---

## Module `mellea.formatters.granite.base`

Source: [`mellea/formatters/granite/base/__init__.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/base/__init__.py) at commit `a535fc6345a0`.

Shared data structures and functions for formatting code.

---

## Module `mellea.formatters.granite.base.io`

Source: [`mellea/formatters/granite/base/io.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/base/io.py) at commit `a535fc6345a0`.

Input and output processing for chat completions-like APIs.

### `InputProcessor`

*class* — [line 20](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/base/io.py#L20) (`abc.ABC`)

Interface for generic input processors.

Methods (defined on this class; inherited members not listed):

- `transform(chat_completion: ChatCompletion, add_generation_prompt: bool = True) -> str` — [line 27](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/base/io.py#L27)
  Convert the structured representation of the inputs to a completion request.

### `OutputProcessor`

*class* — [line 50](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/base/io.py#L50) (`abc.ABC`)

Base class for generic output processors.

Methods (defined on this class; inherited members not listed):

- `transform(model_output: str, chat_completion: ChatCompletion | None = None) -> AssistantMessage` — [line 60](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/base/io.py#L60)
  Convert the model output into a structured representation.

### `ChatCompletionRewriter`

*class* — [line 81](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/base/io.py#L81) (`abc.ABC`)

Base class for objects that rewrite a chat completion request.

Methods (defined on this class; inherited members not listed):

- `transform(chat_completion: ChatCompletion | str | dict, /, **kwargs) -> ChatCompletion` — [line 88](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/base/io.py#L88)
  Rewrite a chat completion request into another one.

### `ChatCompletionResultProcessor`

*class* — [line 138](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/base/io.py#L138) (`abc.ABC`)

Base class for chat completion result processors.

Methods (defined on this class; inherited members not listed):

- `transform(chat_completion_response: ChatCompletionResponse | dict | pydantic.BaseModel, chat_completion: ChatCompletion | None = None) -> ChatCompletionResponse` — [line 145](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/base/io.py#L145)
  Parse and post-process the result of a chat completion request.

### `Retriever`

*class* — [line 199](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/base/io.py#L199) (`abc.ABC`)

Base class for document retrievers.

Methods (defined on this class; inherited members not listed):

- `retrieve(query: str, top_k: int = 10) -> list[Document]` — [line 206](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/base/io.py#L206)
  Retrieve the top-k matching documents for a query from the corpus.

---

## Module `mellea.formatters.granite.base.optional`

Source: [`mellea/formatters/granite/base/optional.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/base/optional.py) at commit `a535fc6345a0`.

Context-manager helpers for gracefully handling optional import dependencies.

### `import_optional()`

*function* — [line 26](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/base/optional.py#L26)

`import_optional(extra_name: str)`

Handle optional imports.

*Annotation gaps in source: return type unannotated.*

### `nltk_check()`

*function* — [line 46](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/base/optional.py#L46)

`nltk_check(feature_name: str)`

Variation on import_optional for nltk.

*Annotation gaps in source: return type unannotated.*

---

## Module `mellea.formatters.granite.base.types`

Source: [`mellea/formatters/granite/base/types.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/base/types.py) at commit `a535fc6345a0`.

Common Pydantic types shared across the Granite formatter package.

### `NoDefaultsMixin`

*class* — [line 22](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/base/types.py#L22) 

Avoid filling JSON with default values.

### `UserMessage`

*class* — [line 130](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/base/types.py#L130) (`_ChatMessageBase`)

User message for an IBM Granite model chat completion request.

### `DocumentMessage`

*class* — [line 140](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/base/types.py#L140) (`_ChatMessageBase`)

Document message for Granite model chat completion.

### `ToolCall`

*class* — [line 154](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/base/types.py#L154) (`pydantic.BaseModel`, `NoDefaultsMixin`)

Represents a single tool-call entry produced by an assistant message.

### `AssistantMessage`

*class* — [line 177](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/base/types.py#L177) (`_ChatMessageBase`)

Assistant message for chat completion.

### `ToolResultMessage`

*class* — [line 196](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/base/types.py#L196) (`_ChatMessageBase`)

Tool result message from chat completion.

### `SystemMessage`

*class* — [line 211](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/base/types.py#L211) (`_ChatMessageBase`)

System message for an IBM Granite model chat completion request.

### `DeveloperMessage`

*class* — [line 221](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/base/types.py#L221) (`_ChatMessageBase`)

Developer system message for a chat completion request.

### `ToolDefinition`

*class* — [line 243](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/base/types.py#L243) (`pydantic.BaseModel`, `NoDefaultsMixin`)

An entry in the `tools` list in an IBM Granite model chat completion request.

### `Document`

*class* — [line 262](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/base/types.py#L262) (`pydantic.BaseModel`, `NoDefaultsMixin`)

RAG document for retrieval.

### `ChatTemplateKwargs`

*class* — [line 282](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/base/types.py#L282) (`pydantic.BaseModel`)

Keyword arguments for chat template.

### `VLLMExtraBody`

*class* — [line 302](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/base/types.py#L302) (`pydantic.BaseModel`, `NoDefaultsMixin`)

Extra body parameters for vLLM API.

### `ChatCompletion`

*class* — [line 368](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/base/types.py#L368) (`pydantic.BaseModel`, `NoDefaultsMixin`)

Chat completion request schema.

### `GraniteChatCompletion`

*class* — [line 424](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/base/types.py#L424) (`ChatCompletion`)

Granite chat completion request.

### `Logprob`

*class* — [line 494](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/base/types.py#L494) (`pydantic.BaseModel`, `NoDefaultsMixin`)

Prompt log-probability from vLLM API.

### `ChatCompletionLogProb`

*class* — [line 518](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/base/types.py#L518) (`pydantic.BaseModel`, `NoDefaultsMixin`)

Token log-probability from vLLM API.

### `ChatCompletionLogProbsContent`

*class* — [line 542](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/base/types.py#L542) (`ChatCompletionLogProb`)

Token log-probabilities content from vLLM API.

### `ChatCompletionLogProbs`

*class* — [line 559](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/base/types.py#L559) (`pydantic.BaseModel`, `NoDefaultsMixin`)

Token logprobs for chat completion choice.

### `ChatCompletionResponseChoice`

*class* — [line 577](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/base/types.py#L577) (`pydantic.BaseModel`, `NoDefaultsMixin`)

Single choice in chat completion result from vLLM API.

### `ChatCompletionResponse`

*class* — [line 602](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/base/types.py#L602) (`pydantic.BaseModel`, `NoDefaultsMixin`)

Chat completion result from vLLM API.

---

## Module `mellea.formatters.granite.base.util`

Source: [`mellea/formatters/granite/base/util.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/base/util.py) at commit `a535fc6345a0`.

Common utility functions for the library and tests.

Declared exports (`__all__`): `import_optional`

### `random_uuid()`

*function* — [line 32](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/base/util.py#L32)

`random_uuid() -> str`

Generate a random UUID string.

### `load_transformers_lora()`

*function* — [line 41](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/base/util.py#L41)

`load_transformers_lora(local_or_remote_path: str) -> tuple`

Load transformers LoRA model placed on the best available device.

### `chat_completion_request_to_transformers_inputs()`

*function* — [line 161](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/base/util.py#L161)

`chat_completion_request_to_transformers_inputs(request: dict, tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel, constrained_decoding_prefix: str | None = None, ll_tokenizer: llguidance.LLTokenizer | None = None) -> tuple[dict, dict]`

Translate an OpenAI-style chat completion request.

### `generate_with_transformers()`

*function* — [line 339](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/base/util.py#L339)

`generate_with_transformers(tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel, generate_input: dict, other_input: dict) -> ChatCompletionResponse`

Call Transformers generate and get usable results.

---

## Module `mellea.formatters.granite.intrinsics`

Source: [`mellea/formatters/granite/intrinsics/__init__.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/intrinsics/__init__.py) at commit `a535fc6345a0`.

Support for input and output processing for intrinsic models.

Declared exports (`__all__`): `IntrinsicsResultProcessor`, `IntrinsicsRewriter`, `obtain_io_yaml`, `obtain_lora`

---

## Module `mellea.formatters.granite.intrinsics.constants`

Source: [`mellea/formatters/granite/intrinsics/constants.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/intrinsics/constants.py) at commit `a535fc6345a0`.

Constants relating to of input and output processing for RAG-related intrinsics.

---

## Module `mellea.formatters.granite.intrinsics.input`

Source: [`mellea/formatters/granite/intrinsics/input.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/intrinsics/input.py) at commit `a535fc6345a0`.

Classes and functions that implement common aspects of input processing for intrinsics.

### `IntrinsicsRewriter`

*class* — [line 171](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/intrinsics/input.py#L171) (`ChatCompletionRewriter`)

General-purpose chat completion rewriter for intrinsics.

Constructor: `IntrinsicsRewriter(/, config_file: str | pathlib.Path | None = None, config_dict: dict | None = None, model_name: str | None = None)`

### `sentence_delimiter()`

*function* — [line 39](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/intrinsics/input.py#L39)

`sentence_delimiter(tag: str, sentence_num: int) -> str`

Return a tag string that identifies the beginning of the indicated sentence.

### `mark_sentence_boundaries()`

*function* — [line 53](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/intrinsics/input.py#L53)

`mark_sentence_boundaries(split_strings: list[list[str]], tag_prefix: str, index: int = 0) -> tuple[list[str], int]`

Modify input strings by inserting sentence boundary markers.

### `move_documents_to_message()`

*function* — [line 83](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/intrinsics/input.py#L83)

`move_documents_to_message(chat_completion: ChatCompletion | dict, how: str = 'string') -> ChatCompletion | dict`

Move RAG documents from extra_body to first message.

---

## Module `mellea.formatters.granite.intrinsics.output`

Source: [`mellea/formatters/granite/intrinsics/output.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/intrinsics/output.py) at commit `a535fc6345a0`.

Classes and functions that implement common aspects of output processing for intrinsics.

### `TransformationRule`

*class* — [line 42](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/intrinsics/output.py#L42) (`abc.ABC`)

Base class for transformation rules to apply to JSON outputs of intrinsics.

Constructor: `TransformationRule(config: dict, input_path_expr: list[str | int | None])`

Methods (defined on this class; inherited members not listed):

- `rule_name() -> str` — [line 108](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/intrinsics/output.py#L108)
  Return the YAML name that identifies this transformation rule.
- `apply(parsed_json: Any, reparsed_json: Any, logprobs: ChatCompletionLogProbs | None, chat_completion: ChatCompletion | None) -> Any` — [line 139](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/intrinsics/output.py#L139)
  Apply this transformation rule to the parsed model output.

### `InPlaceTransformation`

*class* — [line 198](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/intrinsics/output.py#L198) (`TransformationRule`)

Base class for TransformationRules that replace values in place in JSON.

### `AddFieldsTransformation`

*class* — [line 230](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/intrinsics/output.py#L230) (`TransformationRule`)

Base class for TransformationRules that add values to JSON.

### `TokenToFloat`

*class* — [line 294](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/intrinsics/output.py#L294) (`InPlaceTransformation`)

Transformation rule that decodes token logprobs to a floating point number.

Constructor: `TokenToFloat(config: dict, input_path_expr: list[str | int | None], /, categories_to_values: dict[str | int | bool, float] | None = None)`

### `DecodeSentences`

*class* — [line 498](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/intrinsics/output.py#L498) (`AddFieldsTransformation`)

Transformation rule that decodes sentence refs into begin, end, text tuples.

Constructor: `DecodeSentences(config: dict, input_path_expr: list[str | int | None], /, source: str | list[str], output_names: dict)`

### `Explode`

*class* — [line 739](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/intrinsics/output.py#L739) (`InPlaceTransformation`)

Expand list-valued attributes in a list of records.

Constructor: `Explode(config, input_path_expr, /, target_field)`
*Annotation gaps in source: params `config`, `input_path_expr`, `target_field` unannotated.*

### `DropDuplicates`

*class* — [line 811](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/intrinsics/output.py#L811) (`InPlaceTransformation`)

Remove duplicate records from a list of records.

Constructor: `DropDuplicates(config, input_path_expr, /, target_fields)`
*Annotation gaps in source: params `config`, `input_path_expr`, `target_fields` unannotated.*

### `Project`

*class* — [line 866](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/intrinsics/output.py#L866) (`InPlaceTransformation`)

Project records down to a specified set of fields.

Constructor: `Project(config, input_path_expr, /, retained_fields)`
*Annotation gaps in source: params `config`, `input_path_expr`, `retained_fields` unannotated.*

### `Nest`

*class* — [line 921](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/intrinsics/output.py#L921) (`InPlaceTransformation`)

Convert a value within a JSON structure into a record with a single field.

Constructor: `Nest(config, input_path_expr, /, field_name)`
*Annotation gaps in source: params `config`, `input_path_expr`, `field_name` unannotated.*

### `MergeSpans`

*class* — [line 952](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/intrinsics/output.py#L952) (`InPlaceTransformation`)

Merge adjacent spans into larger spans.

Constructor: `MergeSpans(config, input_path_expr, /, group_fields: list, begin_field: str, end_field: str, text_field: str | None = None)`
*Annotation gaps in source: params `config`, `input_path_expr` unannotated.*

### `IntrinsicsResultProcessor`

*class* — [line 1222](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/intrinsics/output.py#L1222) (`ChatCompletionResultProcessor`)

General-purpose chat completion result processor for intrinsics.

Constructor: `IntrinsicsResultProcessor(/, config_file: str | pathlib.Path | None = None, config_dict: dict | None = None)`

---

## Module `mellea.formatters.granite.intrinsics.types`

Source: [`mellea/formatters/granite/intrinsics/types.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/intrinsics/types.py) at commit `a535fc6345a0`.

Dataclasses used by the IO code for models that implement intrinsics.

---

## Module `mellea.formatters.granite.intrinsics.util`

Source: [`mellea/formatters/granite/intrinsics/util.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/intrinsics/util.py) at commit `a535fc6345a0`.

Common utility functions for this package.

### `make_config_dict()`

*function* — [line 25](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/intrinsics/util.py#L25)

`make_config_dict(config_file: str | pathlib.Path | None = None, config_dict: dict | None = None) -> dict | None`

Create a configuration dictionary from YAML file or dict.

### `adapter_subpath()`

*function* — [line 101](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/intrinsics/util.py#L101)

`adapter_subpath(intrinsic_name: str, target_model_name: str, repo_id: str, /, alora: bool = False) -> str`

Return the Hugging Face Hub subpath where an intrinsic's adapter lives.

### `obtain_lora()`

*function* — [line 136](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/intrinsics/util.py#L136)

`obtain_lora(intrinsic_name: str, target_model_name: str, repo_id: str, /, revision: str = 'main', alora: bool = False, cache_dir: str | None = None, file_glob: str = '*') -> pathlib.Path`

Download and cache an adapter that implements and intrinsic.

### `obtain_io_yaml()`

*function* — [line 201](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/intrinsics/util.py#L201)

`obtain_io_yaml(intrinsic_name: str, target_model_name: str, repo_id: str, /, revision: str = 'main', alora: bool = False, cache_dir: str | None = None) -> pathlib.Path`

Download cached `io.yaml` configuration file for an intrinsic.

---

## Module `mellea.formatters.granite.retrievers`

Source: [`mellea/formatters/granite/retrievers/__init__.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/retrievers/__init__.py) at commit `a535fc6345a0`.

Support for retrieving documents from various sources.

Declared exports (`__all__`): `ElasticsearchRetriever`, `InMemoryRetriever`, `Retriever`, `compute_embeddings`, `util`, `write_embeddings`

---

## Module `mellea.formatters.granite.retrievers.elasticsearch`

Source: [`mellea/formatters/granite/retrievers/elasticsearch.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/retrievers/elasticsearch.py) at commit `a535fc6345a0`.

Classes and functions that implement the ElasticsearchRetriever.

### `ElasticsearchRetriever`

*class* — [line 9](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/retrievers/elasticsearch.py#L9) 

Retriever for documents hosted on an Elasticsearch server.

Constructor: `ElasticsearchRetriever(corpus_name: str, host: str, **kwargs: Any)`

Methods (defined on this class; inherited members not listed):

- `create_es_body(limit, query)` — [line 41](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/retrievers/elasticsearch.py#L41)
  Create a query body for Elasticsearch.
  *Annotation gaps in source: params `limit`, `query` unannotated; return type unannotated.*
- `retrieve(query: str, top_k: int = 5) -> list[dict]` — [line 66](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/retrievers/elasticsearch.py#L66)
  Run a query against the Elasticsearch index and return top-k results.

---

## Module `mellea.formatters.granite.retrievers.embeddings`

Source: [`mellea/formatters/granite/retrievers/embeddings.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/retrievers/embeddings.py) at commit `a535fc6345a0`.

Classes and functions that implement the InMemoryRetriever.

### `InMemoryRetriever`

*class* — [line 224](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/retrievers/embeddings.py#L224) 

Simple retriever that keeps docs and embeddings in memory.

Constructor: `InMemoryRetriever(data_file_or_table, embedding_model_name: str)`
*Annotation gaps in source: params `data_file_or_table` unannotated.*

Methods (defined on this class; inherited members not listed):

- `retrieve(query: str, top_k: int = 5) -> list[dict]` — [line 263](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/retrievers/embeddings.py#L263)
  Run a query and return results.

### `compute_embeddings()`

*function* — [line 92](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/retrievers/embeddings.py#L92)

`compute_embeddings(corpus, embedding_model_name: str, chunk_size: int = 512, overlap: int = 128)`

Split documents into windows and compute embeddings for each of the the windows.

*Annotation gaps in source: params `corpus` unannotated; return type unannotated.*

### `write_embeddings()`

*function* — [line 186](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/retrievers/embeddings.py#L186)

`write_embeddings(target_dir: str, corpus_name: str, embeddings, chunks_per_partition: int = 10000) -> pathlib.Path`

Write embeddings.

*Annotation gaps in source: params `embeddings` unannotated.*

---

## Module `mellea.formatters.granite.retrievers.util`

Source: [`mellea/formatters/granite/retrievers/util.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/retrievers/util.py) at commit `a535fc6345a0`.

Various utility functions relating to the MTRAG benchmark data set.

### `download_mtrag_corpus()`

*function* — [line 24](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/retrievers/util.py#L24)

`download_mtrag_corpus(target_dir: str, corpus_name: str) -> pathlib.Path`

Download a corpus file from the MTRAG benchmark if the file hasn't already present.

### `read_mtrag_corpus()`

*function* — [line 57](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/retrievers/util.py#L57)

`read_mtrag_corpus(corpus_file: str | pathlib.Path) -> pa.Table`

Read the documents from one of the MTRAG benchmark's corpora.

### `download_mtrag_embeddings()`

*function* — [line 105](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/formatters/granite/retrievers/util.py#L105)

`download_mtrag_embeddings(embedding_name: str, corpus_name: str, target_dir: str) -> None`

Download precomputed embeddings for a corpus in the MTRAG benchmark.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
