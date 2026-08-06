---
id: adapters
title: "mellea.backends.adapters"
sidebar_label: "adapters"
sidebar_position: 1
description: "Classes and Functions for Backend Adapters."
# diataxis: reference
---

Source: [`mellea/backends/adapters/__init__.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/adapters/__init__.py) at commit `a535fc6345a0`.

Classes and Functions for Backend Adapters.

Declared exports (`__all__`): `KNOWN_CAPABILITIES`, `Adapter`, `AdapterInput`, `AdapterMixin`, `AdapterSchemaMismatchError`, `AdapterType`, `EmbeddedBinding`, `EmbeddedIntrinsicAdapter`, `IOContract`, `Identity`, `IntrinsicAdapter`, `LocalFileBinding`, `LocalHFAdapter`, `ServerMediatedBinding`, `WeightsBinding`, `fetch_intrinsic_metadata`, `get_adapter_for_intrinsic`, `validate_revision`

---

## Module `mellea.backends.adapters.adapter`

Source: [`mellea/backends/adapters/adapter.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/adapters/adapter.py) at commit `a535fc6345a0`.

Adapter classes for adding fine-tuned modules to inference backends.

### `Adapter`

*class* — [line 32](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/adapters/adapter.py#L32) (`abc.ABC`)

An adapter that can be added to a single backend.

Constructor: `Adapter(name: str, adapter_type: AdapterType)`

### `LocalHFAdapter`

*class* — [line 66](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/adapters/adapter.py#L66) (`Adapter`)

Abstract adapter subclass for locally loaded Hugging Face model backends.

Methods (defined on this class; inherited members not listed):

- `get_local_hf_path(base_model_name: str) -> str` — [line 74](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/adapters/adapter.py#L74)
  Return the local filesystem path from which adapter weights should be loaded.

### `IntrinsicAdapter`

*class* — [line 125](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/adapters/adapter.py#L125) (`LocalHFAdapter`, `_AdapterCore`)

Deprecated shim for adapters that implement adapter functions.

Constructor: `IntrinsicAdapter(intrinsic_name: str, adapter_type: AdapterType = AdapterType.ALORA, config_file: str | pathlib.Path | None = None, config_dict: dict | None = None, base_model_name: str | None = None)`

Methods (defined on this class; inherited members not listed):

- `get_local_hf_path(base_model_name: str) -> str` — [line 261](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/adapters/adapter.py#L261)
  Return the local filesystem path from which adapter weights should be loaded.
- `download_and_get_path(base_model_name: str) -> str` — [line 275](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/adapters/adapter.py#L275)
  Download the required adapter function files if necessary and return the path to them.

### `AdapterMixin`

*class* — [line 339](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/adapters/adapter.py#L339) (`Backend`, `abc.ABC`)

Mixin class for backends capable of utilizing adapters.

Properties:

- `base_model_name` → `str` — Return the short model name used for adapter variant lookup.

Methods (defined on this class; inherited members not listed):

- `add_adapter(adapter: AdapterInput) -> None` — [line 366](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/adapters/adapter.py#L366)
  Register an adapter with this backend so it can be loaded later.
- `list_adapters() -> list[str]` — [line 384](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/adapters/adapter.py#L384)
  Return the qualified names of all adapters registered with this backend.
- `load_peft_adapter(adapter_qualified_name: str) -> None` — [line 395](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/adapters/adapter.py#L395)
  Load a previously registered PEFT adapter into the underlying model.
- `unload_peft_adapter(adapter_qualified_name: str) -> None` — [line 414](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/adapters/adapter.py#L414)
  Unload a previously loaded PEFT adapter from the underlying model.
- `render_controls(adapter_qualified_name: str, active: bool) -> None` — [line 432](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/adapters/adapter.py#L432)
  Render or clear the control tokens for a baked-in embedded adapter.
- `set_request_adapter(adapter_qualified_name: str) -> None` — [line 454](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/adapters/adapter.py#L454)
  Select the adapter to use for the next request.
- `resolve_adapter(name: str) -> _AdapterCore` — [line 475](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/adapters/adapter.py#L475)
  Find or lazily register an adapter by capability name.
- `adapter_scope(adapter: '_AdapterCore | None')` — [line 543](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/adapters/adapter.py#L543)
  Context manager wrapping adapter activation and deactivation.
  *Annotation gaps in source: return type unannotated.*

### `EmbeddedIntrinsicAdapter`

*class* — [line 586](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/adapters/adapter.py#L586) (`_AdapterCore`)

Deprecated shim for adapter functions embedded in a Granite Switch model.

Constructor: `EmbeddedIntrinsicAdapter(intrinsic_name: str, config: dict, technology: str = 'lora')`

Methods (defined on this class; inherited members not listed):

- `from_model_directory(model_path: str | pathlib.Path, intrinsic_name: str | None = None) -> list['EmbeddedIntrinsicAdapter']` *(staticmethod)* — [line 668](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/adapters/adapter.py#L668)
  Load embedded adapters from a Granite Switch model directory.
- `from_hub(repo_id: str, revision: str = 'main', cache_dir: str | None = None, intrinsic_name: str | None = None) -> list['EmbeddedIntrinsicAdapter']` *(staticmethod)* — [line 746](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/adapters/adapter.py#L746)
  Load embedded adapters from a Granite Switch model on Hugging Face Hub.
- `from_source(source: str, revision: str = 'main', cache_dir: str | None = None, intrinsic_name: str | None = None) -> list['EmbeddedIntrinsicAdapter']` *(staticmethod)* — [line 801](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/adapters/adapter.py#L801)
  Load embedded adapters from a local directory or Hugging Face Hub.

### `CustomIntrinsicAdapter`

*class* — [line 835](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/adapters/adapter.py#L835) (`IntrinsicAdapter`)

Deprecated shim for user-defined custom adapter functions.

Constructor: `CustomIntrinsicAdapter(*, model_id: str, intrinsic_name: str | None = None, base_model_name: str)`

### `get_adapter_for_intrinsic()`

*function* — [line 301](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/adapters/adapter.py#L301)

`get_adapter_for_intrinsic(intrinsic_name: str, intrinsic_adapter_types: list[AdapterType] | tuple[AdapterType, ...], available_adapters: dict[str, T]) -> T | None`

Find an adapter from a dict of available adapters based on the adapter function name and its allowed adapter types.

---

## Module `mellea.backends.adapters.capabilities`

Source: [`mellea/backends/adapters/capabilities.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/adapters/capabilities.py) at commit `a535fc6345a0`.

Advisory registry of known adapter capabilities.

---

## Module `mellea.backends.adapters.catalog`

Source: [`mellea/backends/adapters/catalog.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/adapters/catalog.py) at commit `a535fc6345a0`.

Catalog of available adapter functions.

### `AdapterType`

*class* — [line 42](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/adapters/catalog.py#L42) (`enum.Enum`)

Possible types of adapters for a backend.

### `IntrinsicsCatalogEntry`

*class* — [line 54](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/adapters/catalog.py#L54) (`pydantic.BaseModel`)

A single row in the main adapter function catalog table.

Properties:

- `effective_capability` → `str` — Return the stable capability token for this adapter function.

### `validate_revision()`

*function* — [line 15](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/adapters/catalog.py#L15)

`validate_revision(revision: str) -> str`

Validate a Hugging Face revision value.

### `known_intrinsic_names()`

*function* — [line 242](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/adapters/catalog.py#L242)

`known_intrinsic_names() -> list[str]`

Return all known user-visible names for adapter functions.

### `fetch_intrinsic_metadata()`

*function* — [line 251](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/backends/adapters/catalog.py#L251)

`fetch_intrinsic_metadata(intrinsic_name: str) -> IntrinsicsCatalogEntry`

Retrieve catalog metadata for the adapter that implements an adapter function.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
