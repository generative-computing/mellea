"""Model context-length lookup table.

Maps known model identifiers to their maximum context window in tokens.
The primary lookup uses `ModelIdentifier.context_length` when the caller
passes a `ModelIdentifier` object.  String lookups fall back to a name
table keyed on `hf_model_name` and `ollama_name` of every
`ModelIdentifier` constant defined in `model_ids`.  The table is built
automatically at import time so there is a single source of truth: the
`context_length` field on each constant.

Only `hf_model_name` and `ollama_name` are indexed — matching the two
fields checked in the `ModelIdentifier` branch of `get_context_length`.
Platform-specific IDs (`mlx_name`, `openai_name`, `bedrock_name`) and
tokenizer references (`hf_tokenizer_name`) are intentionally excluded: the
former would populate the table with deployment-platform strings that are
not valid serving names, and the latter is a tokenizer reference, not a
model-serving identifier.
"""

from __future__ import annotations

import mellea.backends.model_ids as _m

from .model_ids import ModelIdentifier


def get_context_length(model_id: str | ModelIdentifier) -> int | None:
    """Return the maximum context length in tokens for a known model, or `None`.

    Priority:

    1. If *model_id* is a `ModelIdentifier` with a non-`None`
       `context_length` field, return that value directly.
    2. Otherwise perform a name-based lookup in the built-in table,
       checking `hf_model_name` then `ollama_name` (for
       `ModelIdentifier` inputs) or the raw string (for string inputs).

    Args:
        model_id: A `ModelIdentifier` instance or a raw model-name string.

    Returns:
        The context length in tokens, or `None` if the model is unknown or
        its context length is not recorded in the catalog.

    Warning:
        The values returned here reflect each model's theoretical maximum context
        window as recorded in the catalog — **not** the context window actually
        used by a running server.  Ollama in particular defaults to
        ``num_ctx=2048`` regardless of the model's rated maximum; a caller that
        relies on this lookup to size a sliding context window will silently
        overflow Ollama's wire-level limit.
    """
    if isinstance(model_id, ModelIdentifier):
        if model_id.context_length is not None:
            return model_id.context_length
        names_to_check = [model_id.hf_model_name, model_id.ollama_name]
    else:
        # LiteLLM and similar backends prefix model names with a provider
        # slug (e.g. "ollama_chat/granite4.1:3b"). Strip the prefix so the
        # bare name can resolve against the table.
        stripped = model_id.split("/", 1)[-1] if "/" in model_id else model_id
        names_to_check = [model_id, stripped] if stripped != model_id else [model_id]

    for name in names_to_check:
        if name and name in _CONTEXT_LENGTH_TABLE:
            return _CONTEXT_LENGTH_TABLE[name]
    return None


def _build_table() -> dict[str, int]:
    table: dict[str, int] = {}
    for obj in vars(_m).values():
        if not isinstance(obj, ModelIdentifier) or obj.context_length is None:
            continue
        for name in (obj.hf_model_name, obj.ollama_name):
            if name:
                if name in table and table[name] != obj.context_length:
                    raise ValueError(
                        f"context_length collision for {name!r}: "
                        f"{table[name]} vs {obj.context_length}"
                    )
                table[name] = obj.context_length
    return table


# Fallback name-keyed table for raw strings and ModelIdentifiers constructed
# without a context_length field.  Built automatically from the ModelIdentifier
# constants in model_ids so there is a single source of truth.
_CONTEXT_LENGTH_TABLE: dict[str, int] = _build_table()
