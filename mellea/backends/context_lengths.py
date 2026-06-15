"""Model context-length lookup table.

Maps known model identifiers to their maximum context window in tokens.
The primary lookup uses ``ModelIdentifier.context_length`` when the caller
passes a ``ModelIdentifier`` object.  String lookups fall back to a name
table keyed on ``hf_model_name`` and ``ollama_name``.
"""

from __future__ import annotations

from .model_ids import ModelIdentifier


def get_context_length(model_id: str | ModelIdentifier) -> int | None:
    """Return the maximum context length in tokens for a known model, or ``None``.

    Priority:

    1. If *model_id* is a ``ModelIdentifier`` with a non-``None``
       ``context_length`` field, return that value directly.
    2. Otherwise perform a name-based lookup in the built-in table,
       checking ``hf_model_name`` then ``ollama_name`` (for
       ``ModelIdentifier`` inputs) or the raw string (for string inputs).

    Args:
        model_id: A ``ModelIdentifier`` instance or a raw model-name string.

    Returns:
        The context length in tokens, or ``None`` if the model is unknown or
        its context length is not recorded in the catalog.
    """
    if isinstance(model_id, ModelIdentifier):
        if model_id.context_length is not None:
            return model_id.context_length
        names_to_check = [model_id.hf_model_name, model_id.ollama_name]
    else:
        names_to_check = [model_id]

    for name in names_to_check:
        if name and name in _CONTEXT_LENGTH_TABLE:
            return _CONTEXT_LENGTH_TABLE[name]
    return None


# Fallback name-keyed table for raw strings and ModelIdentifiers constructed
# without a context_length field.  Keyed on both hf_model_name and ollama_name
# so either variant resolves correctly.
_CONTEXT_LENGTH_TABLE: dict[str, int] = {
    # IBM Granite 4.x dense
    "ibm-granite/granite-4.1-3b": 131072,
    "granite4.1:3b": 131072,
    "ibm-granite/granite-4.1-8b": 131072,
    "granite4.1:8b": 131072,
    "ibm-granite/granite-4.1-30b": 131072,
    "granite4.1:30b": 131072,
    # IBM Granite 4.x hybrid
    "ibm-granite/granite-4.0-h-micro": 131072,
    "granite4:micro-h": 131072,
    "ibm-granite/granite-4.0-h-tiny": 131072,
    "granite4:tiny-h": 131072,
    "ibm-granite/granite-4.0-h-small": 131072,
    "granite4:small-h": 131072,
    "ibm-granite/granite-4.0-h-1b": 131072,
    "granite4:1b-h": 131072,
    "ibm-granite/granite-4.0-h-350m": 32768,
    "granite4:350m-h": 32768,
    # IBM Granite 4.0 micro (pre-built)
    "ibm-granite/granite-4.0-micro": 131072,
    "granite4:micro": 131072,
    # IBM Granite 3.x
    "ibm-granite/granite-3.3-8b-instruct": 131072,
    "granite3.3:8b": 131072,
    "ibm-granite/granite-3.2-8b-instruct": 131072,
    "granite3.2:8b": 131072,
    # Meta Llama 4
    "unsloth/Llama-4-Scout-17B-16E-Instruct": 10485760,
    "llama4:scout": 10485760,
    "unsloth/Llama-4-Maverick-17B-128E-Instruct": 1048576,
    "llama4:maverick": 1048576,
    # Meta Llama 3
    "unsloth/Llama-3.3-70B-Instruct": 131072,
    "llama3.3:70b": 131072,
    "unsloth/Llama-3.2-3B-Instruct": 131072,
    "llama3.2:3b": 131072,
    "unsloth/Llama-3.2-1B": 131072,
    "llama3.2:1b": 131072,
    # Mistral
    "mistralai/Mistral-7B-Instruct-v0.3": 32768,
    "mistral:7b": 32768,
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503": 131072,
    "mistral-small:latest": 131072,
    "mistralai/Mistral-Large-Instruct-2411": 131072,
    "mistral-large:latest": 131072,
    # Qwen 3
    "Qwen/Qwen3-0.6B": 32768,
    "qwen3:0.6b": 32768,
    "Qwen/Qwen3-1.7B": 32768,
    "qwen3:1.7b": 32768,
    "Qwen/Qwen3-8B": 40960,
    "qwen3:8b": 40960,
    "Qwen/Qwen3-14B": 40960,
    "qwen3:14b": 40960,
    # Microsoft Phi
    "microsoft/phi-4": 16384,
    "phi4:14b": 16384,
    # DeepSeek
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": 131072,
    "deepseek-r1:8b": 131072,
    # SmolLM2
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": 8192,
    "smollm2:1.7b": 8192,
}
