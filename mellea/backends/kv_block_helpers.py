"""Low-level utilities for concatenating transformer KV caches (KV smashing).

Provides functions for merging ``DynamicCache`` and legacy tuple caches along the
time axis (``merge_dynamic_caches``, ``legacy_cache_smash``), and
``tokens_to_legacy_cache`` for converting a tokenized prompt into a prefilled KV
cache. These helpers are used internally by local HuggingFace backends that reuse
cached prefix computations across multiple generation calls.
"""

from collections.abc import Iterable
from functools import reduce
from typing import Any

try:
    import torch
    from transformers import PreTrainedModel
    from transformers.cache_utils import DynamicCache
    from transformers.tokenization_utils_base import BatchEncoding
except ImportError as e:
    raise ImportError(
        "The HuggingFace backend requires extra dependencies. "
        'Please install them with: pip install "mellea[hf]"'
    ) from e

TokenizedCacheIterleaving = Iterable[BatchEncoding | DynamicCache]
LegacyCache = Any


def legacy_cache_smash(a: LegacyCache, b: LegacyCache) -> LegacyCache:
    """Concatenates two LegacyCache Ks and Vs along the time axis.

    Args:
        a: First legacy KV cache (tuple of per-layer (K, V) tensor pairs).
        b: Second legacy KV cache to concatenate after ``a``.

    Returns:
        New legacy cache with ``b`` appended to ``a`` along the sequence dimension.
    """
    legacy_merged = tuple(
        (torch.cat([a[i][0], b[i][0]], dim=2), torch.cat([a[i][1], b[i][1]], dim=2))
        for i in range(len(a))
    )
    return legacy_merged


def merge_dynamic_caches(caches: Iterable[DynamicCache]) -> DynamicCache:
    """Merges two DynamicCache Ks and Vs along the time axis.

    Args:
        caches: Iterable of ``DynamicCache`` objects to merge in order.

    Returns:
        A single ``DynamicCache`` with all caches concatenated along the sequence dimension.
    """
    legacies = [c.to_legacy_cache() for c in caches]  # type: ignore
    assert len(legacies) >= 1
    rv = DynamicCache.from_legacy_cache(reduce(legacy_cache_smash, legacies))  # type: ignore
    return rv  # type: ignore


def tokens_to_legacy_cache(
    model: PreTrainedModel, device: str, tokens_or_cache: BatchEncoding | DynamicCache
) -> Iterable[LegacyCache]:
    """Prefills and returns Ks and Vs as a LegacyCache.

    Args:
        model: The HuggingFace model used for prefill.
        device: Target device string (e.g. ``"cuda"``, ``"cpu"``).
        tokens_or_cache: Either a ``BatchEncoding`` to prefill, or an existing
            ``DynamicCache`` to convert directly.

    Returns:
        Legacy KV cache representation as a tuple of per-layer (K, V) tensor pairs.
    """
    if type(tokens_or_cache) is DynamicCache:
        return tokens_or_cache.to_legacy_cache()  # type: ignore
    else:
        tokens = tokens_or_cache
        dc = DynamicCache()
        with torch.no_grad():
            dc = model(
                tokens["input_ids"].to(device),  # type: ignore
                attention_mask=tokens["attention_mask"].to(device),  # type: ignore
                past_key_values=dc,
            ).past_key_values
        return dc.to_legacy_cache()
