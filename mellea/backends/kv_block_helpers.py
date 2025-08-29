"""Utilities for KV smashing."""

from collections.abc import Iterable
from functools import reduce
from typing import Any

import torch
from transformers import BatchEncoding, DynamicCache

TokenizedCacheIterleaving = Iterable[BatchEncoding | DynamicCache]
LegacyCache = Any


def legacy_cache_smash(a: LegacyCache, b: LegacyCache) -> LegacyCache:
    """Concatenates two LegacyCache Ks and Vs along the time axis."""
    legacy_merged = tuple(
        (torch.cat([a[i][0], b[i][0]], dim=2), torch.cat([a[i][1], b[i][1]], dim=2))
        for i in range(len(a))
    )
    return legacy_merged


def merge_dynamic_caches(caches: Iterable[DynamicCache]) -> DynamicCache:
    """Merges two DynamicCache Ks and Vs along the time axis."""
    legacies = [c.to_legacy_cache() for c in caches]
    assert len(legacies) >= 1
    rv = DynamicCache.from_legacy_cache(reduce(legacy_cache_smash, legacies))  # type: ignore
    return rv  # type: ignore


def combine_representations(
    tokenizer, reps: Iterable[str | DynamicCache]
) -> TokenizedCacheIterleaving:
    rv = []
    for rep in reps:
        if type(rep) is DynamicCache:
            rv.append(rep)
        else:
            rv.append(tokenizer(rep))
    return rv


def tokens_to_legacy_cache(
    model, device: str, tokens_or_cache: BatchEncoding | DynamicCache
) -> Iterable[LegacyCache]:
    """Prefills and returns Ks and Vs as a LegacyCache."""
    if type(tokens_or_cache) is DynamicCache:
        return tokens_or_cache.to_legacy_cache()
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
