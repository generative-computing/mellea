"""Utilities for KV smashing."""

from collections.abc import Iterable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.cache_utils import DynamicCache


@torch.no_grad()
def prefill_cache_v5(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    device: torch.device,
) -> tuple[dict, DynamicCache]:
    """Prefills cache for transformers v5."""
    toks = tokenizer(text, return_tensors="pt")
    toks = {k: v.to(device) for k, v in toks.items()}

    dc = DynamicCache()
    out = model(
        input_ids=toks["input_ids"],
        attention_mask=toks["attention_mask"],
        past_key_values=dc,
        use_cache=True,
    )
    dc = out.past_key_values
    dc.crop(-1)
    return toks, dc  # v5 returns DynamicCache (not legacy tuple)


def merge_dynamic_caches_v5(caches: Iterable[DynamicCache]) -> DynamicCache:
    """Merge multiple v5 DynamicCache objects by concatenating KV states along the time axis."""
    caches = list(caches)
    assert len(caches) >= 1

    for c in caches:
        if any(
            getattr(layer, "is_sliding", False) for layer in getattr(c, "layers", [])
        ):
            raise ValueError("Check the issue.")

    merged = DynamicCache()

    # reuse Cache.update() to append each segment's KV to the merged cache per layer.
    # DynamicLayer.update(): self.keys = cat([self.keys, key_states], dim=-2).
    for c in caches:
        for layer_idx, layer in enumerate(c.layers):
            if layer.keys is None or layer.values is None:
                continue
            merged.update(layer.keys, layer.values, layer_idx=layer_idx)

    return merged


def merge_v5(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    strs: list[str],
    device: torch.device,
):
    """Merges DynamicCache for transformers>=5.0.0."""
    strs_toks, strs_dcs = [], []
    for s in strs:
        toks, dc = prefill_cache_v5(model, tokenizer, s, device)
        strs_toks.append(toks)
        strs_dcs.append(dc)

    merged_toks = torch.cat([t["input_ids"] for t in strs_toks], dim=1)
    merged_masks = torch.cat([t["attention_mask"] for t in strs_toks], dim=1)

    merged_dc = merge_dynamic_caches_v5(strs_dcs)

    return merged_toks, merged_masks, merged_dc


if __name__ == "__main__":
    from mellea.backends.huggingface import LocalHFBackend
    from mellea.backends.model_ids import IBM_GRANITE_3_3_8B

    backend = LocalHFBackend(model_id=IBM_GRANITE_3_3_8B.hf_model_name)
    model, tokenizer, device = backend._model, backend._tokenizer, backend._device
    model: PreTrainedModel = model

    docs = [
        "Nathan Fulton is expert in large language models, formal verification, and reinforcement learning. He holds a Ph.D. from Carnegie Mellon University's Computer Science Department and has worked at Amazon Web Services and IBM Research. He currently works at IBM Research - Cambridge.",
        "IBM Research has a headquarters at 1101 Kitchawan Rd in Yorktown Heights and a Cambridge office at 314 Main Street in Cambridge, MA.",
        "What is the address of Nathan's place of work?",
    ]

    merged_tokens, merged_masks, merged_cache = merge_v5(
        model, tokenizer, docs, device=backend._device
    )
    input_ids = merged_tokens.to(device)
    result = model.generate(
        input_ids=input_ids,
        use_cache=True,
        return_dict_in_generate=True,
        past_key_values=merged_cache,
        max_new_tokens=512,
    )
    result = tokenizer.decode(
        result.sequences[0, input_ids.shape[1] :], skip_special_tokens=True
    )
    print(result)
