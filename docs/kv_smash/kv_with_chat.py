import torch

from mellea.backends.huggingface import LocalHFBackend
from mellea.backends.kv_block_helpers import DynamicCache, merge_dynamic_caches
from mellea.backends.model_ids import IBM_GRANITE_3_3_8B

backend = LocalHFBackend(model_id=IBM_GRANITE_3_3_8B)

model = backend._model
tokenizer = backend._tokenizer
device = backend._device


KV_CACHE: dict[str, DynamicCache] = dict()


def cache(s: str, store=True) -> DynamicCache:
    toks = tokenizer(s, return_tensors="pt")
    dc = DynamicCache()
    with torch.no_grad():
        rv = model(
            toks["input_ids"].to(device),
            attention_mask=toks["attention_mask"].to(device),
            past_key_values=dc,
        ).past_key_values
    KV_CACHE[s] = rv
    return rv


def merge(toks, dcs):
    merged_toks = torch.cat([t["input_ids"] for t in toks], dim=1)
    merged_masks = torch.cat([t["attention_mask"] for t in toks], dim=1)
    merged_dcs = merge_dynamic_caches(dcs)

    return merged_toks, merged_masks, merged_dcs


c_blocks = ["this is a test", "this is another test"]

# pretend this stuff already existed in the cache.
for cb in c_blocks:
    cache(cb)


# apply the chat template to a conversation that contains these strings, but without tokenization.
messages = [
    {"role": "user", "content": c_blocks[0]},
    {"role": "user", "content": "Not cached"},
    {"role": "user", "content": c_blocks[1]},
    {"role": "user", "content": "Also no cash"},
]
templatized_input: str = tokenizer.apply_chat_template(  # type: ignore[assignment]
    conversation=messages, tokenize=False
)

str_parts = []
tok_parts = []
dc_parts = []

current_suffix = templatized_input
partially_cached_templatized_input = list[str | DynamicCache]
for cb in c_blocks:
    parts = current_suffix.split(cb)
    assert len(parts) == 2
    prefix, next_suffix = parts

    if prefix != "":
        # Add the prefix.
        str_parts.append(prefix)
        # Add the tokens and attention mask for the prefix.
        tok_parts.append(tokenizer(prefix, return_tensors="pt"))
        # Add the dynamic cache for the prefix.
        dc_parts.append(cache(prefix, store=False))

    # Add cb itself.
    str_parts.append(cb)
    tok_parts.append(tokenizer(cb, return_tensors="pt"))
    dc_parts.append(KV_CACHE[cb])

    # set the current suffix.
    current_suffix = next_suffix

# REMEMBER: add the final suffix.
if current_suffix != "":
    str_parts.append(current_suffix)
    tok_parts.append(tokenizer(current_suffix, return_tensors="pt"))
    dc_parts.append(cache(current_suffix, store=False))

# Merge everything together.
merged_toks = torch.cat([toks["input_ids"] for toks in tok_parts], dim=1)
merged_masks = torch.cat([toks["attention_mask"] for toks in tok_parts], dim=1)
merged_dcs = merge_dynamic_caches(dc_parts)

# crop the last KV for safety.
merged_dcs.crop(-1)

# generate and print result.
result = model.generate(  # type: ignore[operator]
    merged_toks.to(device),
    attention_mask=merged_masks.to(device),
    past_key_values=merged_dcs,
    use_cache=True,
    return_dict_in_generate=True,
    output_scores=True,
)

result_decoded = tokenizer.decode(
    result.sequences[0, merged_toks.shape[1] :], skip_special_tokens=True
)
print(result_decoded)
