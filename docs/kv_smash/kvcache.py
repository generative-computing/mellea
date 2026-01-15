# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "mellea[hf]",
# ]
# ///
import torch

from mellea.backends.kv_block_helpers import DynamicCache, merge_dynamic_caches
from transformers import AutoModelForCausalLM, PreTrainedTokenizer, AutoTokenizer
import torch

model_id = "ibm-granite/granite-3.3-8b-instruct"
device = torch.device("mps")
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_id)


def cache(toks) -> DynamicCache:
    dc = DynamicCache()
    with torch.no_grad():
        rv = model(
            toks["input_ids"].to(device),
            attention_mask=toks["attention_mask"].to(device),
            past_key_values=dc,
        ).past_key_values
    return rv


def merge(strs: list[str]):
    strs_toks = [tokenizer(x, return_tensors="pt") for x in strs]
    strs_dcs = [cache(toks) for toks in strs_toks]

    merged_toks = torch.cat([toks["input_ids"] for toks in strs_toks], dim=1)
    merged_masks = torch.cat([toks["attention_mask"] for toks in strs_toks], dim=1)
    merged_dcs = merge_dynamic_caches(strs_dcs)

    return merged_toks, merged_masks, merged_dcs


strs = ["this is a test", "this is another test"]

merged_toks, merged_masks, merged_dcs = merge(strs)
merged_dcs.crop(-1)

result = model.generate(
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
