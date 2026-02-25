# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "mellea[hf]",
# ]
# ///
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from transformers.generation import GenerateDecoderOnlyOutput

from mellea.backends.kv_block_helpers import DynamicCache, merge_dynamic_caches

model_id = "ibm-granite/granite-4.0-tiny-preview"
device = torch.device("mps")
model = AutoModelForCausalLM.from_pretrained(model_id)
# model = model.to(device=device) # this part does not pass mypy; possible misconfiguration
tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_id)


def cache(toks) -> DynamicCache:
    dc = DynamicCache()
    with torch.no_grad():
        rv = model(
            toks["input_ids"].to(model.device),
            attention_mask=toks["attention_mask"].to(model.device),
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

# GenerateDecoderOnlyOutput | GenerateEncoderDecoderOutput | GenerateBeamDecoderOnlyOutput | GenerateBeamEncoderDecoderOutput | LongTensor
result = model.generate(
    merged_toks.to(model.device),
    attention_mask=merged_masks.to(model.device),
    past_key_values=merged_dcs,
    use_cache=True,
    return_dict_in_generate=True,
    output_scores=True,
)

assert isinstance(result, GenerateDecoderOnlyOutput)

result_decoded = tokenizer.decode(
    result.sequences[0, merged_toks.shape[1] :], skip_special_tokens=True
)
