# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the flat-layout README sample against a real PEFT adapter.

Compiling the generated sample proves it parses; it does not prove it works. These
tests build a tiny Llama and tokenizer on the fly (no downloads), save genuine
LoRA and aLoRA adapters with PEFT, then execute the rendered sample against them
and compare the prompt it builds with the one `m alora train` formats.
"""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch not installed — install mellea[hf]")
pytest.importorskip("peft", reason="peft not installed — install mellea[hf]")
pytest.importorskip("trl", reason="trl not installed — install mellea[hf]")
pytest.importorskip("jinja2", reason="jinja2 not installed")

from jinja2 import Environment, FileSystemLoader
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from cli.alora.layout import adapter_type_from_config
from cli.alora.readme_generator import make_readme_jinja_dict
from cli.alora.train import load_dataset_from_json

pytestmark = pytest.mark.integration

_SPECIAL_TOKENS = ["<|end_of_text|>", "<|start_of_role|>", "<|end_of_role|>"]
# Braces, quotes, and a newline: the characters that break a naively embedded literal.
_INVOCATION = "<|start_of_role|>check {x}<|end_of_role|>"
_ITEM = 'Flywheel "imbalance" {detected}\nsecond line'
_TEMPLATE_DIR = Path(__file__).parents[2] / "cli" / "alora"


class _FakeSession:
    """Stand-in for `MelleaSession` returning a fixed, valid template answer."""

    def instruct(self, *_args, **_kwargs) -> str:
        return json.dumps(
            {
                "high_level_description": "Finds defective parts.",
                "dataset_description": "Notes and labels.",
                "userid": "acme",
                "intrinsic_name": "stembolts",
                "intrinsic_name_camelcase": "Stembolts",
                "arglist": "description: str",
                "arglist_without_type_annotations": "description",
            }
        )


@pytest.fixture(scope="module")
def tiny_base(tmp_path_factory):
    """A 2-layer Llama with a matching byte-level BPE tokenizer, saved to disk."""
    base_dir = tmp_path_factory.mktemp("tiny_base")

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.train_from_iterator(
        [_ITEM, "Requirement:", "check", "flywheel", "x"] * 20,
        trainers.BpeTrainer(
            vocab_size=400,
            special_tokens=_SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token="<|end_of_text|>",
        pad_token="<|end_of_text|>",
    )
    fast.save_pretrained(base_dir)

    config = LlamaConfig(
        vocab_size=len(fast),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
        pad_token_id=fast.pad_token_id,
        eos_token_id=fast.eos_token_id,
    )
    LlamaForCausalLM(config).save_pretrained(base_dir)
    return base_dir, fast


def _save_adapter(base_dir: Path, tokenizer, adapter_type: str, adapter_dir: Path):
    """Save a freshly initialised adapter the way `m alora train` configures it."""
    activation = (
        {
            "alora_invocation_tokens": tokenizer.encode(
                _INVOCATION, add_special_tokens=False
            )
        }
        if adapter_type == "alora"
        else {}
    )
    peft_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj"],
        task_type="CAUSAL_LM",
        **activation,
    )
    model = get_peft_model(LlamaForCausalLM.from_pretrained(base_dir), peft_config)
    # PEFT's `save_pretrained` overloads want a `str`, unlike transformers'.
    model.save_pretrained(str(adapter_dir))


@pytest.mark.parametrize("adapter_type", ["lora", "alora"])
def test_flat_sample_runs_against_a_real_adapter(tiny_base, tmp_path, adapter_type):
    """The rendered flat sample executes, and its prompt is the one training used."""
    base_dir, tokenizer = tiny_base
    adapter_dir = tmp_path / "adapter"
    _save_adapter(base_dir, tokenizer, adapter_type, adapter_dir)

    # The type must come out of what PEFT really wrote: it saves the activation key
    # for every LoRA config (null unless activated), which once read as "aLoRA".
    saved_config = json.loads((adapter_dir / "adapter_config.json").read_text())
    assert adapter_type_from_config(saved_config) == adapter_type

    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"item": _ITEM, "label": "flywheel"}) + "\n")
    prompt_file = tmp_path / "prompt.json"
    prompt_file.write_text(json.dumps({"invocation_prompt": _INVOCATION}))

    template_vars = make_readme_jinja_dict(
        _FakeSession(),  # type: ignore[arg-type]
        str(data),
        str(base_dir),  # a local path, so `AutoTokenizer.from_pretrained` finds it
        str(prompt_file),
        "acme/stembolts",
        None,
        "flat",
        adapter_type_from_config(saved_config),
    )
    template_vars["samples"] = []
    readme = (
        Environment(loader=FileSystemLoader(_TEMPLATE_DIR))
        .get_template("README_TEMPLATE.jinja")
        .render(**template_vars)
    )
    code = readme.split("```python\n", 1)[1].split("\n```", 1)[0]
    # Point only the Hub repository id at the local adapter.
    code = code.replace('"acme/stembolts"', repr(str(adapter_dir)))

    namespace: dict = {}
    exec(code, namespace)

    training_prompt = load_dataset_from_json(str(data), tokenizer, _INVOCATION)[0][
        "input"
    ]
    assert namespace["prompt"] == training_prompt

    input_ids = namespace["inputs"]["input_ids"][0].tolist()
    output = namespace["output"]
    assert output.shape[1] > len(input_ids), "generate() produced no new tokens"
    if adapter_type == "alora":
        invocation_tokens = saved_config["alora_invocation_tokens"]
        assert any(
            input_ids[i : i + len(invocation_tokens)] == invocation_tokens
            for i in range(len(input_ids))
        ), "the prompt does not contain the aLoRA invocation tokens"
