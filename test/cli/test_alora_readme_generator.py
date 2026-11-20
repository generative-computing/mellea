# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the template variables `cli/alora/readme_generator.py` computes.

The LLM call is replaced with a canned answer, so these exercise only the
deterministic variables the README template depends on.
"""

import ast
import json
from pathlib import Path

import pytest

from cli.alora.readme_generator import (
    DEFAULT_INVOCATION_PROMPT,
    load_invocation_prompt,
    make_readme_jinja_dict,
)


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


def _template_vars(tmp_path, item, **kwargs):
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"item": item, "label": "flywheel"}) + "\n")
    return make_readme_jinja_dict(
        _FakeSession(),  # type: ignore[arg-type]
        str(data),
        "ibm-granite/granite-4.1-3b",
        kwargs.pop("prompt_file", ""),
        "acme/stembolts",
        None,
        **kwargs,
    )


def test_default_invocation_prompt_matches_training():
    """The default must stay in step with what `m alora train` falls back to.

    Read as text rather than imported: importing `train` needs the full training
    stack, and only the literal matters here.
    """
    train_source = (
        Path(__file__).parents[2] / "cli" / "alora" / "train.py"
    ).read_text()

    assert f'invocation_prompt = "{DEFAULT_INVOCATION_PROMPT}"' in train_source


def test_load_invocation_prompt_defaults_without_a_file():
    assert load_invocation_prompt(None) == DEFAULT_INVOCATION_PROMPT
    assert load_invocation_prompt("") == DEFAULT_INVOCATION_PROMPT


def test_load_invocation_prompt_reads_the_training_prompt_file(tmp_path):
    """Same file shape as `m alora train --promptfile`."""
    prompt_file = tmp_path / "prompt.json"
    prompt_file.write_text(json.dumps({"invocation_prompt": "<|go|>"}))

    assert load_invocation_prompt(str(prompt_file)) == "<|go|>"


@pytest.mark.parametrize("contents", ['{"other": 1}', '["invocation_prompt"]', '"x"'])
def test_load_invocation_prompt_rejects_files_training_would_reject(tmp_path, contents):
    """A file without an `invocation_prompt` key fails clearly, not with a KeyError."""
    prompt_file = tmp_path / "prompt.json"
    prompt_file.write_text(contents)

    with pytest.raises(ValueError, match="invocation_prompt"):
        load_invocation_prompt(str(prompt_file))


@pytest.mark.parametrize(
    "item",
    [
        "plain note",
        "quotes \" and ' and {braces} and a\nnewline and a back\\slash",
        # Training formats the loaded JSON value with an f-string, so a nested
        # object appears as its Python repr, not as JSON.
        {"part": "flywheel", "count": 2},
    ],
)
def test_example_item_literal_reproduces_the_training_item(tmp_path, item):
    """The rendered literal evaluates to exactly the text training put in the prompt."""
    result = _template_vars(tmp_path, item)

    assert ast.literal_eval(result["example_item_literal"]) == f"{item}"


def test_invocation_prompt_literal_survives_braces_and_quotes(tmp_path):
    """Values that would break a naive embed in an f-string stay valid Python."""
    prompt_file = tmp_path / "prompt.json"
    awkward = "<|start|>check {x} \"q\" 'q' \\<|end|>"
    prompt_file.write_text(json.dumps({"invocation_prompt": awkward}))

    result = _template_vars(tmp_path, "note", prompt_file=str(prompt_file))

    assert ast.literal_eval(result["invocation_prompt_literal"]) == awkward


@pytest.mark.parametrize(
    ("layout", "adapter_type", "expected"),
    [
        # The intrinsic sample needs a concrete type, so an unknown one is alora.
        ("intrinsic", None, "alora"),
        ("intrinsic", "lora", "lora"),
        ("intrinsic", "alora", "alora"),
        # The flat README must not claim a type it does not know.
        ("flat", None, None),
        ("flat", "lora", "lora"),
        ("flat", "alora", "alora"),
    ],
)
def test_adapter_type_defaulting(tmp_path, layout, adapter_type, expected):
    result = _template_vars(tmp_path, "note", layout=layout, adapter_type=adapter_type)

    assert result["adapter_type"] == expected
