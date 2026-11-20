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
    generate_readme,
    load_invocation_prompt,
    make_readme_jinja_dict,
)


class _FakeSession:
    """Stand-in for `MelleaSession` returning a fixed, valid template answer."""

    def __init__(self) -> None:
        self.description = ""
        self.cleaned_up = False

    def cleanup(self) -> None:
        self.cleaned_up = True

    def instruct(self, description: str, *_args, **_kwargs) -> str:
        self.description = description
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


def _template_vars(tmp_path, item, session=None, **kwargs):
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"item": item, "label": "flywheel"}) + "\n")
    return make_readme_jinja_dict(
        session or _FakeSession(),  # type: ignore[arg-type]
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


def test_llm_prompt_calls_a_flat_adapter_an_adapter_not_an_intrinsic(tmp_path):
    """The prompt steers the description the LLM writes, so it must not call a flat
    repository an intrinsic, which it is not usable as.
    """
    session = _FakeSession()
    _template_vars(tmp_path, "note", session=session, layout="flat")

    prompt = session.description
    assert "machine learning adapter trained on the dataset" in prompt
    assert "what this adapter does" in prompt
    assert "identifier for this adapter" in prompt
    assert "intrinsic adapter" not in prompt
    assert "this intrinsic" not in prompt


def test_llm_prompt_keeps_the_intrinsic_wording_for_intrinsic_repos(tmp_path):
    session = _FakeSession()
    _template_vars(tmp_path, "note", session=session, layout="intrinsic")

    prompt = session.description
    assert "machine learning intrinsic adapter trained on the dataset" in prompt
    assert "what this intrinsic adapter does" in prompt
    assert "identifier for this intrinsic" in prompt


@pytest.mark.parametrize("layout", ["intrinsic", "flat"])
def test_llm_prompt_keeps_the_field_names_the_model_must_fill(tmp_path, layout):
    """`intrinsic_name` is a field identifier the output schema and template use,
    not framing, so it stays in both prompts.
    """
    session = _FakeSession()
    _template_vars(tmp_path, "note", session=session, layout=layout)

    assert "- intrinsic_name:" in session.description
    assert "- intrinsic_name_camelcase:" in session.description


def _generate(monkeypatch, tmp_path, **kwargs):
    """Run `generate_readme` with the LLM session replaced; return (text, session)."""
    session = _FakeSession()
    monkeypatch.setattr("cli.alora.readme_generator.start_session", lambda: session)
    data = tmp_path / "data.jsonl"
    data.write_text(
        json.dumps({"item": "a cracked stembolt", "label": "stembolt"}) + "\n"
    )
    output = tmp_path / "README.md"

    returned = generate_readme(
        dataset_path=str(data),
        base_model="ibm-granite/granite-3.3-2b-instruct",
        prompt_file=None,
        output_path=str(output),
        name="acme/stembolts",
        hints=None,
        **kwargs,
    )

    assert returned == str(output)
    return output.read_text(), session


def test_generate_readme_renders_a_flat_readme_end_to_end(
    monkeypatch, tmp_path, capsys
):
    """The layout and adapter type given to `generate_readme` reach the rendered file."""
    text, session = _generate(monkeypatch, tmp_path, layout="flat", adapter_type="lora")

    assert "AutoPeftModelForCausalLM" in text
    assert "This is a LoRA adapter" in text
    assert "ibm-granite/granite-3.3-2b-instruct" in text
    assert "LocalFileBinding(" not in text
    assert session.cleaned_up
    assert f"Generated README at {tmp_path / 'README.md'}" in capsys.readouterr().out


def test_generate_readme_renders_an_intrinsic_readme_end_to_end(monkeypatch, tmp_path):
    """Without a detected type the intrinsic README documents alora, as before."""
    text, session = _generate(monkeypatch, tmp_path, layout="intrinsic")

    assert "LocalFileBinding(" in text
    assert "AdapterType.ALORA" in text
    assert "LocalHFBackend('ibm-granite/granite-3.3-2b-instruct')" in text
    assert session.cleaned_up


def test_generate_readme_cleans_up_the_session_when_generation_fails(
    monkeypatch, tmp_path
):
    """A failing LLM call must not leave the session running."""
    session = _FakeSession()

    def failing_instruct(*_args, **_kwargs):
        raise RuntimeError("model unavailable")

    session.instruct = failing_instruct  # type: ignore[method-assign]
    monkeypatch.setattr("cli.alora.readme_generator.start_session", lambda: session)
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"item": "x", "label": "y"}) + "\n")

    with pytest.raises(RuntimeError, match="model unavailable"):
        generate_readme(
            dataset_path=str(data),
            base_model="ibm-granite/granite-3.3-2b-instruct",
            prompt_file=None,
            output_path=str(tmp_path / "README.md"),
            name="acme/stembolts",
            hints=None,
        )

    assert session.cleaned_up
