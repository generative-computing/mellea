# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the generated `cli/alora/README_TEMPLATE.jinja` code sample.

Nothing renders or executes this template today outside the LLM-assisted
`readme_generator.py` flow, so a construction bug in it (e.g. a missing
required argument) ships silently to every user who trains a custom aLoRA
adapter. These tests render it with representative variables, extract the
fenced Python sample, and execute the adapter/intrinsic construction it
generates for a non-catalog adapter name.
"""

import os
import re
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("jinja2", reason="jinja2 not installed")

_TEMPLATE_VARS = {
    # Deliberately not granite-4.1-3b, the model the sample used to hardcode.
    "base_model": "ibm-granite/granite-3.3-2b-instruct",
    "base_model_literal": repr("ibm-granite/granite-3.3-2b-instruct"),
    "adapter_name": "Stembolts",
    "high_level_description": "Detects defective stembolts from inspection notes.",
    "dataset_description": "JSONL rows of inspection notes and defect labels.",
    "samples": [{"input": "note text", "output": "defective"}],
    "userid": "acme",
    "intrinsic_name": "stembolts",
    "intrinsic_name_camelcase": "Stembolts",
    "arglist": "description: str",
    "arglist_without_type_annotations": "description",
    "arglist_as_kwargs": "description=description",
    "example_call_kwargs": "description='a cracked stembolt'",
    "intrinsic_layout": True,
    "adapter_type": "alora",
}

_FLAT_TEMPLATE_VARS = {
    **_TEMPLATE_VARS,
    "intrinsic_layout": False,
    "example_item_literal": repr("Flywheel imbalance detected."),
    "invocation_prompt_literal": repr(
        "<|start_of_role|>check_requirement<|end_of_role|>"
    ),
}


def _render_template(vars_dict: dict | None = None) -> str:
    from jinja2 import Environment, FileSystemLoader

    template_dir = os.path.join(os.path.dirname(__file__), "..", "..", "cli", "alora")
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("README_TEMPLATE.jinja")
    return template.render(**(vars_dict or _TEMPLATE_VARS))


def _extract_full_sample(readme_content: str) -> str:
    match = re.search(r"```python\n(.*?)\n```", readme_content, re.DOTALL)
    assert match is not None, "README template has no fenced python code sample"
    return match.group(1)


def _extract_python_sample(readme_content: str) -> str:
    # The `if __name__ == "__main__":` block calls the generated function against
    # a real LocalHFBackend — exercise only the adapter/intrinsic construction,
    # not model loading or generation.
    return _extract_full_sample(readme_content).split('if __name__ == "__main__":')[0]


@pytest.mark.unit
def test_readme_template_renders_and_python_sample_compiles():
    """The generated README's fenced Python sample must be syntactically valid."""
    content = _render_template()
    code = _extract_python_sample(content)
    compile(code, "<generated-readme>", "exec")
    assert "LocalFileBinding" in content


def test_readme_template_flat_layout_documents_peft_not_intrinsic_loading():
    """A flat upload must get a PEFT sample, not intrinsic code that cannot find it.

    Regression guard: `add-readme` used to always render the intrinsic sample,
    which fails against a repository uploaded with plain `m alora upload`
    (weights at the root, not under `<name>/<base-model>/<lora|alora>/`).
    """
    content = _render_template(_FLAT_TEMPLATE_VARS)
    code = _extract_python_sample(content)
    compile(code, "<generated-readme-flat>", "exec")

    assert "AutoPeftModelForCausalLM" in code
    assert "acme/stembolts" in code
    assert (
        "AutoTokenizer.from_pretrained('ibm-granite/granite-3.3-2b-instruct')" in code
    )
    # The prose may mention LocalFileBinding; the sample must not use it.
    assert "LocalFileBinding(" not in content
    assert "Intrinsic" not in code


def test_readme_template_flat_sample_prompts_and_decodes_the_adapter():
    """The flat sample must do something with the adapter, not stop at loading it.

    Regression guard: the sample used to end after `from_pretrained`, so it ran
    and showed nothing, and a reader calling `generate()` on plain text got
    base-model output and concluded the adapter was broken. It must build the
    prompt shape training used and print only the generated reply.
    """
    code = _extract_python_sample(_render_template(_FLAT_TEMPLATE_VARS))

    assert (
        'prompt = f"{item}\\nRequirement: <|end_of_text|>\\n{INVOCATION_PROMPT}"'
        in code
    )
    assert (
        "INVOCATION_PROMPT = '<|start_of_role|>check_requirement<|end_of_role|>'"
        in code
    )
    assert "item = 'Flywheel imbalance detected.'" in code
    assert "add_special_tokens=False" in code
    assert "model.generate(" in code
    assert 'inputs["input_ids"].shape[1]' in code


@pytest.mark.parametrize(
    ("adapter_type", "expected", "unexpected"),
    [
        ("alora", "activated LoRA (aLoRA)", "This is a LoRA adapter"),
        ("lora", "This is a LoRA adapter", "aLoRA"),
        (None, "trained on prompts of one fixed shape, ending in", "aLoRA"),
    ],
)
def test_readme_template_flat_prose_follows_the_adapter_type(
    adapter_type, expected, unexpected
):
    """The explanation only claims a type the repository's config confirmed."""
    content = _render_template({**_FLAT_TEMPLATE_VARS, "adapter_type": adapter_type})

    assert expected in content
    assert unexpected not in content


@pytest.mark.unit
def test_readme_template_constructs_composed_adapter_for_non_catalog_name():
    """The generated composed `Adapter`/`Intrinsic` must construct for a non-catalog name.

    Regression guard: the generated `LocalFileBinding(...)` previously omitted
    `revision=`, and the generated `Intrinsic.__init__(...)` previously omitted
    `adapter_types=`. Since a trained user adapter is never in Mellea's
    intrinsics catalog, both omissions raised
    `ValueError: Unknown intrinsic name '...'` the first time a user followed
    this exact generated code.
    """
    code = _extract_python_sample(_render_template())
    namespace: dict = {}
    exec(code, namespace)

    adapter = namespace["_stembolts_adapter"]()
    assert adapter.weights.revision is not None

    intrinsic = namespace["StemboltsIntrinsic"](description="a cracked stembolt")
    assert intrinsic.metadata.name == "stembolts"


@pytest.mark.unit
@pytest.mark.parametrize("adapter_type", ["lora", "alora"])
def test_readme_template_uses_the_detected_adapter_type_throughout(adapter_type):
    """Every place the sample names the adapter type must follow `adapter_type`.

    Regression guard: the sample used to hardcode `AdapterType.ALORA` (the
    binding, the `Intrinsic`, the identity, and the qualified name), so a LoRA
    adapter was documented, registered, and looked up as an aLoRA.
    """
    from mellea.backends.adapters.catalog import AdapterType

    code = _extract_python_sample(
        _render_template({**_TEMPLATE_VARS, "adapter_type": adapter_type})
    )
    namespace: dict = {}
    exec(code, namespace)

    expected = AdapterType(adapter_type)
    adapter = namespace["_stembolts_adapter"]()
    assert adapter.weights.adapter_type is expected
    assert adapter.identity.adapter_type == adapter_type
    assert namespace["_INTRINSIC_QUALIFIED_NAME"] == f"stembolts_{adapter_type}"

    intrinsic = namespace["StemboltsIntrinsic"](description="a cracked stembolt")
    assert intrinsic.adapter_types == (expected,)


@pytest.mark.parametrize(
    "base_model",
    [
        "ibm-granite/granite-3.3-2b-instruct",
        "myorg/my-finetune",
        # A path with characters that would break a naively quoted string.
        '/models/it\'s \\ "mine"',
    ],
)
def test_readme_template_main_loads_the_base_model_the_adapter_was_trained_on(
    monkeypatch, base_model
):
    """The runnable `__main__` must build its backend from `--basemodel`.

    Regression guard: it hardcoded `LocalHFBackend(IBM_GRANITE_4_1_3B)`, but the
    loader looks for weights under the *backend's* base model, so for an adapter
    trained on anything else the README's own code failed with the same
    "not found" error as a flat upload.
    """
    import sys
    import types
    from types import SimpleNamespace

    import mellea.stdlib.functional as mfuncs

    built: list = []

    class RecordingBackend:
        def __init__(self, model_id):
            built.append(model_id)

        def list_adapters(self):
            return []

        def add_adapter(self, adapter):
            pass

    fake_module = types.ModuleType("mellea.backends.huggingface")
    fake_module.LocalHFBackend = RecordingBackend  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mellea.backends.huggingface", fake_module)
    monkeypatch.setattr(
        mfuncs, "act", lambda action, ctx, backend: (SimpleNamespace(value="ok"), ctx)
    )

    code = _extract_full_sample(
        _render_template(
            {
                **_TEMPLATE_VARS,
                "base_model": base_model,
                "base_model_literal": repr(base_model),
            }
        )
    )
    exec(code, {"__name__": "__main__"})

    assert built == [base_model]
    assert "IBM_GRANITE_4_1_3B" not in code


@pytest.mark.unit
def test_readme_template_generated_intrinsic_forwards_args_as_intrinsic_kwargs():
    """The generated `Intrinsic` subclass must pass its constructor arguments
    through as `intrinsic_kwargs`, not only store them on the `SimpleComponent`
    side.

    Regression guard: `_generate_from_intrinsic` explicitly never adds the
    action itself to the context ("Intrinsics modify the context through
    their rewriters") — the *only* path from an `Intrinsic` action's own data
    to the rewritten request is `rewriter.transform(request_json,
    **action.intrinsic_kwargs)`. The generated class used to call
    `Intrinsic.__init__` with no `intrinsic_kwargs=` at all and store the
    caller's arguments only via `SimpleComponent.__init__`, whose
    `format_for_llm()` is never invoked for an `Intrinsic` — every
    user-supplied argument to a generated custom adapter was silently
    dropped before reaching the model.
    """
    code = _extract_python_sample(_render_template())
    namespace: dict = {}
    exec(code, namespace)

    intrinsic = namespace["StemboltsIntrinsic"](description="a cracked stembolt")
    assert intrinsic.intrinsic_kwargs == {"description": "a cracked stembolt"}


@pytest.mark.unit
async def test_readme_template_async_wrapper_constructs_intrinsic_correctly():
    """The generated `async_<name>` wrapper must construct its `Intrinsic` subclass
    with only the arglist, not an extra positional intrinsic-name argument.

    Regression guard: `async_stembolts(...)` previously called
    `StemboltsIntrinsic("stembolts", description)` — an extra leading
    positional the sync `stembolts(...)` wrapper (right below it) never had
    — raising `TypeError: __init__() takes 2 positional arguments but 3 were
    given` for every user who called the generated async wrapper.
    """
    code = _extract_python_sample(_render_template())
    namespace: dict = {}
    exec(code, namespace)

    backend = MagicMock()
    backend.list_adapters.return_value = ["stembolts_alora"]
    fake_ctx = MagicMock()
    backend.generate_from_context = AsyncMock(return_value=("mocked output", fake_ctx))

    result, result_ctx = await namespace["async_stembolts"](
        description="a cracked stembolt", ctx=MagicMock(), backend=backend
    )

    assert result == "mocked output"
    assert result_ctx is fake_ctx
    backend.generate_from_context.assert_awaited_once()
    action = backend.generate_from_context.await_args.args[0]
    assert isinstance(action, namespace["StemboltsIntrinsic"])
