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

import pytest

pytest.importorskip("jinja2", reason="jinja2 not installed")

_TEMPLATE_VARS = {
    "base_model": "ibm-granite/granite-4.1-3b",
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
}


def _render_template() -> str:
    from jinja2 import Environment, FileSystemLoader

    template_dir = os.path.join(os.path.dirname(__file__), "..", "..", "cli", "alora")
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("README_TEMPLATE.jinja")
    return template.render(**_TEMPLATE_VARS)


def _extract_python_sample(readme_content: str) -> str:
    match = re.search(r"```python\n(.*?)\n```", readme_content, re.DOTALL)
    assert match is not None, "README template has no fenced python code sample"
    code = match.group(1)
    # The `if __name__ == "__main__":` block calls the generated function against
    # a real LocalHFBackend — exercise only the adapter/intrinsic construction,
    # not model loading or generation.
    return code.split('if __name__ == "__main__":')[0]


@pytest.mark.unit
def test_readme_template_renders_and_python_sample_compiles():
    """The generated README's fenced Python sample must be syntactically valid."""
    code = _extract_python_sample(_render_template())
    compile(code, "<generated-readme>", "exec")


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
