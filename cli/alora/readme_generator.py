# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM-assisted generator for adapter intrinsic README files.

Uses a `MelleaSession` with rejection sampling to derive README template variables
from a JSONL training dataset — including a high-level description, the inferred
Python argument list, and Jinja2-renderable sample rows. Validates the generated
output with deterministic requirements (correct naming conventions, syntactically
valid argument lists) before rendering the final `INTRINSIC_README.md` via a
Jinja2 template. The rendered integration code matches the target repository's
adapter layout: intrinsic (Mellea adapter-function usage) or flat (plain PEFT).
"""

import ast
import json
import os
from typing import Any, Literal

from pydantic import BaseModel

from cli.alora.layout import AdapterTypeName
from mellea import start_session
from mellea.stdlib.requirements.requirement import check, req, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy
from mellea.stdlib.session import MelleaSession

DEFAULT_INVOCATION_PROMPT = "<|start_of_role|>check_requirement<|end_of_role|>"
"""Invocation string `m alora train` uses when no `--promptfile` is given.

Must match the default in `cli/alora/train.py`; a test checks that it does.
"""


def load_invocation_prompt(prompt_file: str | None) -> str:
    """Return the invocation string the adapter was trained with.

    Reads the file the same way `m alora train` does: a JSON object with an
    `invocation_prompt` key. Without a file, returns the training default.

    Args:
        prompt_file: Path to the `--promptfile` JSON, or `None`/empty for the default.

    Returns:
        The invocation string appended to every training input.

    Raises:
        ValueError: The file has no `invocation_prompt` key, which training would
            have rejected too.
    """
    if not prompt_file:
        return DEFAULT_INVOCATION_PROMPT
    with open(prompt_file, encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict) or "invocation_prompt" not in config:
        raise ValueError(
            f"{prompt_file} must be a JSON object with an 'invocation_prompt' key."
        )
    return config["invocation_prompt"]


class ReadmeTemplateVars(BaseModel):
    """Pydantic model holding all variables required to render the intrinsic README template.

    Attributes:
        high_level_description (str): A 2-3 sentence description of what the intrinsic adapter does.
        dataset_description (str): Brief description of the training dataset contents and format.
        userid (str): HuggingFace user ID (the namespace portion of the model name).
        intrinsic_name (str): Short snake_case identifier for the intrinsic (e.g. `"carbchecker"`).
        intrinsic_name_camelcase (str): CamelCase version of `intrinsic_name` (e.g. `"CarbChecker"`).
        arglist (str): Python function argument list with type hints (e.g. `"description: str"`).
        arglist_without_type_annotations (str): Argument list without type hints (e.g. `"description"`).
    """

    high_level_description: str
    dataset_description: str
    userid: str
    intrinsic_name: str
    intrinsic_name_camelcase: str
    arglist: str
    arglist_without_type_annotations: str


def _parses_as_arglist(code: str) -> bool:
    """Check if a string parses as a valid Python function argument list."""
    try:
        ast.parse(f"def f({code}):\n\tpass")
        return True
    except SyntaxError:
        return False


def _make_requirements(name: str) -> list:
    """Build deterministic check requirements for README generation.

    All are check_only=True so they are not included in the prompt,
    avoiding prompt pollution while still being validated by the
    rejection sampling loop.
    """
    expected_userid = name.split("/")[0]
    expected_intrinsic_name = name.split("/")[1]

    def _parse(output: str) -> ReadmeTemplateVars:
        return ReadmeTemplateVars.model_validate_json(output)

    return [
        check(
            f"intrinsic_name must be '{expected_intrinsic_name}'",
            validation_fn=simple_validate(
                lambda output: (
                    _parse(output).intrinsic_name == expected_intrinsic_name,
                    f"intrinsic_name was '{_parse(output).intrinsic_name}', expected '{expected_intrinsic_name}'",
                )
            ),
        ),
        req(
            "arglist should not be enclosed in parens",
            validation_fn=simple_validate(
                lambda output: (
                    not (
                        _parse(output).arglist[0] == "("
                        and _parse(output).arglist[-1] == ")"
                    ),
                    "arglist was enclosed in parens.",
                )
            ),
        ),
        check(
            f"userid must be '{expected_userid}'",
            validation_fn=simple_validate(
                lambda output: (
                    _parse(output).userid == expected_userid,
                    f"userid was '{_parse(output).userid}', expected '{expected_userid}'",
                )
            ),
        ),
        check(
            "intrinsic_name_camelcase must match intrinsic_name",
            validation_fn=simple_validate(
                lambda output: (
                    _parse(output).intrinsic_name_camelcase.lower()
                    == _parse(output)
                    .intrinsic_name.replace("_", "")
                    .replace("-", "")
                    .lower(),
                    f"'{_parse(output).intrinsic_name_camelcase}' doesn't match '{_parse(output).intrinsic_name}'",
                )
            ),
        ),
        check(
            "arglist must be a valid Python argument list",
            validation_fn=simple_validate(
                lambda output: (
                    _parses_as_arglist(_parse(output).arglist),
                    f"arglist '{_parse(output).arglist}' does not parse as valid Python",
                )
            ),
        ),
        check(
            "arglist_without_type_annotations must be a valid Python argument list",
            validation_fn=simple_validate(
                lambda output: (
                    _parses_as_arglist(_parse(output).arglist_without_type_annotations),
                    f"arglist_without_type_annotations '{_parse(output).arglist_without_type_annotations}' does not parse as valid Python",
                )
            ),
        ),
    ]


def make_readme_jinja_dict(
    m: MelleaSession,
    dataset_path: str,
    base_model: str,
    prompt_file: str,
    name: str,
    hints: str | None,
    layout: Literal["intrinsic", "flat"] = "intrinsic",
    adapter_type: AdapterTypeName | None = None,
) -> dict[str, Any]:
    """Generate all template variables for the intrinsic README using an LLM.

    Loads the first five lines of the JSONL dataset, determines the input structure,
    and uses `m.instruct` with deterministic requirements and rejection sampling to
    generate README template variables.

    Args:
        m: Active `MelleaSession` to use for LLM generation.
        dataset_path: Path to the JSONL training dataset file.
        base_model: Base model ID or path used to train the adapter.
        prompt_file: Path to the prompt format file (empty string if not provided).
        name: Destination model name on Hugging Face Hub
            (e.g. `"acme/carbchecker-alora"`).
        hints: Optional string of additional domain hints to include in the prompt.
        layout: Adapter layout of the target repository; selects the integration
            code the rendered README documents.
        adapter_type: The adapter's type, if known. The intrinsic sample loads it
            as that type and assumes `"alora"` when `None`; the flat README
            describes it as that type, or makes no claim when `None`.

    Returns:
        Dict of Jinja2 template variables for rendering the `INTRINSIC_README.md`.
    """
    # Load first 5 lines of the dataset.
    samples = []
    with open(dataset_path) as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    # Determine if the "item" field values are JSON objects or plain strings.
    item_is_json = False
    item_keys: list[str] = []
    for sample in samples:
        item = sample.get("item", "")
        if isinstance(item, dict):
            item_is_json = True
            item_keys = list(item.keys())
            break
        elif isinstance(item, str):
            try:
                parsed = json.loads(item)
                if isinstance(parsed, dict):
                    item_is_json = True
                    item_keys = list(parsed.keys())
                    break
            except (json.JSONDecodeError, ValueError):
                pass

    if item_is_json:
        arglist_hint = (
            f"The input data items are JSON objects with keys: {item_keys}. "
            f"Use these as the function arglist with str type hints "
            f"E.g., '{', '.join(f'{k}: str' for k in item_keys)}'."
        )
    else:
        arglist_hint = (
            "The input data items are plain strings. Choose a descriptive single "
            "argument name with a str type hint. E.g., 'description: str' or "
            "'notes: str'. Pick a name that reflects what the input data represents."
        )

    # Load prompt file content if provided.
    prompt_content = ""
    if prompt_file:
        with open(prompt_file) as f:
            prompt_content = f.read()

    # Build the instruction description.
    sample_text = "\n".join(json.dumps(s) for s in samples)

    hints_text = ""
    if hints is not None:
        hints = "\n\nHere are some additional details about the domain:\n" + hints

    # A flat repository is not usable as an intrinsic, so its README, and the prompt
    # that steers the LLM writing it, should not call the adapter one.
    kind = "intrinsic adapter" if layout == "intrinsic" else "adapter"
    noun = "intrinsic" if layout == "intrinsic" else "adapter"

    description = f"""You are generating metadata for a README file for a machine learning {kind} trained on the dataset below.{hints_text}

Here are the first few samples from the training dataset (JSONL format):
{sample_text}

Base model: {base_model}
{("Prompt configuration: " + prompt_content) if prompt_content else ""}

{arglist_hint}

Generate appropriate values for each field:
- high_level_description: A 2-3 sentence description of what this {kind} does based on the training data
- dataset_description: A brief description of the training dataset contents and format
- userid: Set this to "your-username" as a placeholder for the HuggingFace user ID. MUST be {name.split("/")[0]}
- intrinsic_name: A short snake_case identifier for this {noun} (e.g., "stembolts"). No hyphens. MUST be {name.split("/")[1]}
- intrinsic_name_camelcase: The CamelCase version of intrinsic_name (e.g., "Stembolt"). No underscores.
- arglist: The Python function argument list with type hints based on the input data structure. This will be used as function parameters.
- arglist_without_type_annotations: The arglist without any type annotations."""

    result = m.instruct(
        description,
        requirements=_make_requirements(name),
        format=ReadmeTemplateVars,
        strategy=RejectionSamplingStrategy(loop_budget=10),
    )
    vars_dict = ReadmeTemplateVars.model_validate_json(str(result)).model_dump()

    # Use model name from the --name arg (strip username/ prefix)
    model_name = name.split("/")[-1] if "/" in name else name
    vars_dict["adapter_name"] = model_name

    # Add formatted samples for template rendering
    formatted_samples = []
    for s in samples:
        item = s.get("item", "")
        if isinstance(item, dict):
            item_str = json.dumps(item)
        else:
            item_str = str(item)
        formatted_samples.append({"input": item_str, "output": str(s.get("label", ""))})
    vars_dict["samples"] = formatted_samples

    # Programmatically build kwargs forwarding string from arglist.
    # E.g. "description, notes" -> "description=description, notes=notes"
    tree = ast.parse(f"def f({vars_dict['arglist_without_type_annotations']}): pass")
    arg_names = [arg.arg for arg in tree.body[0].args.args]  # type: ignore
    vars_dict["arglist_as_kwargs"] = ", ".join(f"{n}={n}" for n in arg_names)

    # Build example call kwargs using actual values from the first sample,
    # so the __main__ block has runnable code instead of undefined variables.
    first_item = samples[0].get("item", "")
    sample_values: dict[str, str] | None = None
    if isinstance(first_item, dict):
        sample_values = {k: str(v) for k, v in first_item.items()}
    elif isinstance(first_item, str):
        try:
            parsed = json.loads(first_item)
            if isinstance(parsed, dict):
                sample_values = {k: str(v) for k, v in parsed.items()}
        except (json.JSONDecodeError, ValueError):
            pass

    if sample_values is not None:
        parts = [f"{n}={sample_values.get(n, '')!r}" for n in arg_names]
    else:
        parts = [f"{arg_names[0]}={str(first_item)!r}"]
    vars_dict["example_call_kwargs"] = ", ".join(parts)

    # What the flat sample feeds the model: the first training item, formatted the
    # way training formatted it, then the invocation string. Both are rendered as
    # Python literals so quotes, braces, and backslashes in the data stay valid.
    vars_dict["example_item_literal"] = repr(str(first_item))
    vars_dict["invocation_prompt_literal"] = repr(load_invocation_prompt(prompt_file))

    vars_dict["base_model"] = base_model
    # Rendered into sample code, so as a literal: it may be a path with odd characters.
    vars_dict["base_model_literal"] = repr(base_model)
    vars_dict["intrinsic_layout"] = layout == "intrinsic"
    vars_dict["adapter_type"] = adapter_type or (
        "alora" if layout == "intrinsic" else None
    )

    return vars_dict


def generate_readme(
    dataset_path: str,
    base_model: str,
    prompt_file: str | None,
    output_path: str,
    name: str,
    hints: str | None,
    layout: Literal["intrinsic", "flat"] = "intrinsic",
    adapter_type: AdapterTypeName | None = None,
) -> str:
    """Generate an INTRINSIC_README.md file from the dataset and template.

    Creates a `MelleaSession`, uses the LLM to generate template variables,
    renders the Jinja template, and writes the result to `output_path`.

    Args:
        dataset_path: Path to the JSONL training dataset file.
        base_model: Base model ID or path used to train the adapter.
        prompt_file: Path to the prompt format file, or `None`.
        output_path: Destination path for the generated README file.
        name: Destination model name on Hugging Face Hub.
        hints: Optional string of additional domain hints for the LLM.
        layout: Adapter layout of the target repository. `"intrinsic"` documents
            Mellea adapter-function usage; `"flat"` documents plain PEFT usage.
        adapter_type: The adapter's type, if known. The intrinsic sample loads it
            as that type and assumes `"alora"` when `None`; the flat README
            describes it as that type, or makes no claim when `None`.

    Returns:
        The path to the written output file (same as `output_path`).
    """
    from jinja2 import Environment, FileSystemLoader

    m = start_session()

    try:
        template_vars = make_readme_jinja_dict(
            m,
            dataset_path,
            base_model,
            prompt_file or "",
            name,
            hints,
            layout,
            adapter_type,
        )

        template_dir = os.path.dirname(os.path.abspath(__file__))
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template("README_TEMPLATE.jinja")

        readme_content = template.render(**template_vars)

        with open(output_path, "w") as f:
            f.write(readme_content)

        print(f"Generated README at {output_path}")
        return output_path
    finally:
        m.cleanup()
