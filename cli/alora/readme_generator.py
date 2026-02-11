import json
import os
from typing import Any

from pydantic import BaseModel

from mellea import start_session
from mellea.stdlib.session import MelleaSession


class ReadmeTemplateVars(BaseModel):
    high_level_description: str
    dataset_description: str
    userid: str
    intrinsic_name: str
    intrinsic_name_camelcase: str
    arglist: str
    arglist_without_type_annotations: str


def make_readme_jinja_dict(
    m: MelleaSession,
    dataset_path: str,
    base_model: str,
    prompt_file: str,
    name: str,
    hints: str | None,
) -> dict[str, Any]:
    """Generate all template variables for the intrinsic README using an LLM.

    Loads the first five lines of the JSONL dataset, determines the input structure,
    and uses m.chat with constrained decoding to generate README template variables.
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

    # Build the LLM prompt.
    sample_text = "\n".join(json.dumps(s) for s in samples)

    prompt = f"""You are generating metadata for a README file for a machine learning intrinsic adapter trained on the dataset below.

{"Here are some additional details about the domain:\n" + hints + "\n\n" if hints is not None else ""}
Here are the first few samples from the training dataset (JSONL format):
{sample_text}

Base model: {base_model}
{("Prompt configuration: " + prompt_content) if prompt_content else ""}

{arglist_hint}

Generate appropriate values for each field:
- high_level_description: A 2-3 sentence description of what this intrinsic adapter does based on the training data
- dataset_description: A brief description of the training dataset contents and format
- userid: Set this to "your-username" as a placeholder for the HuggingFace user ID. MUST be {name.split("/")[0]}
- intrinsic_name: A short snake_case identifier for this intrinsic (e.g., "stembolts"). No hyphens. MUST be {name.split("/")[1]}
- intrinsic_name_camelcase: The CamelCase version of intrinsic_name (e.g., "Stembolt"). No underscores.
- arglist: The Python function argument list with type hints based on the input data structure. This will be used as function parameters.
- arglist_without_type_annotations: The arglist without any type annotations. Should NOT start or end with parens."""

    result = m.chat(prompt, format=ReadmeTemplateVars)
    vars_dict = ReadmeTemplateVars.model_validate_json(result.content).model_dump()

    # TODO this should be a requirement
    if (
        vars_dict["intrinsic_name_camelcase"].lower()
        != vars_dict["intrinsic_name"].replace("_", "").replace("-", "").lower()
    ):
        print(
            f"Fixing {vars_dict['intrinsic_name_camelcase']} to be the camelcase version of {vars_dict['intrinsic_name']}"
        )
        snake = vars_dict["intrinsic_name"]
        camel = ""
        while i < len(snake):
            if i == 0:
                camel += snake[i].upper()
                continue
            if snake[i] == "-" or snake[i] == "_" or i == 0:
                camel += snake[i + 1].upper() if len(snake) < i + 1 else ""
                i += 2
                continue
        vars_dict["intrinsic_name_camelcase"] = camel

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

    return vars_dict


def generate_readme(
    dataset_path: str,
    base_model: str,
    prompt_file: str | None,
    output_path: str,
    name: str,
    hints: str | None,
) -> str:
    """Generate an INTRINSIC_README.md file from the dataset and template.

    Creates a MelleaSession, uses the LLM to generate template variables,
    renders the Jinja template, and writes the result to output_path.
    """
    from jinja2 import Environment, FileSystemLoader

    m = start_session()

    try:
        template_vars = make_readme_jinja_dict(
            m, dataset_path, base_model, prompt_file or "", name, hints
        )

        # TODO this should be a requirement.
        assert template_vars["intrinsic_name"] == name.split("/")[1], (
            f"intrinsic_name {template_vars['intrinsic_name']} should be the same as {name.split('/')[1]}. TODO-rf: we need to robustify this generator. If you are seeing this message, just try running the same command again. Sorry about that, this feature is still beta."
        )

        # TODO this should be a requirement
        # TODO the actual requirement should be that both of these things parse as tuples and arg lists respectively in python
        # TODO the full python code block parsing should also be a requirement
        # TODO these should be 'check' requirements, ie not included in thep rompt.
        if template_vars["arglist"].startswith("(") and template_vars[
            "arglist"
        ].endswith(")"):
            template_vars["arglist"] = template_vars["arglist"][1:-1]
        if template_vars["arglist_without_type_annotations"].startswith(
            "("
        ) and template_vars["arglist_without_type_annotations"].endswith(")"):
            template_vars["arglist_without_type_annotations"] = template_vars[
                "arglist_without_type_annotations"
            ][1:-1]

        template_dir = os.path.dirname(os.path.abspath(__file__))
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template("README_TEMPLATE.jinja")

        readme_content = template.render(**template_vars)

        with open(output_path, "w") as f:
            f.write(readme_content)

        print(f"Generated INTRINSIC_README.md at {output_path}")
        return output_path
    finally:
        m.cleanup()
