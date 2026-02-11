import ast
import json
import os
from typing import Any

from pydantic import BaseModel

from mellea import start_session
from mellea.stdlib.requirements.requirement import check, req, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy
from mellea.stdlib.session import MelleaSession


class ReadmeTemplateVars(BaseModel):
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
) -> dict[str, Any]:
    """Generate all template variables for the intrinsic README using an LLM.

    Loads the first five lines of the JSONL dataset, determines the input structure,
    and uses m.instruct with deterministic requirements and rejection sampling to
    generate README template variables.
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

    description = f"""You are generating metadata for a README file for a machine learning intrinsic adapter trained on the dataset below.{hints_text}

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

    vars_dict["base_model"] = base_model

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
