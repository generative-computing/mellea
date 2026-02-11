import json
import os

from pydantic import BaseModel

from mellea import start_session
from mellea.stdlib.session import MelleaSession


class ReadmeTemplateVars(BaseModel):
    adapter_name: str
    high_level_description: str
    dataset_description: str
    userid: str
    instrinsic_name: str
    instrinsic_name_camelcase: str
    intrinsic_name: str
    arglist: str


def make_readme_jinja_dict(
    m: MelleaSession,
    dataset_path: str,
    base_model: str,
    prompt_file: str,
) -> dict[str, str]:
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
            f"(e.g., '{', '.join(f'{k}: str' for k in item_keys)}')."
        )
    else:
        arglist_hint = (
            "The input data items are plain strings. Choose a descriptive single "
            "argument name with a str type hint (e.g., 'description: str' or "
            "'notes: str'). Pick a name that reflects what the input data represents."
        )

    # Load prompt file content if provided.
    prompt_content = ""
    if prompt_file:
        with open(prompt_file) as f:
            prompt_content = f.read()

    # Build the LLM prompt.
    sample_text = "\n".join(json.dumps(s) for s in samples)

    prompt = f"""You are generating metadata for a README file for a machine learning intrinsic adapter trained on the dataset below.

Here are the first few samples from the training dataset (JSONL format):
{sample_text}

Base model: {base_model}
{("Prompt configuration: " + prompt_content) if prompt_content else ""}

{arglist_hint}

Generate appropriate values for each field:
- adapter_name: A human-readable title for this adapter (e.g., "Stembolt Failure Analysis")
- high_level_description: A 2-3 sentence description of what this intrinsic adapter does based on the training data
- dataset_description: A brief description of the training dataset contents and format
- userid: Set this to "your-username" as a placeholder for the HuggingFace user ID
- instrinsic_name: A short snake_case identifier for this intrinsic (e.g., "stembolts"). No hyphens.
- instrinsic_name_camelcase: The CamelCase version of instrinsic_name (e.g., "Stembolt"). No underscores.
- intrinsic_name: Same value as instrinsic_name
- arglist: The Python function argument list with type hints based on the input data structure. This will be used as function parameters."""

    result = m.chat(prompt, format=ReadmeTemplateVars)
    vars_dict = ReadmeTemplateVars.model_validate_json(result.content).model_dump()

    return vars_dict


def generate_readme(
    dataset_path: str,
    base_model: str,
    prompt_file: str | None,
    output_path: str,
) -> str:
    """Generate an INTRINSIC_README.md file from the dataset and template.

    Creates a MelleaSession, uses the LLM to generate template variables,
    renders the Jinja template, and writes the result to output_path.
    """
    from jinja2 import Environment, FileSystemLoader

    m = start_session()

    try:
        template_vars = make_readme_jinja_dict(
            m, dataset_path, base_model, prompt_file or ""
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
