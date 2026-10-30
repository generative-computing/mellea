# pytest: skip, huggingface, e2e
# SKIP REASON: documentation only.
"""Generating a Hugging Face README for a trained adapter programmatically.

Demonstrates calling `generate_readme()` directly, which is the same code path as
the `m alora add-readme` command. A `MelleaSession` analyzes the training dataset
and writes a model README (description, data examples, integration code) to
`stembolts_model_readme.md`. Replace the placeholder `name` with your own
`<hf-username>/<adapter-name>`; `hints` and `prompt_file` are optional.

Prerequisites:
    uv sync --extra cli
    ollama serve
    ollama pull granite4.2:3b

Usage:
    cd docs/examples/aLora && uv run python example_readme_generator.py
"""

from cli.alora.readme_generator import generate_readme, make_readme_jinja_dict

if __name__ == "__main__":
    generate_readme(
        dataset_path="stembolt_failure_dataset.jsonl",
        base_model="granite-4.1-3b",
        prompt_file=None,
        output_path="stembolts_model_readme.md",
        name="your-username/stembolts-alora",
        hints=None,
    )
