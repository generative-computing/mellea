# pytest: skip, huggingface, requires_heavy_ram, llm
# SKIP REASON: documentation only.

from cli.alora.readme_generator import generate_readme, make_readme_jinja_dict

generate_readme(
    dataset_path="stembolt_failure_dataset.jsonl",
    base_model="granite-4.0-micro",
    prompt_file=None,
    output_path="stembolts_model_readme.md",
    name="your-username/stembolts-alora",
    hints=None,
)
