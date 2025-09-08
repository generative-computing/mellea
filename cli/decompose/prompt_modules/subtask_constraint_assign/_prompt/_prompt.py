from collections.abc import Sequence
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from ._icl_examples import ICLExample, icl_examples as default_icl_examples

this_file_dir = Path(__file__).resolve().parent

environment = Environment(loader=FileSystemLoader(this_file_dir), autoescape=False)
system_template = environment.get_template("system_template.jinja2")
user_template = environment.get_template("user_template.jinja2")


def get_system_prompt(icl_examples: list[ICLExample] = default_icl_examples) -> str:
    return system_template.render(icl_examples=icl_examples).strip()


def get_user_prompt(
    execution_plan: list[str],
    constraint_list: Sequence[str],
    subtask_title: str,
    subtask_prompt: str,
) -> str:
    return user_template.render(
        execution_plan=execution_plan,
        constraint_list=constraint_list,
        subtask_title=subtask_title,
        subtask_prompt=subtask_prompt,
    ).strip()
