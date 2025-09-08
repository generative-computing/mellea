from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from ._icl_example_groups import (
    ICLExampleGroup,
    icl_example_groups as default_icl_example_groups,
)

this_file_dir = Path(__file__).resolve().parent

environment = Environment(loader=FileSystemLoader(this_file_dir), autoescape=False)
system_template = environment.get_template("system_template.jinja2")
user_template = environment.get_template("user_template.jinja2")


def get_system_prompt(
    icl_example_groups: list[ICLExampleGroup] = default_icl_example_groups,
    user_input_variables_exists: bool = False,
) -> str:
    return system_template.render(
        icl_example_groups=icl_example_groups,
        user_input_variables_exists=user_input_variables_exists,
    ).strip()


def get_user_prompt(
    task_prompt: str,
    execution_plan: list[str],
    available_content_variables: list[str],
    target_subtask: str,
) -> str:
    return user_template.render(
        task_prompt=task_prompt,
        execution_plan=execution_plan,
        available_content_variables=available_content_variables,
        target_subtask=target_subtask,
    ).strip()
