"""Scripts to ingest subtasks and compiles into a .py file."""

import ast
import re
from typing import Optional

from mellea import MelleaSession


def compile_task(result_data, m_session: MelleaSession):
    """Compiles all subtasks in sequence, resolving intermediate variables,and populates result_data with generated answers."""
    input_data_fields = {f"{{{{{v}}}}}": "" for v in result_data.get("input_data", [])}

    for i, subtask in enumerate(result_data["subtask_data"]):
        subtask_answer = compile_subtask(subtask, input_data_fields, m_session)

        # Update state
        var_tag = subtask.get("var_tag")
        if var_tag:
            input_data_fields[var_tag] = subtask_answer  # type: ignore

        subtask["subtask_answer"] = subtask_answer

        if i == len(result_data["subtask_data"]) - 1:
            result_data["generated_final_answer"] = subtask_answer

    return result_data, "Ok"


def compile_subtask(subtask, input_data_fields, m_session: MelleaSession) -> str | None:
    """Compiles one subtask with provided variable inputs and return generated result."""
    file_contents = "import mellea\n\nm = mellea.start_session()\n\n"
    reqs_and_conditions = "\n".join(subtask.get("requirements", []))

    raw_prompt = (
        subtask["instruction"]
        + "\n\nWhen writing your answer, Follow the requirements and conditions below:\n"
        + reqs_and_conditions
    )

    if input_data_fields:
        pattern = "|".join(re.escape(k) for k in input_data_fields)
        populated_prompt = re.sub(
            pattern, lambda m: input_data_fields[m.group(0)], raw_prompt
        )
    else:
        populated_prompt = raw_prompt

    # Define roles
    sys_prompt = (
        subtask["step"][3:] if subtask["step"].startswith("1.") else "Assistant"
    )

    try:
        file_contents += f'result = m.instrt(description="{populated_prompt}", prefix="{sys_prompt}")\nreturn result.value'
        ast.parse(file_contents)  # This will throw an exception. Which one?
        return file_contents
    except Exception as e:
        print(f"[ERROR] Failed to execute subtask '{subtask['step']}': {e}")
        return None
