"""Scripts to decompose a prompt."""

import re

from mellea import MelleaSession
from mellea.stdlib.instruction import Instruction

from .prompts.metaprompts import (
    metaprompt_get_input_data__system,
    metaprompt_get_input_data__user,
    metaprompt_subtask_gen__system,
    metaprompt_subtask_gen__user,
    metaprompt_subtask_list__system,
    metaprompt_subtask_list__user,
)


def extract_between_tags(tag, text):
    """Extracts all occurrences between <tag>...</tag> from the given text.

    If extraction fails, prints the head and tail of the input text for debugging.
    """
    pattern = rf"<\s*{tag}\s*>(.*?)<\s*/\s*{tag}\s*>"
    matches = re.findall(pattern, text, flags=re.DOTALL)

    if matches:
        print(f"[SUCCESS] Found {len(matches)} <{tag}> tag(s).")
        return [m.strip() for m in matches]
    else:
        print(f"[FAIL] No <{tag}> tag found.")
        print("---- INPUT TEXT HEAD ----")
        print(text[:300])
        print("---- INPUT TEXT TAIL ----")
        print(text[-300:])
        return []


def get_step_and_var(line):
    """Parses a line into a step and variable.

    Parses a line like:
    '1. Research and brainstorm... - Variable: RESEARCH'
    and returns:
    ('Research and brainstorm...', '{{RESEARCH}}')

    If no variable is found, returns the step with None for variable.
    Skips lines that are None or empty.
    """
    if not line:
        print(f"[SKIP] Line is None or empty: {line}")
        return None

    pattern = r"^\d+\.\s*(.*?)\s*-\s*Variable:\s*([A-Z0-9_]+)$"
    match = re.match(pattern, line.strip())

    if match:
        step_text = match.group(1).strip()
        var_name = match.group(2).strip()
        if not step_text:
            print(f"[SKIP] Step text is empty in line: {line}")
            return None
        print(f"[SUCCESS] Parsed step: '{step_text}', Variable: {{ {var_name} }}")
        return step_text, f"{{{{{var_name}}}}}"
    else:
        # Try extracting step only (without variable)
        step_match = re.match(r"^\d+\.\s*(.*)", line.strip())
        if step_match and step_match.group(1).strip():
            step_text = step_match.group(1).strip()
            print(f"[SUCCESS] Parsed step without variable: '{step_text}'")
            return step_text, None
        print(f"[FAIL] Could not parse step from line: '{line}'")
        return None


def decompose_task(task: str, m_session: MelleaSession):
    """Decompose a given prompt into smaller tasks.

    Args:
        task: Input prompt to be decomposed.
        m_session (MelleaSession): Mellea session with a backend.
    """
    # Subtask list
    subtask_prompt = metaprompt_subtask_list__user.replace("{{TASK}}", task)
    instr = Instruction(
        description=subtask_prompt, prefix=metaprompt_subtask_list__system
    )
    subtask_list_output = m_session.backend.generate_from_context(
        action=instr, ctx=m_session.ctx
    ).value
    subtask_lines = (
        extract_between_tags("Final Subtask List", subtask_list_output)[0]
        .strip()
        .splitlines()
    )
    steps_and_vars = [
        result
        for line in subtask_lines
        if (result := get_step_and_var(line)) is not None
    ]

    # Input data
    input_prompt = metaprompt_get_input_data__user.replace("{{TASK}}", task)

    input_data_output = m_session.instruct(
        description=input_prompt, prefix=metaprompt_get_input_data__system
    ).value  # type: ignore

    input_data_str = extract_between_tags("User Input Data", input_data_output)[
        0
    ].strip()

    input_vars = (
        []
        if input_data_str == "N/A"
        else [x.strip() for x in input_data_str.split(",")]
    )

    print("Subtask List:")
    for line in subtask_lines:
        print(" -", line)

    print("\nInput Data Variables:", input_vars)

    return subtask_lines, steps_and_vars, input_vars


def build_subtasks(task, steps_and_vars, input_data_vars, m_session: MelleaSession):
    """Build subtasks based on the decomposed steps, available input variables, and available requirements.

    Returns a list of dictionaries with keys: step, var_tag, instruction, requirements
    """
    subtasks = []

    input_field_map = {f"{{{{{v}}}}}": "" for v in input_data_vars}
    input_keys = list(input_field_map.keys())

    for i, step_info in enumerate(steps_and_vars):
        if not step_info or len(step_info) != 2:
            print(f"[SKIP] Invalid step info at index {i}: {step_info}")
            continue

        step, var_tag = step_info
        if not step:
            print(f"[SKIP] Empty step text at index {i}")
            continue

        prev = steps_and_vars[:i]
        previous_steps = "\n".join(f"{s} - Variable: {v}" for s, v in prev if v)
        step_vars = ", ".join(input_keys + [v for _, v in prev if v])

        user_prompt = (
            metaprompt_subtask_gen__user.replace("{{TASK}}", task)
            .replace("{{STEP}}", step)
            .replace("{{PREVIOUS_STEPS}}", previous_steps)
            .replace("{{INPUT_DATA}}", step_vars)
        )

        output = m_session.instruct(
            description=user_prompt, prefix=metaprompt_subtask_gen__system
        ).value  # type: ignore

        # Extract instruction and requirements
        try:
            step_instr = extract_between_tags("Step Prompt Instruction", output)[
                0
            ].strip()
        except IndexError:
            print(
                f"[FAIL] No <Step Prompt Instruction> found in output for step '{step}'"
            )
            step_instr = "[ERROR: No instruction generated]"

        try:
            req_block = extract_between_tags("Requirements and Conditions", output)[0]
            reqs = [
                line.strip()[2:] if line.strip().startswith("- ") else line.strip()
                for line in req_block.splitlines()
                if line.strip() and line.strip() != "N/A"
            ]
        except IndexError:
            print(
                f"[WARN] No <Requirements and Conditions> found in output for step '{step}'"
            )
            reqs = []

        subtasks.append(
            {
                "step": step,
                "var_tag": var_tag,
                "instruction": step_instr,
                "requirements": reqs,
            }
        )

    return subtasks
