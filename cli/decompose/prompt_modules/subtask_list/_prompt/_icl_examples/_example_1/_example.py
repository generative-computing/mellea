from pathlib import Path

from .._types import ICLExample

this_file_dir = Path(__file__).resolve().parent

with open(this_file_dir / "task_prompt.txt") as f:
    task_prompt = f.read().strip()

with open(this_file_dir / "thinking_process.txt") as f:
    thinking_process = f.read()

example: ICLExample = {
    "task_prompt": task_prompt.strip(),
    "thinking_process": thinking_process.strip(),
    "subtask_list": [],
}

example["subtask_list"] = [
    "1. Receive and validate the input data, then extract relevant information to generate personas. - Variable: INPUT_VALIDATION",
    "2. Analyze the extracted information and generate at least two personas with the required properties. - Variable: PERSONA_GENERATION",
    "3. Rewrite the generated personas using the Markdown format and respecting the provided constraints. - Variable: FORMATTED_PERSONAS",
    "4. Extract only the generated personas asked on the task and answer the user without any additional explanation information. - Variable: TASK_ANSWER",
]
