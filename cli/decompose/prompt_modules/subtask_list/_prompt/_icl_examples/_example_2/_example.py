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
    "1. Research and brainstorm about the prospected company and create a detailed overview of their needs and interests. - Variable: RESEARCH",
    "2. Use the information previously researched, write an email to the prospected person introducing your product and explaining the value it holds for the prospected company. - Variable: EMAIL",
    "3. Write an email subject line that can increase the open rate. - Variable: SUBJECT",
    "4. Format the generated email in the JSON format provided by the task description and output only the JSON's text. - Variable: JSON",
]
