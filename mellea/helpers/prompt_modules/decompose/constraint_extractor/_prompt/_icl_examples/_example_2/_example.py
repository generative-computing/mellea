from pathlib import Path

from .._types import ICLExample

this_file_dir = Path(__file__).resolve().parent

with open(this_file_dir / "task_prompt.txt") as f:
    task_prompt = f.read().strip()

example: ICLExample = {
    "task_prompt": task_prompt.strip(),
    "constraints_and_conditions": [],
}

example["constraints_and_conditions"] = [
    "Emphasize the responsibilities and support offered to survivors of crime",
    "Ensure the word 'assistance' appears less than 4 times",
    "Wrap the entire response with double quotation marks",
]
