from pathlib import Path

from .._types import ICLExample

this_file_dir = Path(__file__).resolve().parent

with open(this_file_dir / "task_prompt.txt") as f:
    task_prompt = f.read().strip()

example: ICLExample = {
    "task_prompt": task_prompt.strip(),
    "constraints_and_requirements": [],
}

example["constraints_and_requirements"] = [
    "The salutation should only include the recipient's first name at the start of the email's body",
    'Do not use the phrase "I hope this email finds you well," "I hope this email finds you doing well," or any similar variations',
    'You must generate the email in JSON structure with the following keys: "compelling_subject" and "email_body"',
    "Do not format your final answer with Markdown",
    "The output must be the JSON only, no additional comments",
]
