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
    'Finish your letter with the phrase, "In pursuit of a peaceful future."',
    "Add a postscript at the end of your letter",
]
