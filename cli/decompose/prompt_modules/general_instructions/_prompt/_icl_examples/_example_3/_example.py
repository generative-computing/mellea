from pathlib import Path

from .._types import ICLExample

this_file_dir = Path(__file__).resolve().parent

with open(this_file_dir / "task_prompt.txt") as f:
    task_prompt = f.read().strip()

with open(this_file_dir / "general_instructions.txt") as f:
    general_instructions = f.read().strip()

example: ICLExample = {
    "task_prompt": task_prompt.strip(),
    "general_instructions": general_instructions.strip(),
}
