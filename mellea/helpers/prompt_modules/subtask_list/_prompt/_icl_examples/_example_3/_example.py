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
    "1. Gather and analyze information about the operational status of Shanghai A Shipping Engineering Service Co., Ltd. - Variable: INFORMATION_GATHERING",
    '2. Write a formal paper titled "An Investigation Report on the Operational Status of Shanghai A Shipping Engineering Service Co., Ltd." with a steadily improving viewpoint. - Variable: FORMAL_PAPER',
    "3. Propose three refuting questions about the paper from the perspective of a freshman majoring in marine engineering. - Variable: REFUTING_QUESTIONS",
    "4. Select one of the three refuting questions to write a short essay opposing the views expressed in the paper based on the selected refuting question. - Variable: OPPOSING_ESSAY",
    "5. Compile the formal paper, the three refuting questions, and the short essay into a single cohesive output. - Variable: FINAL_OUTPUT",
]
