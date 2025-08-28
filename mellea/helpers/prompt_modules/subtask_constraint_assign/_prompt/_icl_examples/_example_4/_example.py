from pathlib import Path

from .._types import ICLExample

this_file_dir = Path(__file__).resolve().parent

with open(this_file_dir / "subtask_prompt.txt") as f:
    subtask_prompt = f.read().strip()

example: ICLExample = {
    "execution_plan": [],
    "constraint_list": [],
    "subtask_title": "1. Analyze and understand the poetic content provided in the task prompt.",
    "subtask_prompt": subtask_prompt.strip(),
    "assigned_constraints": [],
}

example["execution_plan"] = [
    "1. Analyze and understand the poetic content provided in the task prompt. - Variable: CONTENT_ANALYSIS",
    "2. Draft an appreciation from the perspective of a literature-loving ninth-grade student, ensuring it is concise, beautiful, and positive, and does not exceed 500 characters. - Variable: STUDENT_APPRECIATION",
    "3. Draft an appreciation from the perspective of a weather-beaten retired old teacher, ensuring it is deep, philosophical, and carries negative and pessimistic emotions, and does not exceed 500 characters. - Variable: TEACHER_APPRECIATION",
    "4. Compile both appreciations into a single output that meets the requirements of the task prompt. - Variable: FINAL_OUTPUT",
]

example["constraint_list"] = [
    "Write an appreciation for the above content in no more than 500 characters",
    "Use concise and beautiful language, and positive emotions",
    "Write another appreciation also not exceeding 500 characters",
    "Use deep and philosophical language, and negative and pessimistic emotions",
]

example["assigned_constraints"] = ["N/A"]
