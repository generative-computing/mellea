from pathlib import Path

from .._types import ICLExample

this_file_dir = Path(__file__).resolve().parent

with open(this_file_dir / "subtask_prompt.txt") as f:
    subtask_prompt = f.read().strip()

example: ICLExample = {
    "execution_plan": [],
    "constraint_list": [],
    "subtask_title": "3. Rewrite the generated personas using the Markdown format and respecting the provided constraints.",
    "subtask_prompt": subtask_prompt.strip(),
    "assigned_constraints": [],
}

example["execution_plan"] = [
    "1. Receive and validate the input data, then extract relevant information to generate personas. - Variable: INPUT_VALIDATION",
    "2. Analyze the extracted information and generate at least two personas with the required properties. - Variable: PERSONA_GENERATION",
    "3. Rewrite the generated personas using the Markdown format and respecting the provided constraints. - Variable: FORMATTED_PERSONAS",
    "4. Extract only the generated personas asked on the task and answer the user without any additional explanation information. - Variable: TASK_ANSWER",
]

example["constraint_list"] = [
    "Your answers should not include harmful, unethical, racist, sexist, toxic, dangerous, or illegal content",
    "If a question does not make sense, or not factually coherent, explain to the user why, instead of just answering something incorrect",
    "You must always answer the user with markdown formatting",
    "The markdown formats you can use are the following: heading; link; table; list; code block; block quote; bold; italic",
    "When answering with code blocks, include the language",
    "All HTML tags must be enclosed in block quotes",
    "The personas must include the following properties: name; age; occupation; demographics; goals; behaviors; pain points; motivations",
    "The assistant must provide a comprehensive understanding of the target audience",
    "The assistant must analyze the user input data and generate at least 2 personas",
]

example["assigned_constraints"] = [
    "You must always answer the user with markdown formatting",
    "The markdown formats you can use are the following: heading; link; table; list; code block; block quote; bold; italic",
    "The personas must include the following properties: name; age; occupation; demographics; goals; behaviors; pain points; motivations",
]
