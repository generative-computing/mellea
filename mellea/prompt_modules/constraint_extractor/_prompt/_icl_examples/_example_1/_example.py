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
