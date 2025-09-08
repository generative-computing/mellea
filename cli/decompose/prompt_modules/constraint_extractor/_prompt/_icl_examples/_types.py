from typing import TypedDict


class ICLExample(TypedDict):
    task_prompt: str
    constraints_and_conditions: list[str]
