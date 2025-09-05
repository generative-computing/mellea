from pathlib import Path

from .._types import ICLExample, ICLExampleGroup
from ._example_1 import example as example_1
from ._example_2 import example as example_2
from ._example_3 import example as example_3
from ._example_4 import example as example_4
from ._example_5 import example as example_5

this_file_dir = Path(__file__).resolve().parent

with open(this_file_dir / "task_prompt.txt") as f:
    task_prompt = f.read().strip()

examples_items: list[ICLExample] = [
    example_1,
    example_2,
    example_3,
    example_4,
    example_5,
]

example_group: ICLExampleGroup = {
    "task_prompt": task_prompt,
    "examples_items": examples_items,
}
