from textwrap import fill
from typing import Any

from mellea.core import Requirement, ValidationResult


# Just for printing stuff nicely...
def w(x: Any) -> str:
    return fill(str(x), width=120, replace_whitespace=False)


def req_print(rv_list: list[tuple[Requirement, ValidationResult]]) -> str:
    parts = [f"{bool(rv[1])}\t: {rv[0].description}" for rv in rv_list]
    return "\n".join(parts)
