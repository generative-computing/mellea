"""Mypy checks that `ComputedModelOutputThunk.parsed` tracks the type parameter.

`.parsed` is typed `S | None`, so a thunk parameterized with a Pydantic model
(`ComputedModelOutputThunk[MyModel]`) exposes `.parsed` as `MyModel | None` —
callers need no `cast()`. The `format=` overloads in `session.py` /
`functional.py` bind `S` to the format model (companion issue #1274), at which
point these checks hold end-to-end from the call site.
"""

from typing import assert_type, cast

import pydantic

from mellea.core import ComputedModelOutputThunk


class _Person(pydantic.BaseModel):
    name: str


def check_parsed_tracks_format_model() -> None:
    thunk = cast(ComputedModelOutputThunk[_Person], None)
    assert_type(thunk.parsed, _Person | None)


def check_parsed_is_str_for_str_thunk() -> None:
    thunk = cast(ComputedModelOutputThunk[str], None)
    assert_type(thunk.parsed, str | None)
