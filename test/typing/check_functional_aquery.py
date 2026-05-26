"""Mypy overload-resolution checks for functional aquery and query."""

from typing import Any, assert_type, cast

from mellea.core import Backend, ComputedModelOutputThunk, Context, ModelOutputThunk
from mellea.stdlib.functional import aquery, query

ctx = cast(Context, None)
backend = cast(Backend, None)


async def check_computed() -> None:
    r = await aquery("obj", "q", ctx, backend, await_result=True)
    assert_type(r, tuple[ComputedModelOutputThunk[Any], Context])


async def check_uncomputed() -> None:
    r = await aquery("obj", "q", ctx, backend)
    assert_type(r, tuple[ModelOutputThunk[Any], Context])


def check_query_sync() -> None:
    r = query("obj", "q", ctx, backend)
    assert_type(r, tuple[ComputedModelOutputThunk[Any], Context])
