"""Mypy overload-resolution checks for functional ainstruct."""

from typing import assert_type, cast

from mellea.core import (
    Backend,
    ComputedModelOutputThunk,
    Context,
    ModelOutputThunk,
    SamplingResult,
)
from mellea.stdlib.functional import ainstruct

ctx = cast(Context, None)
backend = cast(Backend, None)


async def check_computed() -> None:
    r = await ainstruct("test", ctx, backend, strategy=None, await_result=True)
    assert_type(r, tuple[ComputedModelOutputThunk[str], Context])


async def check_uncomputed() -> None:
    r = await ainstruct("test", ctx, backend, strategy=None)
    assert_type(r, tuple[ModelOutputThunk[str], Context])


async def check_sampling() -> None:
    r = await ainstruct("test", ctx, backend, return_sampling_results=True)
    assert_type(r, SamplingResult[str])
