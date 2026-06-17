"""Mypy overload-resolution checks for functional ainstruct."""

from typing import assert_type, cast

from pydantic import BaseModel

from mellea.core import (
    Backend,
    ComputedModelOutputThunk,
    Context,
    ModelOutputThunk,
    SamplingResult,
)
from mellea.stdlib.functional import ainstruct
from mellea.stdlib.sampling import RejectionSamplingStrategy

ctx = cast(Context, None)
backend = cast(Backend, None)


class _M(BaseModel):
    value: str


async def check_computed() -> None:
    r = await ainstruct("test", ctx, backend, strategy=None, await_result=True)
    assert_type(r, tuple[ComputedModelOutputThunk[str], Context])


async def check_uncomputed() -> None:
    r = await ainstruct("test", ctx, backend, strategy=None)
    assert_type(r, tuple[ModelOutputThunk[str], Context])


async def check_sampling() -> None:
    r = await ainstruct("test", ctx, backend, return_sampling_results=True)
    assert_type(r, SamplingResult[str])


async def check_format_computed_await() -> None:
    r = await ainstruct(
        "test", ctx, backend, strategy=None, await_result=True, format=_M
    )
    assert_type(r, tuple[ComputedModelOutputThunk[_M], Context])


async def check_format_computed_strategy() -> None:
    strat = RejectionSamplingStrategy(loop_budget=2)
    r = await ainstruct("test", ctx, backend, strategy=strat, format=_M)
    assert_type(r, tuple[ComputedModelOutputThunk[_M], Context])


async def check_format_uncomputed() -> None:
    r = await ainstruct("test", ctx, backend, strategy=None, format=_M)
    assert_type(r, tuple[ModelOutputThunk[_M], Context])
