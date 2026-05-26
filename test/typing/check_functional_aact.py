"""Mypy overload-resolution checks for functional aact."""

from typing import assert_type, cast

from mellea.core import (
    Backend,
    ComputedModelOutputThunk,
    Context,
    ModelOutputThunk,
    SamplingResult,
)
from mellea.stdlib.components import Instruction
from mellea.stdlib.functional import aact
from mellea.stdlib.sampling import RejectionSamplingStrategy

ctx = cast(Context, None)
backend = cast(Backend, None)
action: Instruction = cast(Instruction, None)


async def check_computed_await() -> None:
    r = await aact(action, ctx, backend, strategy=None, await_result=True)
    assert_type(r, tuple[ComputedModelOutputThunk[str], Context])


async def check_computed_strategy() -> None:
    strat = RejectionSamplingStrategy(loop_budget=2)
    r = await aact(action, ctx, backend, strategy=strat)
    assert_type(r, tuple[ComputedModelOutputThunk[str], Context])


async def check_uncomputed() -> None:
    r = await aact(action, ctx, backend, strategy=None)
    assert_type(r, tuple[ModelOutputThunk[str], Context])


async def check_sampling() -> None:
    r = await aact(action, ctx, backend, return_sampling_results=True)
    assert_type(r, SamplingResult[str])
