# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mypy overload-resolution checks for functional aact."""

from typing import Any, assert_type, cast

from mellea.core import (
    Backend,
    CBlock,
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
cblock_action: CBlock = cast(CBlock, None)
mot_action: ModelOutputThunk[str] = cast(ModelOutputThunk[str], None)


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
    strat = RejectionSamplingStrategy(loop_budget=2)
    r = await aact(action, ctx, backend, strategy=strat, return_sampling_results=True)
    assert_type(r, SamplingResult[str])


async def check_cblock_action() -> None:
    # The core widening of #356: aact accepts a raw CBlock action. A CBlock is
    # not generic, so S falls back to its Any default.
    r = await aact(cblock_action, ctx, backend, strategy=None, await_result=True)
    assert_type(r, tuple[ComputedModelOutputThunk[Any], Context])

    u = await aact(cblock_action, ctx, backend, strategy=None)
    assert_type(u, tuple[ModelOutputThunk[Any], Context])


async def check_mot_action() -> None:
    # aact also accepts a ModelOutputThunk action.
    r = await aact(mot_action, ctx, backend, strategy=None, await_result=True)
    assert_type(r, tuple[ComputedModelOutputThunk[Any], Context])

    u = await aact(mot_action, ctx, backend, strategy=None)
    assert_type(u, tuple[ModelOutputThunk[Any], Context])
