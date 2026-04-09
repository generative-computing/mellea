"""Mypy overload-resolution checks for MelleaSession methods."""

from typing import Any, assert_type, cast

from mellea.core import ComputedModelOutputThunk, ModelOutputThunk, SamplingResult
from mellea.stdlib.components import Instruction
from mellea.stdlib.session import MelleaSession

s = cast(MelleaSession, None)
action: Instruction = cast(Instruction, None)


async def check_aact_computed() -> None:
    r = await s.aact(action, strategy=None, await_result=True)
    assert_type(r, ComputedModelOutputThunk[str])


async def check_aact_uncomputed() -> None:
    r = await s.aact(action, strategy=None)
    assert_type(r, ModelOutputThunk[str])


async def check_aact_sampling() -> None:
    r = await s.aact(action, return_sampling_results=True)
    assert_type(r, SamplingResult[str])


async def check_ainstruct_computed() -> None:
    r = await s.ainstruct("test", strategy=None, await_result=True)
    assert_type(r, ComputedModelOutputThunk[str])


async def check_ainstruct_uncomputed() -> None:
    r = await s.ainstruct("test", strategy=None)
    assert_type(r, ModelOutputThunk[str])


async def check_ainstruct_sampling() -> None:
    r = await s.ainstruct("test", return_sampling_results=True)
    assert_type(r, SamplingResult[str])


async def check_aquery_computed() -> None:
    r = await s.aquery("obj", "q", await_result=True)
    assert_type(r, ComputedModelOutputThunk[Any])


async def check_aquery_uncomputed() -> None:
    r = await s.aquery("obj", "q")
    assert_type(r, ModelOutputThunk[Any])


def check_query_sync() -> None:
    r = s.query("obj", "q")
    assert_type(r, ComputedModelOutputThunk[Any])
