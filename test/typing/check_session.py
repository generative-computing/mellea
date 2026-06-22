"""Mypy overload-resolution checks for MelleaSession methods."""

from typing import Any, assert_type, cast

from pydantic import BaseModel

from mellea.core import ComputedModelOutputThunk, ModelOutputThunk, SamplingResult
from mellea.stdlib.components import Instruction
from mellea.stdlib.session import MelleaSession

s = cast(MelleaSession, None)
action: Instruction = cast(Instruction, None)


class _M(BaseModel):
    value: str


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


def check_act_format() -> None:
    r = s.act(action, format=_M)
    assert_type(r, ComputedModelOutputThunk[_M])


def check_act_format_attributes() -> None:
    # Locks in what the format= overloads actually narrow at the attribute level.
    r = s.act(action, format=_M)

    # `parsed_repr` is the attribute the overloads narrow: it carries the generic
    # element type S, so with format=_M it resolves to `_M | None`.
    assert_type(r.parsed_repr, _M | None)

    # KNOWN LIMITATION: `.value` is typed `-> str` unconditionally on
    # ComputedModelOutputThunk (see mellea/core/base.py), so it does NOT narrow to
    # `_M` even though the thunk is parameterised `[_M]`. At runtime `.value` is the
    # raw string and `.parsed_repr` is also a plain str (Instruction._parse returns
    # str), so `parsed_repr.value` type-checks but AttributeErrors. Asserting
    # `assert_type(r.value, _M)` here would (correctly) fail mypy. Both the static
    # `.value` type and the runtime parsed_repr mismatch are pending the coordinated
    # thunk-generics / `.parsed` redesign (PR #1282).
    assert_type(r.value, str)


def check_instruct_format() -> None:
    r = s.instruct("test", format=_M)
    assert_type(r, ComputedModelOutputThunk[_M])


async def check_aact_format_computed() -> None:
    r = await s.aact(action, strategy=None, await_result=True, format=_M)
    assert_type(r, ComputedModelOutputThunk[_M])


async def check_aact_format_uncomputed() -> None:
    r = await s.aact(action, strategy=None, format=_M)
    assert_type(r, ModelOutputThunk[_M])


async def check_ainstruct_format_computed() -> None:
    r = await s.ainstruct("test", strategy=None, await_result=True, format=_M)
    assert_type(r, ComputedModelOutputThunk[_M])


async def check_ainstruct_format_uncomputed() -> None:
    r = await s.ainstruct("test", strategy=None, format=_M)
    assert_type(r, ModelOutputThunk[_M])
