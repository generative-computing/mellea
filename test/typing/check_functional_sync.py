"""Mypy overload-resolution checks for sync functional APIs (act, instruct)."""

from typing import assert_type, cast

from pydantic import BaseModel

from mellea.core import Backend, ComputedModelOutputThunk, Context
from mellea.stdlib.components import Instruction
from mellea.stdlib.functional import act, instruct
from mellea.stdlib.session import MelleaSession

ctx = cast(Context, None)
backend = cast(Backend, None)
action: Instruction = cast(Instruction, None)
s = cast(MelleaSession, None)


class _M(BaseModel):
    value: str


def check_act_sync() -> None:
    r = act(action, ctx, backend)
    assert_type(r, tuple[ComputedModelOutputThunk[str], Context])


def check_instruct_sync() -> None:
    r = instruct("test", ctx, backend)
    assert_type(r, tuple[ComputedModelOutputThunk[str], Context])


def check_session_act_sync() -> None:
    r = s.act(action)
    assert_type(r, ComputedModelOutputThunk[str])


def check_session_instruct_sync() -> None:
    r = s.instruct("test")
    assert_type(r, ComputedModelOutputThunk[str])


def check_act_format() -> None:
    r = act(action, ctx, backend, format=_M)
    assert_type(r, tuple[ComputedModelOutputThunk[_M], Context])


def check_instruct_format() -> None:
    r = instruct("test", ctx, backend, format=_M)
    assert_type(r, tuple[ComputedModelOutputThunk[_M], Context])
