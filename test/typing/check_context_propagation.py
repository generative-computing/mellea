# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mypy checks for context-type propagation (issue #1522).

Mellea functions must return the *same* `Context` subtype they were given, not
a widened base `Context`. These `assert_type` checks fail under mypy until the
functional layer threads the input context subtype through to its return type.
"""

from typing import Any, assert_type, cast

from mellea.core import Backend, ComputedModelOutputThunk, ModelOutputThunk
from mellea.stdlib.components import Instruction, Message
from mellea.stdlib.context import ChatContext, SimpleContext
from mellea.stdlib.functional import (
    aact,
    achat,
    act,
    ainstruct,
    aquery,
    chat,
    instruct,
    query,
    transform,
)

backend = cast(Backend, None)
action: Instruction = cast(Instruction, None)
chat_ctx = cast(ChatContext, None)
simple_ctx = cast(SimpleContext, None)


# --- sync functions preserve the concrete context subtype ---


def check_act_chat_ctx() -> None:
    r = act(action, chat_ctx, backend)
    assert_type(r, tuple[ComputedModelOutputThunk[str], ChatContext])


def check_act_simple_ctx() -> None:
    r = act(action, simple_ctx, backend)
    assert_type(r, tuple[ComputedModelOutputThunk[str], SimpleContext])


def check_instruct_chat_ctx() -> None:
    r = instruct("test", chat_ctx, backend)
    assert_type(r, tuple[ComputedModelOutputThunk[str], ChatContext])


def check_query_chat_ctx() -> None:
    r = query(object(), "q", chat_ctx, backend)
    assert_type(r, tuple[ComputedModelOutputThunk[Any], ChatContext])


def check_transform_simple_ctx() -> None:
    r = transform(object(), "t", simple_ctx, backend)
    assert_type(r, tuple[ModelOutputThunk[Any] | Any, SimpleContext])


def check_chat_chat_ctx() -> None:
    r = chat("hi", chat_ctx, backend)
    assert_type(r, tuple[Message, ChatContext])


# --- async functions preserve the concrete context subtype ---


async def check_aact_chat_ctx() -> None:
    r = await aact(action, chat_ctx, backend, strategy=None, await_result=True)
    assert_type(r, tuple[ComputedModelOutputThunk[str], ChatContext])


async def check_aact_simple_ctx() -> None:
    r = await aact(action, simple_ctx, backend, strategy=None)
    assert_type(r, tuple[ModelOutputThunk[str], SimpleContext])


async def check_ainstruct_chat_ctx() -> None:
    r = await ainstruct("test", chat_ctx, backend, strategy=None, await_result=True)
    assert_type(r, tuple[ComputedModelOutputThunk[str], ChatContext])


async def check_aquery_simple_ctx() -> None:
    r = await aquery(object(), "q", simple_ctx, backend)
    assert_type(r, tuple[ModelOutputThunk[Any], SimpleContext])


async def check_achat_chat_ctx() -> None:
    r = await achat("hi", chat_ctx, backend)
    assert_type(r, tuple[Message, ChatContext])
