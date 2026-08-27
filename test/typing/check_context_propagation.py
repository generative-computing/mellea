# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mypy checks for context-type propagation (issue #1522).

Mellea functions must return the *same* `Context` subtype they were given, not
a widened base `Context`. These `assert_type` checks fail under mypy until the
functional layer threads the input context subtype through to its return type.
"""

from typing import Any, Self, assert_type, cast

from mellea.core import (
    Backend,
    ComputedModelOutputThunk,
    Context,
    ModelOutputThunk,
    Span,
)
from mellea.stdlib.components import Instruction, Message
from mellea.stdlib.context import ChatContext, SimpleContext
from mellea.stdlib.functional import (
    aact,
    achat,
    act,
    ainstruct,
    aquery,
    atransform,
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


# --- allow_context_type_change=True widens the returned context to `Context` ---
#
# The escape hatch permits the returned context to differ in subtype from the
# input, so the static return type must widen to the base `Context` rather than
# keep the input subtype (which would be an unsound narrowing).


def check_act_allow_change_widens() -> None:
    r = act(action, chat_ctx, backend, allow_context_type_change=True)
    assert_type(r, tuple[ComputedModelOutputThunk[str], Context])


def check_instruct_allow_change_widens() -> None:
    r = instruct("test", chat_ctx, backend, allow_context_type_change=True)
    assert_type(r, tuple[ComputedModelOutputThunk[str], Context])


def check_transform_allow_change_widens() -> None:
    r = transform(object(), "t", chat_ctx, backend, allow_context_type_change=True)
    assert_type(r, tuple[ModelOutputThunk[Any] | Any, Context])


async def check_aact_allow_change_widens() -> None:
    r = await aact(
        action,
        chat_ctx,
        backend,
        strategy=None,
        await_result=True,
        allow_context_type_change=True,
    )
    assert_type(r, tuple[ComputedModelOutputThunk[str], Context])


async def check_atransform_allow_change_widens() -> None:
    r = await atransform(
        object(), "t", chat_ctx, backend, allow_context_type_change=True
    )
    assert_type(r, tuple[ModelOutputThunk[Any] | Any, Context])


# --- a user-defined subclass keeps its own type through `.add()` ---
#
# `Context.add` narrows to `Self` on the built-in contexts, so a subclass that
# does not override `add` inherits a `-> Self` signature and `.add()` stays the
# subclass type rather than widening to `ChatContext`/`SimpleContext`.


class MyChatContext(ChatContext):
    """A user subclass that does not override `add`."""


class MySimpleContext(SimpleContext):
    """A user subclass that does not override `add`."""


def check_subclass_add_preserves_type() -> None:
    span = cast(Span, None)
    my_chat = cast(MyChatContext, None)
    my_simple = cast(MySimpleContext, None)
    assert_type(my_chat.add(span), MyChatContext)
    assert_type(my_simple.add(span), MySimpleContext)


def check_builtin_add_is_self() -> None:
    span = cast(Span, None)
    assert_type(chat_ctx.add(span), ChatContext)
    assert_type(simple_ctx.add(span), SimpleContext)
