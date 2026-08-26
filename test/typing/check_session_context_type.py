# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mypy checks for MelleaSession context-type parameterization (issue #1522).

`MelleaSession` is generic in its context type, so `session.ctx` must narrow to
the concrete `Context` subtype the session was built with rather than widening
to the base `Context`. These `assert_type` checks verify that the constructor
overloads, the `start_session` overloads, and `clone()` all preserve the
parameter. Verification is via `uv run mypy .`; the functions never execute.
"""

from typing import assert_type, cast

from mellea import MelleaSession, start_session
from mellea.core import Backend
from mellea.stdlib.context import ChatContext, SimpleContext

backend = cast(Backend, None)
chat_ctx = cast(ChatContext, None)
simple_ctx = cast(SimpleContext, None)


# --- constructor infers the context type parameter ---


def check_ctor_chat_ctx() -> None:
    session = MelleaSession(backend, chat_ctx)
    assert_type(session, MelleaSession[ChatContext])
    assert_type(session.ctx, ChatContext)


def check_ctor_simple_ctx() -> None:
    session = MelleaSession(backend, simple_ctx)
    assert_type(session, MelleaSession[SimpleContext])
    assert_type(session.ctx, SimpleContext)


def check_ctor_no_ctx_defaults_simple() -> None:
    session = MelleaSession(backend)
    assert_type(session, MelleaSession[SimpleContext])
    assert_type(session.ctx, SimpleContext)


# --- start_session infers the context type parameter ---


def check_start_session_context_type_chat() -> None:
    session = start_session(context_type="chat")
    assert_type(session, MelleaSession[ChatContext])
    assert_type(session.ctx, ChatContext)


def check_start_session_context_type_simple() -> None:
    session = start_session(context_type="simple")
    assert_type(session, MelleaSession[SimpleContext])
    assert_type(session.ctx, SimpleContext)


def check_start_session_default_is_simple() -> None:
    session = start_session()
    assert_type(session, MelleaSession[SimpleContext])
    assert_type(session.ctx, SimpleContext)


def check_start_session_explicit_ctx() -> None:
    session = start_session(ctx=chat_ctx)
    assert_type(session, MelleaSession[ChatContext])
    assert_type(session.ctx, ChatContext)


# --- clone preserves the context type parameter ---


def check_clone_preserves_type() -> None:
    session = MelleaSession(backend, chat_ctx)
    assert_type(session.clone(), MelleaSession[ChatContext])
