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
from mellea.core import Backend, Context
from mellea.stdlib.context import ChatContext, SimpleContext

backend = cast(Backend, None)
chat_ctx = cast(ChatContext, None)
simple_ctx = cast(SimpleContext, None)
runtime_flag = cast(bool, None)


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


# --- a runtime bool for allow_context_type_change widens to Context ---
#
# A `Literal[True]` opts into a deliberate type change, so the session can no
# longer promise its input subtype and widens to `MelleaSession[Context]`. A
# runtime `bool` variable (matching neither literal) must resolve to the same
# widened overload rather than falling through to `Any`.


def check_ctor_literal_true_widens() -> None:
    session = MelleaSession(backend, chat_ctx, allow_context_type_change=True)
    assert_type(session, MelleaSession[Context])
    assert_type(session.ctx, Context)


def check_ctor_runtime_bool_widens() -> None:
    session = MelleaSession(backend, chat_ctx, allow_context_type_change=runtime_flag)
    assert_type(session, MelleaSession[Context])
    assert_type(session.ctx, Context)


def check_ctor_literal_false_preserves_type() -> None:
    session = MelleaSession(backend, chat_ctx, allow_context_type_change=False)
    assert_type(session, MelleaSession[ChatContext])
    assert_type(session.ctx, ChatContext)


def check_start_session_runtime_bool_widens() -> None:
    session = start_session(ctx=chat_ctx, allow_context_type_change=runtime_flag)
    assert_type(session, MelleaSession[Context])
    assert_type(session.ctx, Context)


# --- start_session context_type / default overloads also widen when switching ---
#
# A session built from `context_type=` (or the default context) that permits
# switching cannot statically promise the concrete subtype either, so it must
# widen to `MelleaSession[Context]` just like the `ctx=`-supplied forms above.


def check_start_session_context_type_chat_literal_true_widens() -> None:
    session = start_session(context_type="chat", allow_context_type_change=True)
    assert_type(session, MelleaSession[Context])
    assert_type(session.ctx, Context)


def check_start_session_context_type_chat_runtime_bool_widens() -> None:
    session = start_session(context_type="chat", allow_context_type_change=runtime_flag)
    assert_type(session, MelleaSession[Context])
    assert_type(session.ctx, Context)


def check_start_session_context_type_chat_literal_false_preserves() -> None:
    session = start_session(context_type="chat", allow_context_type_change=False)
    assert_type(session, MelleaSession[ChatContext])
    assert_type(session.ctx, ChatContext)


def check_start_session_context_type_simple_literal_true_widens() -> None:
    session = start_session(context_type="simple", allow_context_type_change=True)
    assert_type(session, MelleaSession[Context])
    assert_type(session.ctx, Context)


def check_start_session_default_runtime_bool_widens() -> None:
    session = start_session(allow_context_type_change=runtime_flag)
    assert_type(session, MelleaSession[Context])
    assert_type(session.ctx, Context)


def check_start_session_default_literal_false_preserves_simple() -> None:
    session = start_session(allow_context_type_change=False)
    assert_type(session, MelleaSession[SimpleContext])
    assert_type(session.ctx, SimpleContext)


# --- clone preserves the context type parameter ---


def check_clone_preserves_type() -> None:
    session = MelleaSession(backend, chat_ctx)
    assert_type(session.clone(), MelleaSession[ChatContext])
