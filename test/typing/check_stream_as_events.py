# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mypy overload-resolution checks for `stream(as_events=...)`."""

from typing import assert_type, cast

from mellea.core import Backend, CBlock, Context
from mellea.stdlib.streaming import EventStreamer, Streamer, stream

ctx = cast(Context, None)
backend = cast(Backend, None)
action: CBlock = cast(CBlock, None)


async def check_default_returns_streamer() -> None:
    s = await stream(action, backend, ctx)
    assert_type(s, Streamer)


async def check_as_events_false_returns_streamer() -> None:
    s = await stream(action, backend, ctx, as_events=False)
    assert_type(s, Streamer)


async def check_as_events_true_returns_event_streamer() -> None:
    s = await stream(action, backend, ctx, as_events=True)
    assert_type(s, EventStreamer)


async def check_as_events_dynamic_bool_returns_union() -> None:
    flag: bool = True
    s = await stream(action, backend, ctx, as_events=flag)
    assert_type(s, Streamer | EventStreamer)
