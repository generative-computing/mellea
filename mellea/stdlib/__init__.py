# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The mellea standard library of components, sessions, and sampling strategies.

This package provides the high-level building blocks for writing generative programs
with mellea. It contains ready-to-use `Component` types (`Instruction`,
`Message`, `Document`, `Intrinsic`, `SimpleComponent`, and more), context
implementations (`ChatContext`, `SimpleContext`), sampling strategies (rejection
sampling, budget forcing), session management via `MelleaSession`, and the
`@mify` decorator for turning ordinary Python objects into components. Import from
the sub-packages — `mellea.stdlib.components`, `mellea.stdlib.sampling`, and
`mellea.stdlib.session` — for day-to-day use.

Streaming chunking strategies (for use with streaming validation) are available at
`mellea.stdlib.chunking` and re-exported here for convenience, alongside the
`Chunker` that drives them over a stream.  The core streaming primitive `stream()`
and its async-iterable handle `Streamer` are also re-exported here, alongside the
full `StreamEvent` vocabulary for typed event observation.

Low-level primitives for tool execution are available in `mellea.stdlib.functional`:
`call_tools` and `acall_tools` for executing model-requested tool calls with full
hook and telemetry support. Higher-level APIs like `act()`, `instruct()`, or
`chat()` generate tool calls but do not execute them—use `call_tools()` to run
the generated tools. These primitives are rarely needed outside custom agentic loops.
"""

from .chunking import (
    Chunker,
    ChunkingStrategy,
    ParagraphChunking,
    SentenceChunking,
    WordChunking,
)
from .streaming import (
    ChunkEvent,
    CompletedEvent,
    ErrorEvent,
    EventStreamer,
    FullValidationEvent,
    QuickCheckEvent,
    Streamer,
    StreamEvent,
    StreamingDoneEvent,
    stream,
)

__all__ = [
    "ChunkEvent",
    "Chunker",
    "ChunkingStrategy",
    "CompletedEvent",
    "ErrorEvent",
    "EventStreamer",
    "FullValidationEvent",
    "ParagraphChunking",
    "QuickCheckEvent",
    "SentenceChunking",
    "StreamEvent",
    "Streamer",
    "StreamingDoneEvent",
    "WordChunking",
    "stream",
]
