"""The mellea standard library of components, sessions, and sampling strategies.

This package provides the high-level building blocks for writing generative programs
with mellea. It contains ready-to-use ``Component`` types (``Instruction``,
``Message``, ``Document``, ``Intrinsic``, ``SimpleComponent``, and more), context
implementations (``ChatContext``, ``SimpleContext``), sampling strategies (rejection
sampling, budget forcing), session management via ``MelleaSession``, and the
``@mify`` decorator for turning ordinary Python objects into components. Import from
the sub-packages — ``mellea.stdlib.components``, ``mellea.stdlib.sampling``, and
``mellea.stdlib.session`` — for day-to-day use.

Streaming chunking strategies (for use with streaming validation) are available at
``mellea.stdlib.chunking`` and re-exported here for convenience.  The core streaming
orchestration primitive :func:`~mellea.stdlib.streaming.stream_with_chunking` and
its result type :class:`~mellea.stdlib.streaming.StreamChunkingResult` are also
re-exported here.
"""

from .chunking import ChunkingStrategy, ParagraphChunker, SentenceChunker, WordChunker
from .streaming import StreamChunkingResult, stream_with_chunking

__all__ = [
    "ChunkingStrategy",
    "ParagraphChunker",
    "SentenceChunker",
    "StreamChunkingResult",
    "WordChunker",
    "stream_with_chunking",
]
