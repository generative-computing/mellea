"""Context compaction strategies for the ReACT framework.

Provides modular, callable strategy objects to compact a ``ChatContext`` that
has grown too large during a react loop.  Three strategies are available:

- ``ClearAll`` — discard the entire conversation body, keeping only the prefix
  (everything up to and including the ``ReactInitiator``).
- ``KeepLastN`` — keep the prefix plus the *n* most recent body components.
- ``LLMSummarize`` — ask the backend to summarize old body components into a
  single ``Message``, then keep the last *n* body components verbatim.

All strategies preserve the **prefix** (every component up to and including the
first ``ReactInitiator``) so the model retains its goal and tool definitions.

Example::

    from mellea.stdlib.frameworks.react_compaction import KeepLastN
    from mellea.stdlib.frameworks.react import react

    # Compact once the most recent model call reports > 8000 prompt+completion tokens.
    await react(
        goal="...",
        context=ChatContext(),
        backend=m.backend,
        tools=[search_tool],
        compaction=KeepLastN(keep_n=5, threshold=8000),
    )
"""

from __future__ import annotations

import abc

from mellea.core.backend import Backend
from mellea.core.base import CBlock, Component, ModelOutputThunk
from mellea.core.utils import MelleaLogger
from mellea.stdlib.components.chat import Message, ToolMessage
from mellea.stdlib.components.react import ReactInitiator
from mellea.stdlib.context import ChatContext

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def rebuild_chat_context(
    components: list[Component | CBlock], *, window_size: int | None = None
) -> ChatContext:
    """Build a fresh ``ChatContext`` from an ordered list of components.

    Args:
        components: Components to add, in chronological order.
        window_size: Optional sliding-window size for the new context.

    Returns:
        A new ``ChatContext`` containing all *components*.
    """
    ctx = ChatContext(window_size=window_size)
    for c in components:
        ctx = ctx.add(c)
    return ctx


def _find_prefix_end(components: list[Component | CBlock]) -> int:
    """Return the index *after* the first ``ReactInitiator``.

    Everything in ``components[:idx]`` is the prefix that must be preserved by
    every compaction strategy.  Returns 0 when no ``ReactInitiator`` is found.
    """
    for i, c in enumerate(components):
        if isinstance(c, ReactInitiator):
            return i + 1
    return 0


def _last_usage_tokens(context: ChatContext) -> int | None:
    """Return ``total_tokens`` from the most recent ``ModelOutputThunk`` with usage.

    Walks *context* back-to-front looking for a ``ModelOutputThunk`` whose
    ``usage`` dict has been populated by a backend's ``post_processing``.
    Falls back to ``prompt_tokens + completion_tokens`` when ``total_tokens``
    is missing.  Returns ``None`` if no usable token count can be recovered —
    typically the case before the first model call completes.
    """
    for c in reversed(context.as_list()):
        if isinstance(c, ModelOutputThunk) and c.generation.usage is not None:
            total = c.generation.usage.get("total_tokens")
            if total is None:
                pt = c.generation.usage.get("prompt_tokens") or 0
                ct = c.generation.usage.get("completion_tokens") or 0
                total = pt + ct
            return total if total and total > 0 else None
    return None


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class CompactionStrategy(abc.ABC):
    """Abstract base class for context compaction strategies.

    Each strategy carries a ``threshold`` — the token count above which
    compaction should fire.  The :meth:`should_compact` helper reads the
    most recent ``ModelOutputThunk.usage`` populated by the backend and
    compares its total token count to ``threshold``.

    Because ``usage`` is recorded when a model call completes, the measured
    token count reflects the context as of the *previous* turn — any
    components appended since (e.g. a tool response) are not yet included.
    In practice this one-turn lag is negligible unless a single tool call
    adds a very large payload.

    Subclasses implement :meth:`compact` which receives the current
    ``ChatContext`` and returns a compacted copy.  The method is ``async``
    so that strategies requiring LLM calls (e.g. ``LLMSummarize``) work
    transparently; synchronous strategies simply never ``await``.

    Args:
        threshold (int): Trigger compaction when the most recent thunk's
            total token usage exceeds this value.  ``0`` disables compaction.
    """

    def __init__(self, *, threshold: int = 0) -> None:
        """Initialize with the token-count threshold."""
        self.threshold = threshold

    def should_compact(self, context: ChatContext) -> bool:
        """Return ``True`` when the last thunk's token usage exceeds ``threshold``.

        Reads ``total_tokens`` from the most recent ``ModelOutputThunk.usage``
        in *context*.  Returns ``False`` when no thunk with usage is present
        (e.g. before the first model call) or when ``threshold`` is not
        positive.

        Args:
            context: The context to check.

        Returns:
            ``True`` if the recovered token count exceeds ``self.threshold``
            and ``self.threshold`` is greater than 0.
        """
        if self.threshold <= 0:
            return False
        tokens = _last_usage_tokens(context)
        if tokens is None:
            return False
        return tokens > self.threshold

    async def maybe_compact(
        self,
        context: ChatContext,
        *,
        backend: Backend | None = None,
        goal: str | None = None,
    ) -> ChatContext:
        """Compact *context* only if it exceeds the threshold, otherwise return it unchanged.

        Args:
            context: The context to check and potentially compact.
            backend: The backend (forwarded to :meth:`compact`).
            goal: The react goal string (forwarded to :meth:`compact`).

        Returns:
            A compacted ``ChatContext`` if the threshold was exceeded,
            or the original *context* unchanged.
        """
        if self.should_compact(context):
            return await self.compact(context, backend=backend, goal=goal)
        return context

    @abc.abstractmethod
    async def compact(
        self,
        context: ChatContext,
        *,
        backend: Backend | None = None,
        goal: str | None = None,
    ) -> ChatContext:
        """Return a compacted copy of *context*.

        Args:
            context: The context to compact.
            backend: The backend (required by ``LLMSummarize``).
            goal: The react goal string (required by ``LLMSummarize``).

        Returns:
            A new, compacted ``ChatContext``.
        """


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------


class ClearAll(CompactionStrategy):
    """Discard the entire conversation body, keeping only the prefix.

    The prefix is everything up to and including the first ``ReactInitiator``.

    Args:
        threshold (int): Trigger compaction when the most recent thunk's total
            token usage exceeds this value.
    """

    async def compact(
        self,
        context: ChatContext,
        *,
        backend: Backend | None = None,
        goal: str | None = None,
    ) -> ChatContext:
        """Return a context containing only the prefix.

        Args:
            context: The context to compact.
            backend: Unused by this strategy; accepted for interface compatibility.
            goal: Unused by this strategy; accepted for interface compatibility.

        Returns:
            A new ``ChatContext`` containing only the prefix components.
        """
        components = context.as_list()
        prefix_end = _find_prefix_end(components)
        compacted = components[:prefix_end]

        MelleaLogger.get_logger().info(
            f"ClearAll: compacted context from {len(components)} to "
            f"{len(compacted)} components"
        )
        return rebuild_chat_context(compacted, window_size=context._window_size)


class KeepLastN(CompactionStrategy):
    """Keep the prefix plus the last *keep_n* body components.

    Args:
        keep_n (int): Number of recent body components to retain.
        threshold (int): Trigger compaction when the most recent thunk's total
            token usage exceeds this value.
    """

    def __init__(self, *, keep_n: int = 5, threshold: int = 0) -> None:
        """Initialize with the number of recent body components to keep."""
        super().__init__(threshold=threshold)
        self.keep_n = keep_n

    async def compact(
        self,
        context: ChatContext,
        *,
        backend: Backend | None = None,
        goal: str | None = None,
    ) -> ChatContext:
        """Return a context with the prefix and the last *keep_n* body components.

        Args:
            context: The context to compact.
            backend: Unused by this strategy; accepted for interface compatibility.
            goal: Unused by this strategy; accepted for interface compatibility.

        Returns:
            A new ``ChatContext`` with the prefix plus the most recent *keep_n*
            body components, or the original *context* if the body is already
            at or below *keep_n* in length.
        """
        components = context.as_list()
        prefix_end = _find_prefix_end(components)
        prefix = components[:prefix_end]
        body = components[prefix_end:]

        if len(body) <= self.keep_n:
            return context  # nothing to compact

        compacted = prefix + body[-self.keep_n :]

        MelleaLogger.get_logger().info(
            f"KeepLastN(keep_n={self.keep_n}): compacted context from "
            f"{len(components)} to {len(compacted)} components"
        )
        return rebuild_chat_context(compacted, window_size=context._window_size)


class LLMSummarize(CompactionStrategy):
    """Summarize old body components with the LLM, keep last *keep_n* verbatim.

    Requires ``backend`` and ``goal`` to be passed to :meth:`compact`.

    Args:
        keep_n (int): Number of recent body components to retain verbatim.
        threshold (int): Trigger compaction when the most recent thunk's total
            token usage exceeds this value.
    """

    def __init__(self, *, keep_n: int = 5, threshold: int = 0) -> None:
        """Initialize with the number of recent body components to keep."""
        super().__init__(threshold=threshold)
        self.keep_n = keep_n

    async def compact(
        self,
        context: ChatContext,
        *,
        backend: Backend | None = None,
        goal: str | None = None,
    ) -> ChatContext:
        """Return a context with the prefix, an LLM summary, and recent body components.

        Args:
            context: The context to compact.
            backend: Backend used to generate the summary; required.
            goal: The react goal string, included in the summary prompt; required.

        Returns:
            A new ``ChatContext`` containing the prefix, a single summary
            ``Message`` produced by the backend, and the most recent *keep_n*
            body components verbatim. Returns the original *context* if the
            body is already at or below *keep_n* in length.

        Raises:
            ValueError: If *backend* or *goal* are not provided.
        """
        if backend is None or goal is None:
            raise ValueError(
                "LLMSummarize requires both 'backend' and 'goal' arguments"
            )

        from mellea.stdlib import functional as mfuncs
        from mellea.stdlib.context import SimpleContext

        components = context.as_list()
        prefix_end = _find_prefix_end(components)
        prefix = components[:prefix_end]
        body = components[prefix_end:]

        if len(body) <= self.keep_n:
            return context  # nothing to compact

        old = body[: -self.keep_n] if self.keep_n > 0 else body
        recent = body[-self.keep_n :] if self.keep_n > 0 else []

        # Build a textual representation of old components for summarization.
        context_lines: list[str] = []
        for c in old:
            if isinstance(c, ToolMessage):
                context_lines.append(f"tool ({c.name}): {c.content}")
            elif isinstance(c, Message):
                context_lines.append(f"{c.role}: {c.content}")
            elif isinstance(c, ModelOutputThunk):
                context_lines.append(f"assistant: {c.value}")
            elif isinstance(c, CBlock):
                context_lines.append(str(c))
            else:
                context_lines.append(str(getattr(c, "content", c)))

        summary_prompt = (
            "You are summarizing research progress to maintain context "
            "within token limits.\n\n"
            f"GOAL: {goal}\n\n"
            "Provide a comprehensive summary of the research context below. "
            "Your summary should:\n"
            "- Preserve ALL specific facts, numbers, names, URLs, and search "
            "queries found\n"
            "- Note which tools were called and what results were obtained\n"
            "- Highlight key findings and any dead ends encountered\n"
            "- Be structured clearly so the research can continue seamlessly"
            "\n\nContext to summarize:\n"
            f"{chr(10).join(context_lines)}"
        )

        summary_action = Message(role="user", content=summary_prompt)
        result, _ = await mfuncs.aact(
            action=summary_action,
            context=SimpleContext(),
            backend=backend,
            requirements=[],
            strategy=None,
            await_result=True,
        )

        summary_text = result.value or ""
        summary_message = Message(
            role="user",
            content=(
                f"[CONTEXT SUMMARY]\n{summary_text}\n\nContinue working on: {goal}"
            ),
        )

        compacted = [*prefix, summary_message, *recent]

        MelleaLogger.get_logger().info(
            f"LLMSummarize(keep_n={self.keep_n}): compacted context from "
            f"{len(components)} to {len(compacted)} components"
        )
        return rebuild_chat_context(compacted, window_size=context._window_size)
