"""Chat-style context with pluggable compaction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mellea.core import CBlock, Component, Context

if TYPE_CHECKING:
    from mellea.stdlib.context.compactor import InlineCompactor


class ChatContext(Context):
    """Chat context that accumulates turns and optionally compacts on each `add`.

    By default the context performs **no compaction** — the full history is
    retained. Compaction is opt-in: pass `compactor=` for a custom
    strategy, or `window_size=` as sugar for `WindowCompactor(size=...)`.

    Note:
        Compaction is now applied at `add()` time and persists in the linked
        list, so `as_list()` and `view_for_generation()` both reflect the
        post-compaction history. Earlier versions kept the full history in
        `as_list()` and only windowed the model-facing view, so any caller
        that used `len(ctx.as_list())` as a session-wide interaction count
        will now silently undercount once the compactor fires. Track turn
        counts out-of-band (e.g. on the session) if you need them.

    Args:
        compactor (InlineCompactor | None): The compactor invoked on every
            `add`. `None` (the default) means no compaction; full history
            is kept.
        window_size (int | None): Sugar that constructs a
            :class:`WindowCompactor`. Mutually exclusive with `compactor`.
            `None` (the default) means no windowing.
    """

    def __init__(
        self,
        *,
        compactor: InlineCompactor | None = None,
        window_size: int | None = None,
    ) -> None:
        """Initialize a ChatContext with an optional compactor."""
        if compactor is not None and window_size is not None:
            raise ValueError(
                "ChatContext: pass either `compactor` or `window_size`, not both."
            )
        if compactor is not None:
            from mellea.stdlib.context.compactor import InlineCompactor

            if not isinstance(compactor, InlineCompactor):
                raise TypeError(
                    f"ChatContext requires an InlineCompactor; got "
                    f"{type(compactor).__name__}. Wrap it in ThresholdCompactor, "
                    "use via react(compactor=...), or call compact(ctx, ...) "
                    "manually instead."
                )
        super().__init__()
        if compactor is None and window_size is not None:
            from mellea.stdlib.context.compactor import WindowCompactor

            self._compactor: InlineCompactor | None = WindowCompactor(size=window_size)
        else:
            self._compactor = compactor

    def add(self, c: Component | CBlock) -> ChatContext:
        """Append `c` and run the compactor; return the resulting context.

        Args:
            c (Component | CBlock): The component or content block to append.

        Returns:
            ChatContext: A new `ChatContext` carrying the same compactor.
        """
        new = ChatContext.from_previous(self, c)
        new._compactor = self._compactor
        if self._compactor is not None:
            new = self._compactor.compact(new)
        return new

    def view_for_generation(self) -> list[Component | CBlock] | None:
        """Return the components to forward to the model.

        Compaction is now applied at `add` time (Pattern 1), so this just
        returns the linear history. `None` is returned when the underlying
        history is non-linear.

        Returns:
            list[Component | CBlock] | None: Ordered list of context entries.
        """
        return self.as_list()


def _rebuild_chat_context(
    components: list[Component | CBlock], *, compactor: InlineCompactor | None = None
) -> ChatContext:
    """Build a fresh `ChatContext` linked-list without triggering compaction.

    Used by `WindowCompactor` (and any future compactors that need to rebuild
    a chat history). Manual node construction sidesteps `ChatContext.add` so
    compactors don't recurse during their own work.

    Args:
        components: Components to materialise as the new context, in order.
        compactor: Compactor to attach to every node of the rebuilt context.

    Returns:
        A new `ChatContext` whose linear history is exactly `components`.
    """
    ctx: ChatContext = ChatContext.__new__(ChatContext)
    Context.__init__(ctx)
    ctx._compactor = compactor
    for c in components:
        new: ChatContext = ChatContext.__new__(ChatContext)
        new._previous = ctx
        new._data = c
        new._is_root = False
        new._is_chat_context = ctx._is_chat_context
        new._compactor = compactor
        ctx = new
    return ctx
