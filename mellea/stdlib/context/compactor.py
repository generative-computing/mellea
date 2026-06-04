"""Generic `Compactor` protocol for shrinking a `Context`.

A `Compactor` returns a fresh, compacted copy of a context. Implementations
must never mutate the input — by convention, every alteration must produce a
new `Context` instance (the base class enforces this via `from_previous`).

Two usage patterns are supported:

- **Pattern 1 (in `Context.add`):** A subclass of `Context` holds a
  `Compactor` and applies it whenever a new component is appended.
- **Pattern 2 (manual):** The caller invokes `compactor.compact(ctx)`
  directly between turns, e.g. when compaction is exposed to the model as a
  tool.

See `docs/examples/context/` for full usage examples.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, TypeAlias, TypeVar

from mellea.core import CBlock, Component, Context, ModelOutputThunk
from mellea.core.backend import Backend
from mellea.core.utils import MelleaLogger
from mellea.stdlib.components.chat import Message, ToolMessage
from mellea.stdlib.context.chat import _rebuild_chat_context
from mellea.stdlib.context.simple import SimpleContext

if TYPE_CHECKING:
    from mellea.stdlib.context.chat import ChatContext

T = TypeVar("T", bound=Context)


# --------------------------------------------------------------------------- #
# Pin predicates                                                              #
# --------------------------------------------------------------------------- #

PinPredicate: TypeAlias = Callable[[list[Component | CBlock]], int]
"""A function that returns the index after the pinned prefix.

Given the full ordered list of context components, a `PinPredicate`
returns the integer index `idx` such that `components[:idx]` is the
pinned prefix that the compactor must preserve, and `components[idx:]`
is the body that compaction acts on.

The shape subsumes both "contiguous role-based prefix" (e.g.
:func:`pin_system`) and "find the first marker component" styles.
"""


def pin_nothing(components: list[Component | CBlock]) -> int:
    """A :class:`PinPredicate` that pins nothing — pure body, no protected prefix."""
    return 0


def pin_system(components: list[Component | CBlock]) -> int:
    """Pin contiguous leading `Message(role="system")` components.

    Stops at the first non-system component. A system message that appears
    later in the conversation is *not* pinned.
    """
    for i, c in enumerate(components):
        if not (isinstance(c, Message) and c.role == "system"):
            return i
    return len(components)


def pin_system_and_initial_user(components: list[Component | CBlock]) -> int:
    """Pin leading system messages PLUS the first user message that follows.

    Useful when the initial user prompt encodes the goal of the conversation
    and should survive compaction along with any system instructions.
    """
    i = pin_system(components)
    if i < len(components):
        c = components[i]
        if isinstance(c, Message) and c.role == "user":
            i += 1
    return i


def _last_usage_tokens(ctx: Context) -> int | None:
    """Return cumulative token count of the conversation as of the most recent turn.

    Walks `ctx` back-to-front looking for a `ModelOutputThunk` whose
    `generation.usage` dict has been populated by a backend's
    `post_processing`. Returns `total_tokens` from that thunk — which,
    for a chat backend, is `prompt_tokens` (size of the full conversation
    sent to the model) plus `completion_tokens` (the model's reply). It
    is therefore an estimate of the *current* conversation size, not just
    one call's tokens in isolation.

    Falls back to `prompt_tokens + completion_tokens` when `total_tokens`
    is missing. Returns `None` if no usable token count can be recovered
    (typical before the first model call completes).
    """
    for c in reversed(ctx.as_list()):
        if isinstance(c, ModelOutputThunk) and c.generation.usage is not None:
            usage = c.generation.usage
            total = usage.get("total_tokens")
            if total is None:
                pt = usage.get("prompt_tokens") or 0
                ct = usage.get("completion_tokens") or 0
                total = pt + ct
            return total if total and total > 0 else None
    return None


class Compactor(Protocol):
    """Protocol for objects that compact a `Context` into a smaller copy.

    A compactor receives a context and returns a new context that retains only
    the data the strategy considers worth keeping. Implementations MUST NOT
    mutate the input context; they must return a fresh instance and copy over
    any data that should be preserved.

    `compact()` is generic in `T` (a `Context` subtype) so concrete
    compactors can narrow their input/output type — for example a
    chat-only compactor overrides the method as
    ``def compact(self, ctx: ChatContext, *, backend=None) -> ChatContext``.

    The protocol is sync. Compactors that need to perform a backend call
    (e.g. :class:`LLMSummarizeCompactor`) hide the async work behind the sync
    method internally — see that class for the strategy used.
    """

    def compact(self, ctx: T, *, backend: Backend | None = None) -> T:
        """Return a compacted copy of `ctx`.

        Args:
            ctx: The context to compact. Must be left unchanged.
            backend: Optional backend. Generic compactors that only filter
                components can ignore it.

        Returns:
            A new context of the same type as `ctx` containing only the
            retained data.
        """
        ...


class InlineCompactor:
    """Marker base for compactors safe to attach directly to `ChatContext`.

    A compactor is "inline-safe" when its `compact()` does not call a backend
    on every `add()`. `ChatContext.add()` invokes `compact()` without a
    backend argument, so any compactor wired into `ChatContext(compactor=...)`
    must either avoid backend calls (e.g. :class:`WindowCompactor`) or gate
    them sparsely (e.g. :class:`ThresholdCompactor`). Compactors that would
    invoke the backend on every `add()` (e.g. :class:`LLMSummarizeCompactor`)
    must NOT inherit this marker — use them via `react(compactor=...)` or
    by calling `compact(ctx, backend=...)` manually instead.

    The marker is purely nominal: opt in by inheriting, opt out by not. Pure
    structural :class:`Compactor` Protocol satisfaction is not enough.

    Subclasses must override :meth:`compact`; the base implementation raises
    :class:`NotImplementedError`. Carrying the method signature here lets
    `InlineCompactor` be used as a static type (`ChatContext` parameters,
    `_compactor` attribute) without losing the `Compactor` contract.
    """

    def compact(
        self, ctx: ChatContext, *, backend: Backend | None = None
    ) -> ChatContext:
        """Subclasses must override this with their concrete strategy."""
        raise NotImplementedError("InlineCompactor subclasses must implement compact()")


class WindowCompactor(InlineCompactor):
    """Retains the last `size` body components of a `ChatContext`.

    Uses `pin_predicate` to decide which leading components to preserve as
    a protected prefix; the size limit is then applied to the body that
    remains. The total context length after compaction is
    `len(prefix) + min(size, body_len)`. `size` counts only body
    components.

    When the body is already at or below `size`, `ctx` is returned
    unchanged so the original linked-list and `previous_node` chain are
    preserved. The result carries the same `Compactor` as the input so
    subsequent `add()` calls keep compacting.

    Args:
        size (int): Maximum number of most-recent body components to retain.
            Pinned prefix components do NOT count against this budget.
            `size=0` is a special case that drops the body entirely,
            keeping only the pinned prefix. Negative values raise
            :class:`ValueError`.
        pin_predicate (PinPredicate): Function that decides the prefix
            boundary. Defaults to :func:`pin_system`, which pins contiguous
            leading `Message(role="system")` components. Pass
            :func:`pin_nothing` for pure last-N behaviour or any other
            `PinPredicate` (e.g. :func:`pin_system_and_initial_user`).
    """

    def __init__(self, *, size: int, pin_predicate: PinPredicate = pin_system) -> None:
        """Initialize with the desired body window size and a pin predicate."""
        if size < 0:
            raise ValueError("WindowCompactor size must be non-negative")
        self.size = size
        self.pin_predicate = pin_predicate

    def compact(
        self, ctx: ChatContext, *, backend: Backend | None = None
    ) -> ChatContext:
        """Return a copy of `ctx` truncated to the last `size` body components.

        Args:
            ctx: The chat context to compact.
            backend: Unused by this strategy; accepted for protocol compatibility.

        Returns:
            A new `ChatContext` whose history is the pinned prefix plus the
            last `size` body components, carrying `ctx`'s compactor.
            Returns `ctx` itself if no truncation is required.
        """
        full = ctx.as_list()
        pin_end = self.pin_predicate(full)
        body_len = len(full) - pin_end

        if body_len <= self.size:
            return ctx

        keep_body = full[pin_end:][-self.size :] if self.size > 0 else []
        compacted = full[:pin_end] + keep_body
        return _rebuild_chat_context(compacted, compactor=ctx._compactor)


class ThresholdCompactor(InlineCompactor):
    """Wraps an inner `Compactor`, gating it on the conversation's token size.

    Despite the suffix, this class does not compact directly — it forwards
    to `inner.compact` only when the conversation has grown larger than
    `threshold` tokens; otherwise the input is returned unchanged.

    The token measurement is read off the most recent `ModelOutputThunk`'s
    `generation.usage` (via :func:`_last_usage_tokens`). Because chat
    backends report `prompt_tokens` as the size of the full history they
    were given as input, `total_tokens = prompt_tokens + completion_tokens`
    on the latest thunk effectively measures *the size of the conversation
    after that turn*, not just one isolated call. So the gate fires once
    cumulative context size crosses `threshold`.

    Caveats:

    - Components appended *after* the last thunk (e.g. a tool response in
      the same turn) are not yet reflected in the reading — there is a
      one-turn lag, negligible unless a single tool call adds a very large
      payload.
    - When the inner compactor shrinks the context, the *next* model call
      will produce a smaller `prompt_tokens`, so the gate will close
      again. The threshold is not a high-water mark.
    - Returns the input unchanged if no thunk with usage is found yet
      (typical before the first model call completes).

    Args:
        inner (Compactor): The compactor to invoke once the threshold is
            exceeded.
        threshold (int): Trigger the inner compactor when the conversation's
            measured token size (most recent thunk's `total_tokens`)
            exceeds this value. `0` or negative disables the gate (the
            inner is never invoked).
    """

    def __init__(self, inner: Compactor, *, threshold: int) -> None:
        """Initialize with the inner compactor and token threshold."""
        self.inner = inner
        self.threshold = threshold

    def compact(
        self, ctx: ChatContext, *, backend: Backend | None = None
    ) -> ChatContext:
        """Forward to `inner.compact` only when `ctx` exceeds the threshold.

        Args:
            ctx: The context to potentially compact.
            backend: Forwarded to the inner compactor.

        Returns:
            `inner.compact(ctx, backend=backend)` when the recovered token
            count exceeds `self.threshold`, otherwise `ctx` unchanged.
        """
        if self.threshold <= 0:
            return ctx
        tokens = _last_usage_tokens(ctx)
        if tokens is None or tokens <= self.threshold:
            return ctx
        return self.inner.compact(ctx, backend=backend)


_DEFAULT_SUMMARY_PROMPT = (
    "You are summarizing a conversation to maintain context within token "
    "limits.\n\n"
    "Provide a concise summary that:\n"
    "- Preserves specific facts, numbers, names, URLs, and key data\n"
    "- Notes which tools were called and what results were obtained\n"
    "- Highlights key decisions, findings, and unresolved issues\n"
    "- Is structured clearly so the conversation can continue seamlessly\n\n"
    "Conversation to summarize:\n{conversation}"
)


def _run_coro_blocking(coro):  # type: ignore[no-untyped-def]
    """Run an awaitable to completion regardless of the calling context.

    - Outside any event loop: `asyncio.run(coro)`.
    - Inside a running event loop: spawn a worker thread that runs a fresh
      event loop with `asyncio.run` and block until it returns.

    Used by sync compactors that need to call async backend code (e.g.
    :class:`LLMSummarizeCompactor`).

    Warning:
        When called from inside a running event loop (e.g. `react()`), the
        second branch above blocks the calling thread — and therefore the
        loop — for the full duration of the coroutine. **Nothing else on the
        loop can make progress** while the worker runs: scheduled callbacks,
        telemetry flushers, cancellation signals, other sessions sharing the
        loop, periodic keepalives — all are stalled. Acceptable for a
        strictly serial flow like ReACT (the next iteration cannot start
        until compaction finishes anyway), but unsafe if the loop has
        concurrent tasks that need to keep running.

        Backends that hold *per-loop* resources may behave unexpectedly.
        :class:`httpx.AsyncClient`, for instance, is bound to the event
        loop on which it was created; the coroutine here runs on a fresh
        loop inside a worker thread, so any async resource captured in a
        closure or stored on a backend instance from the outer loop cannot
        be used directly. The typical symptom is `RuntimeError: This event
        loop is already running` or a hung request.

        The long-term fix is an async variant on the :class:`Compactor`
        protocol so callers can `await` natively instead of bridging
        through a worker thread. Until then, only invoke compactors that
        need a backend from contexts where this trade-off is acceptable
        (typically: inside `react`, in a manual `compact()` call between
        turns, or from a synchronous script).
    """
    import asyncio
    import concurrent.futures

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class LLMSummarizeCompactor:
    """Replace old body components with an LLM-generated summary, keep last `keep_n` verbatim.

    Implements the sync :class:`Compactor` protocol. The compactor's body
    needs to call the (async) backend; that async work is hidden inside the
    sync `compact` method via :func:`_run_coro_blocking`. The pinned
    prefix (chosen by `pin_predicate`) is preserved unchanged; body
    components older than the last `keep_n` are flattened into a single
    `Message(role="user")` whose content is a structured summary; the
    last `keep_n` body components are kept verbatim.

    Default `pin_predicate` is :func:`pin_nothing`, which means the entire
    conversation participates in summarisation. For react workflows pass
    :func:`mellea.stdlib.components.react.pin_react_initiator` so the goal
    and tool registration survive untouched.

    Note:
        This class does NOT inherit :class:`InlineCompactor`, so it cannot be
        passed to `ChatContext(compactor=...)` directly — that would invoke
        the backend on every `add()`. Use via `react(compactor=...)`,
        wrap in :class:`ThresholdCompactor` (which gates by token usage), or
        call `compact(ctx, backend=...)` manually.

    Note:
        Summarisation is text-only and lossy for multimodal or heavy-tool
        sessions. Image and document attachments on `Message` components
        are noted by count only ("[N image(s) attached]") rather than
        reproduced; `ModelOutputThunk` entries that carry only tool calls
        (`value is None`) render the call name and arguments. If your
        application depends on faithful preservation of attachments or
        full tool-call payloads across compaction, prefer
        :class:`WindowCompactor` (which keeps recent components verbatim)
        or implement a domain-specific :class:`Compactor`.

    Args:
        default_backend (Backend): Backend used by `compact()` when the
            caller does not supply one. Required: `LLMSummarizeCompactor`
            cannot do its job without a backend at compaction time. A
            `backend=` kwarg passed to `compact()` overrides this default
            for that call only.
        keep_n (int): Number of recent body components to keep verbatim.
            `0` summarises everything below the prefix.
        pin_predicate (PinPredicate): Function that decides the prefix
            boundary. Defaults to :func:`pin_nothing`.
        prompt_template (str | None): Custom summary prompt. Must contain
            the literal `{conversation}` placeholder, which is filled in
            with a textual rendering of the body to summarise. Defaults to
            a generic conversation-summary template.
        model_options (dict | None): Forwarded to `mfuncs.aact` for the
            summarisation call. Use this to set a real `max_tokens` budget
            (most local backends default to 256-512, which silently truncates
            long summaries) or any other backend-specific knob. Note:
            :func:`react_summary_prompt`'s `max_tokens_hint` adds only a
            soft prompt-side nudge; pair it with `model_options={"max_tokens": N}`
            for hard enforcement.
    """

    def __init__(
        self,
        *,
        default_backend: Backend,
        keep_n: int = 5,
        pin_predicate: PinPredicate = pin_nothing,
        prompt_template: str | None = None,
        model_options: dict | None = None,
    ) -> None:
        """Initialize with a default backend, recent-body window, pin predicate, prompt, and model options."""
        if keep_n < 0:
            raise ValueError("LLMSummarizeCompactor keep_n must be non-negative")
        template = (
            prompt_template if prompt_template is not None else _DEFAULT_SUMMARY_PROMPT
        )
        if "{conversation}" not in template:
            raise ValueError(
                "LLMSummarizeCompactor prompt_template must contain '{conversation}'"
            )
        self.default_backend = default_backend
        self.keep_n = keep_n
        self.pin_predicate = pin_predicate
        self.prompt_template = template
        self.model_options = model_options

    def compact(
        self, ctx: ChatContext, *, backend: Backend | None = None
    ) -> ChatContext:
        """Return a context with the prefix, an LLM summary, and recent body components.

        Args:
            ctx: The chat context to compact.
            backend: Backend used to generate the summary. When `None` the
                `default_backend` set at construction is used instead.

        Returns:
            A new `ChatContext` containing the prefix, a single summary
            `Message` produced by the backend, and the most-recent
            `keep_n` body components verbatim. Returns `ctx` unchanged
            when the body is already at or below `keep_n` in length, or
            when the backend call fails (see Note).

        Note:
            Compaction is best-effort: if the backend call raises (rate
            limit, network error, timeout, etc.) the exception is caught, a
            warning is logged, and `ctx` is returned unchanged. The next
            `compact()` invocation will retry. Programming-error classes
            (`TypeError`, `AttributeError`, `AssertionError`, `LookupError`
            — which covers `KeyError` and `IndexError`) propagate so genuine
            bugs surface instead of being silently masked as "backend
            failure". `KeyboardInterrupt` and other `BaseException`s also
            propagate so users can still interrupt a stuck loop.
        """
        backend = backend or self.default_backend

        full = ctx.as_list()
        pin_end = self.pin_predicate(full)
        body_len = len(full) - pin_end
        if body_len <= self.keep_n:
            return ctx

        try:
            return _run_coro_blocking(
                self._async_compact(ctx, backend, full, pin_end)
            )
        except (TypeError, AttributeError, AssertionError, LookupError):
            raise
        except Exception as exc:
            MelleaLogger.get_logger().warning(
                "LLMSummarizeCompactor: summarisation backend call failed "
                "(%s: %s); returning context unchanged. The conversation will "
                "keep growing until the next successful compaction.",
                type(exc).__name__,
                exc,
            )
            return ctx

    async def _async_compact(
        self,
        ctx: ChatContext,
        backend: Backend,
        full: list[Component | CBlock],
        pin_end: int,
    ) -> ChatContext:
        """Async core — renders the body, calls the backend, rebuilds the context.

        `full` and `pin_end` are passed in by `compact()` to avoid re-running
        `ctx.as_list()` and `self.pin_predicate(full)` after the early-return
        check.
        """
        # mfuncs has to stay lazy: mellea.stdlib.functional imports SimpleContext
        # via the context package init, which re-exports from this module.
        from mellea.stdlib import functional as mfuncs

        prefix = full[:pin_end]
        body = full[pin_end:]

        old = body[: -self.keep_n] if self.keep_n > 0 else body
        recent = body[-self.keep_n :] if self.keep_n > 0 else []

        # Render `old` to text the LLM can consume. This is intentionally a
        # text-only rendering: image and document attachments on Messages are
        # noted as markers (count only) rather than reproduced, and tool-call
        # arguments are stringified. The summary is lossy for multimodal and
        # heavy-tool sessions by design — see class docstring.
        lines: list[str] = []
        for c in old:
            if isinstance(c, ToolMessage):
                lines.append(f"tool ({c.name}): {c.content}")
            elif isinstance(c, Message):
                attachments: list[str] = []
                imgs = getattr(c, "_images", None)
                if imgs:
                    attachments.append(f"[{len(imgs)} image(s) attached]")
                docs = getattr(c, "_docs", None)
                if docs:
                    attachments.append(f"[{len(docs)} document(s) attached]")
                attached = (" " + " ".join(attachments)) if attachments else ""
                lines.append(f"{c.role}: {c.content}{attached}")
            elif isinstance(c, ModelOutputThunk):
                if c.value:
                    lines.append(f"assistant: {c.value}")
                elif c.tool_calls:
                    rendered = ", ".join(
                        f"{name}({dict(tc.args)})" for name, tc in c.tool_calls.items()
                    )
                    lines.append(f"assistant called tools: {rendered}")
                # else: thunk with neither value nor tool_calls is skipped —
                # nothing useful to summarise and a literal "<empty>" marker
                # tends to show up verbatim in the resulting summary.
            elif isinstance(c, CBlock):
                lines.append(str(c))
            else:
                # Catch-all for `Component` subclasses that aren't `Message`/
                # `ToolMessage`/`ModelOutputThunk` (e.g. `ReactInitiator`).
                # Without special handling these would render as the default
                # `<… object at 0x…>` repr and the summary would lose all
                # information that the entry existed at all. Emit at minimum
                # the type name plus a `content` attribute if present, so
                # the summariser sees a marker.
                content = getattr(c, "content", None)
                if content is not None:
                    lines.append(f"<{type(c).__name__}: {content}>")
                else:
                    lines.append(f"<{type(c).__name__}>")

        prompt = self.prompt_template.format(conversation="\n".join(lines))
        result, _ = await mfuncs.aact(
            action=Message(role="user", content=prompt),
            context=SimpleContext(),
            backend=backend,
            requirements=[],
            strategy=None,
            model_options=self.model_options,
            await_result=True,
            # Internal framework call: silence aact's context-type warning so
            # it stays quiet if the context argument is later changed to a
            # non-SimpleContext. Matches react.py's pattern.
            silence_context_type_warning=True,
        )

        summary_message = Message(
            role="user", content=f"[CONTEXT SUMMARY]\n{result.value or ''}"
        )
        compacted = [*prefix, summary_message, *recent]
        return _rebuild_chat_context(compacted, compactor=ctx._compactor)
