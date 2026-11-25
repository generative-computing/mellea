# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Learnable functions: functions a student can be trained to emulate.

A *learnable function* is an ordinary Python function marked ``@learnable``.
Its signature is the student's interface; its body is the teacher, and can be
anything (model calls, code that plans, judges, repair loops). Learning it is
emulation: a student (an adapter) learns to produce what the body produces,
given what the signature lets it see. Learning prescribes no judge and no
strategy: how the body earns its output is the body's own business.

The signature:

* At most one parameter is typed as a ``Context``: the **context argument**,
  the conversation the call takes place in. The others are **structured
  arguments**.
* **Privileged parameters** (``@learnable(privileged=("turn",))``) are facts
  the caller knows and the student is never told, given to the body to steer
  and check its output. They are recorded by name on every ``Record`` and
  stripped at training-data derivation time (``sft_rows``). **The rule that
  comes with them:** they may steer and check, but any fact the output states
  must be recoverable from the visible arguments, or the student learns to
  invent it.
* On a method, ``self`` is the teacher's state between calls: hidden state,
  which the student has to infer from its context. It is never recorded.
* The **return annotation is the output type**, and it is text: ``str`` (or no
  annotation), or a ``Literal`` of strings. The output is the text itself, a
  ``Literal``'s checked against its values (``Learnable.parse``). Any other
  type raises ``TypeError``: a structured result is a tool call, not JSON in
  the content. ``Learnable.io_config`` is the ``io.yaml`` a trained adapter
  of the function would declare.

A body returns its outputs, not a value: a non-empty
``list[ComputedModelOutputThunk]``, or that list with a ``Score`` for each
(``(outputs, scores)``). The first output is what the call returned; the rest
are candidates not taken. Each output's text is its ``value``; what the body
notes about it goes in its ``meta``.

**Recording captures the faithful trace.** Each thunk is snapshot *after* the
call, reading the thunk's ``_call`` for the context and tools the backend
actually used, and its ``generation`` for the model, latency, and usage. The
snapshot is a ``Thunk``: the context as OpenAI messages, the tools, the full
assistant response (content, tool calls, reasoning), and metadata. The student
receives the visible arguments with Context objects intact — not a pre-rendered
view — so it operates at the same Component-level abstraction the body does.
OpenAI format conversion happens only at serialization time (``Record.to_json``)
and training-data derivation (``sft_rows``, ``preference_rows``).

**Two roles in every call.** The *policy* produces the output the call returns,
which the caller uses, so the rest of the episode follows from it. The
*teacher* labels the call: its outputs and their scores are what training
learns. In a plain call the body is both. ``fn.policy(student)`` makes a
student the policy for the calls inside its block (given the visible arguments,
it returns what a body returns): with no teacher, it is inference (serving,
evaluation, a scripted test double); with ``label=True``, the body still runs
and labels, on the state the student's earlier outputs led to: on-policy data.
The caller gets the policy's first output's text (checked, for a ``Literal``)
on every path, so the driver making the calls cannot tell who answered, nor
whether a ``Recorder`` was listening.

Called inside a ``Recorder``, each call leaves a ``Record``: the call's
arguments, its thunks (the teacher's snapshots, or with no teacher the
policy's), their scores if given, who answered (``policy``), and the learnable
calls its body made (``calls``), so the calls of an episode form a tree. A
recorder may keep only some of a function's calls (``keep``). Outside a
recorder, a learnable function is a plain function call.
"""

from __future__ import annotations

import contextvars
import dataclasses
import enum
import functools
import hashlib
import inspect
import json
import types
import typing
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel

from ...core import ComputedModelOutputThunk, Context, ModelOutputThunk, Span
from ...helpers.annotation_helpers import resolve_signature_annotations
from ...helpers.openai_compatible_helpers import (
    build_tool_calls,
    message_to_openai_message,
)


def _spans_to_openai(spans: list[Span]) -> list[dict[str, Any]]:
    """Convert a flat span list (as stored on ``_CallInfo.context``) to OpenAI messages.

    `Message` spans are serialized via `message_to_openai_message`.
    `ModelOutputThunk` spans become assistant messages with content, tool_calls,
    and reasoning_content when present.  Other component types are skipped.
    """
    from ..components.chat import Message

    out: list[dict[str, Any]] = []
    for span in spans:
        if isinstance(span, ModelOutputThunk):
            msg: dict[str, Any] = {"role": "assistant", "content": span.value or ""}
            tc = build_tool_calls(span)
            if tc:
                msg["tool_calls"] = tc
            if span.thinking:
                msg["reasoning_content"] = span.thinking
            out.append(msg)
        elif isinstance(span, Message):
            out.append(message_to_openai_message(span))
    return out


def plain(value: Any) -> Any:
    """`value` as JSON-ready data.

    A context is its OpenAI messages, a dataclass or a Pydantic model its
    fields, an enum its value.

    Args:
        value: The value.

    Returns:
        Its JSON-ready form.

    Raises:
        TypeError: For any other kind of object (the learnable view must be
            exact, so nothing is turned into text by guessing).
    """
    if isinstance(value, Context):
        return _spans_to_openai(value.as_list())
    if isinstance(value, enum.Enum):
        return plain(value.value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list | tuple):
        return [plain(x) for x in value]
    if isinstance(value, dict):
        return {str(k): plain(x) for k, x in value.items()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: plain(getattr(value, f.name)) for f in dataclasses.fields(value)
        }
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    raise TypeError(f"no JSON form for a {type(value).__name__}")


@dataclass
class Score:
    """A verdict on one output.

    Args:
        value: A number training may use (a reward), or `None`.
        parts: Its components, for analysis (e.g. each judge's verdict).
        keep: Whether training takes it as an output to produce.
    """

    value: float | None = None
    parts: dict[str, Any] = field(default_factory=dict)
    keep: bool = True


@dataclass
class Thunk:
    """A serializable snapshot of one completion's ModelOutputThunk.

    Captured AFTER the call, from what the backend actually sent.

    Args:
        context: ``_call.context`` as OpenAI messages.
        tools: Tools from ``_call.model_options``, or ``None``.
        response: The full assistant message: content, tool_calls, reasoning_content.
        meta: The thunk's metadata dict.
        model: The model identifier from ``thunk.generation``.
        latency_ms: Time-to-first-token in milliseconds, or ``None``.
    """

    context: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    response: dict[str, Any]
    meta: dict[str, Any]
    model: str | None
    latency_ms: float | None


def _snapshot(thunk: ComputedModelOutputThunk) -> Thunk:
    """Capture a `Thunk` from a computed thunk, reading ``_call`` for what the backend used."""
    # Context: convert the span list the backend saw.
    context = _spans_to_openai(thunk._call.context or [])

    # Tools: extract from model_options if present.
    tools: list[dict[str, Any]] | None = None
    opts = thunk._call.model_options
    if opts:
        from ...backends.model_options import ModelOption
        from ...backends.tools import convert_tools_to_json

        raw_tools = opts.get(ModelOption.TOOLS)
        if raw_tools:
            tool_dict = (
                raw_tools
                if isinstance(raw_tools, dict)
                else {t.name: t for t in raw_tools}
            )
            tools = convert_tools_to_json(tool_dict) or None

    # Response: the assistant message.
    response: dict[str, Any] = {"role": "assistant", "content": thunk.value}
    tc = build_tool_calls(thunk)
    if tc:
        response["tool_calls"] = tc
    if thunk.thinking:
        response["reasoning_content"] = thunk.thinking

    return Thunk(
        context=context,
        tools=tools,
        response=response,
        meta=dict(thunk.meta),
        model=thunk.generation.model,
        latency_ms=thunk.generation.ttfb_ms,
    )


Returned: TypeAlias = (
    list[ComputedModelOutputThunk] | tuple[list[ComputedModelOutputThunk], list[Score]]
)
"""What a body, or a student standing in for it as the policy, returns: its
outputs, or its outputs and a score for each."""

Student: TypeAlias = Callable[[dict[str, Any]], Returned]
"""A policy other than the body: given a call's visible arguments (a dict from
parameter name to value, Context objects intact, privileged keys excluded), it
returns what a body returns."""


@dataclass
class Policy:
    """Who answered a call that was not answered by its body, and what it returned.

    Args:
        by: The policy's name (`fn.policy`'s `name`, or the student's own
            `__name__`, or its type's).
        output: Its first output: what the call returned.
    """

    by: str
    output: ComputedModelOutputThunk


@dataclass(eq=False)
class Record:
    """One call of a learnable function.

    Records are compared by identity: they are the nodes of a tree, and two
    calls with the same arguments and outputs are still two calls.

    Args:
        function: The function's qualified name.
        arguments: All arguments as passed (self excluded), including privileged.
        privileged: Names of the privileged parameters.
        thunks: One `Thunk` per completion (the teacher's outputs, or the
            policy's when there is no teacher).
        scores: ``None``, or one `Score` per thunk, in order: a call is
            scored whole or not at all.
        calls: The learnable calls its body made, in the order they started.
        policy: ``None`` when the body answered (a plain call); otherwise who
            did and what it returned.
    """

    function: str
    arguments: dict[str, Any]
    privileged: tuple[str, ...]
    thunks: list[Thunk]
    scores: list[Score] | None = None
    calls: list[Record] = field(default_factory=list)
    policy: Policy | None = None

    def __post_init__(self) -> None:
        """Check that the scores, if any, are one per thunk.

        Raises:
            ValueError: If ``scores`` is not ``None`` and its length differs
                from that of ``thunks``.
        """
        if self.scores is not None and len(self.scores) != len(self.thunks):
            raise ValueError(
                f"{self.function}: {len(self.scores)} scores for "
                f"{len(self.thunks)} thunks"
            )

    def to_json(self) -> dict[str, Any]:
        """The record as written (the standard row), its nested calls inside it.

        Returns:
            Its fields as JSON-ready data: each thunk as its dataclass dict,
            ``scores`` as ``None`` or a list of ``Score``'s fields, ``calls``
            as their own rows, and ``policy`` as ``None`` or ``{"by",
            "completion", "meta"}``.
        """
        return {
            "function": self.function,
            "arguments": plain(self.arguments),
            "privileged": list(self.privileged),
            "thunks": [dataclasses.asdict(t) for t in self.thunks],
            "scores": None
            if self.scores is None
            else [dataclasses.asdict(s) for s in self.scores],
            "calls": [r.to_json() for r in self.calls],
            "policy": None
            if self.policy is None
            else {
                "by": self.policy.by,
                "completion": self.policy.output.value,
                "meta": plain(self.policy.output.meta),
            },
        }


_recorder: contextvars.ContextVar[Recorder | None] = contextvars.ContextVar(
    "mellea_learnable_recorder", default=None
)

Keep: TypeAlias = Callable[[int], bool]
"""Which of a function's calls a `Recorder` keeps: given the call's index among
that function's calls in the recording (0 for the first), whether to record it."""


def every(n: int) -> Keep:
    """Keep one call in `n`: the first, then every `n`-th after it.

    Args:
        n: How many calls each one kept stands for (1: every call).

    Returns:
        The rule, for `Recorder`'s (and `capture`'s) `keep`.

    Raises:
        ValueError: If `n` is not at least 1.
    """
    if n < 1:
        raise ValueError(f"every({n}): n is at least 1")
    return lambda i: i % n == 0


class Recorder:
    """Collects a `Record` for every learnable call made inside it.

    Use as a context manager. ``records`` holds the top-level calls, in the
    order they started; each holds the calls its body made (``Record.calls``).
    A call that raised leaves no record, nor do any of the calls it made.
    Recorders do not nest.

    ``keep`` records only some of a function's calls: it maps a learnable
    function to a rule (``every(n)``, or any ``Keep``), which sees the call's
    index among that function's calls since the recorder started. A function
    not in it is kept every time. A call not kept runs exactly as one kept does
    and returns its output; it leaves no record, and neither do the calls its
    body made.

    Each thunk is snapshot after the call, reading ``_call`` for the context
    and tools the backend actually used.

    Args:
        keep: Each function's rule.
    """

    def __init__(self, keep: dict[Learnable, Keep] | None = None) -> None:
        """Start with no records."""
        self.records: list[Record] = []
        # The calls still running, innermost last: each one's `calls`, filling.
        self._open: list[list[Record]] = [self.records]
        self._token: contextvars.Token | None = None
        self._keep = {fn.name: rule for fn, rule in (keep or {}).items()}
        self._seen: Counter[str] = Counter()  # each function's calls so far

    def _keeps(self, fn: Learnable) -> bool:
        """Whether this call of `fn` is recorded; counts it."""
        i = self._seen[fn.name]
        self._seen[fn.name] += 1
        rule = self._keep.get(fn.name)
        return rule is None or bool(rule(i))

    def __enter__(self) -> Recorder:
        """Start recording.

        Raises:
            RuntimeError: If another `Recorder` is already recording.
        """
        if _recorder.get() is not None:
            raise RuntimeError("Recorders do not nest")
        self._token = _recorder.set(self)
        return self

    def __exit__(self, *exc: Any) -> None:
        """Stop recording."""
        assert self._token is not None
        _recorder.reset(self._token)
        self._token = None


@dataclass(frozen=True)
class _Spec:
    """A learnable function's parameters, sorted by who sees them."""

    owner: str | None  # `self`, on a method
    context: str | None
    structured: tuple[str, ...]
    privileged: tuple[str, ...]


@dataclass(frozen=True)
class _Output:
    """A learnable function's output type, as the text a student writes.

    ``kind="choice"`` is an internal classification: the function returns one
    of a fixed set of strings (a ``Literal``).  At serving time this is
    implemented by Mellea's single-token adapter infrastructure (the
    ``first_token`` transformation rule, shared with probes and the decision
    helpers), not by a separate choice mechanism.  ``io_config()`` emits the
    ``first_token`` declaration; ``parse()`` validates the output against the
    choices at training/recording time.
    """

    kind: Literal["text", "choice"]
    choices: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Installed:
    """A policy installed on a learnable function (`Learnable.policy`)."""

    student: Student
    label: bool
    by: str  # its name in records (`Policy.by`)


class Learnable:
    """A function marked `@learnable` (see the module's docstring).

    Called, it runs its policy (the body, or inside `policy` the student given)
    and returns the policy's first output parsed; inside a `Recorder`, it also
    leaves a `Record`.

    Args:
        fn: The function (or method) marked.
        privileged: Its privileged parameters.
    """

    def __init__(self, fn: Callable, privileged: tuple[str, ...] = ()) -> None:
        """Wrap `fn`, its `privileged` parameters named."""
        if inspect.iscoroutinefunction(fn):
            raise TypeError("@learnable takes a synchronous function")
        self._fn = fn
        self.name: str = fn.__qualname__
        self._privileged = tuple(privileged)
        self._method = False
        self._installed: _Installed | None = None
        functools.update_wrapper(self, fn)

    def __set_name__(self, owner: type, name: str) -> None:
        """Defined in a class body: a method, its first parameter `self`."""
        self._method = True

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        """On an instance, the function bound to it."""
        return self if obj is None else types.MethodType(self, obj)

    @functools.cached_property
    def spec(self) -> _Spec:
        """Its parameters, sorted by who sees them.

        The owner (`self`), the context argument, the structured arguments and
        the privileged ones; resolved at first use, so annotations may name
        classes defined after the function.

        Raises:
            TypeError: If a privileged parameter does not exist, if there is
                more than one context argument, or if the context argument is
                privileged.
        """
        params = list(resolve_signature_annotations(self._fn).parameters.values())
        owner = params.pop(0).name if self._method else None
        names = [p.name for p in params]
        unknown = [n for n in self._privileged if n not in names]
        if unknown:
            raise TypeError(f"{self.name}: no parameter {unknown} to make privileged")
        contexts = [
            p.name
            for p in params
            if isinstance(p.annotation, type) and issubclass(p.annotation, Context)
        ]
        if len(contexts) > 1:
            raise TypeError(f"{self.name}: more than one context argument {contexts}")
        context = contexts[0] if contexts else None
        if context in self._privileged:
            raise TypeError(f"{self.name}: the context argument cannot be privileged")
        structured = tuple(
            n for n in names if n != context and n not in self._privileged
        )
        return _Spec(owner, context, structured, self._privileged)

    @functools.cached_property
    def interface(self) -> str:
        """The hash of its interface.

        The signature without `self` and the privileged parameters, and the
        docstring. The body is not in it.
        """
        sig = inspect.signature(self._fn)
        hidden = {self.spec.owner, *self.spec.privileged}
        visible = [p for n, p in sig.parameters.items() if n not in hidden]
        text = f"{sig.replace(parameters=visible)}\n{inspect.getdoc(self._fn) or ''}"
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    @functools.cached_property
    def _output(self) -> _Output:
        """Its output type, read from the return annotation at first use.

        Only the return annotation is resolved, so a parameter typed with a
        name that exists only for the type checker does not stop it.

        Raises:
            TypeError: If the return annotation does not resolve, or is neither
                text (`str`, or none) nor a `Literal` of strings.
        """
        ret = inspect.signature(self._fn).return_annotation
        if ret is inspect.Signature.empty:
            ret = str
        elif isinstance(ret, str):  # postponed (PEP 563): resolve it alone
            target = inspect.unwrap(self._fn)
            only = types.SimpleNamespace(__annotations__={"return": ret})
            try:
                ret = typing.get_type_hints(
                    only, globalns=getattr(target, "__globals__", {})
                )["return"]
            except Exception as e:
                raise TypeError(
                    f"{self.name}: its return annotation {ret!r} does not resolve: {e}"
                ) from e
        if ret is str:
            return _Output("text")
        args = typing.get_args(ret)
        if typing.get_origin(ret) is Literal and all(isinstance(a, str) for a in args):
            return _Output("choice", choices=args)
        raise TypeError(
            f"{self.name}: returns {ret!r}; a learnable function returns text "
            f"(str) or a Literal of strings. A structured result is a tool call, "
            f"not JSON in the content."
        )

    def parse(self, text: str) -> str:
        """The value an output's text stands for: the text, checked.

        Text (`str`, or no annotation) is the value itself; a `Literal` is the
        text, checked against its values.

        Args:
            text: An output's text (`ComputedModelOutputThunk.value`).

        Returns:
            The text.

        Raises:
            ValueError: If the text is not one of a `Literal`'s values.
        """
        out = self._output
        if out.kind == "choice" and text not in out.choices:
            raise ValueError(f"{self.name}: {text!r} is not one of {out.choices}")
        return text

    def io_config(self) -> dict[str, Any]:
        """The ``io.yaml`` a trained adapter of this function would declare.

        A text function (``-> str``) declares ``output: {type: text}`` and no
        transformations.  A choice function (``-> Literal["a", "b", ...]``)
        declares a ``first_token`` transformation with the choices as labels —
        the same one-token readout the probe and decision infrastructure uses.
        The caller adds what depends on the tokenizer and the deployment
        (``parameters``, ``system_prompt``, etc.).

        Returns:
            The configuration, ready for ``yaml.safe_dump``.
        """
        out = self._output
        if out.kind == "choice":
            return {
                "model": None,
                "response_format": None,
                "transformations": [
                    {"type": "first_token", "labels": list(out.choices)}
                ],
            }
        return {
            "model": None,
            "response_format": None,
            "transformations": None,
            "output": {"type": "text"},
        }

    @contextmanager
    def policy(
        self, student: Student | None, *, label: bool = False, name: str | None = None
    ) -> Iterator[None]:
        """Make `student` the policy for calls of this function inside the block.

        The policy's output is what each call returns, so the episode follows
        it. With `label=False` (inference: serving, evaluation, a scripted test
        double) the body does not run, and a call's record holds the student's
        outputs. With `label=True` (on-policy data) the body still runs on
        every call, first, as the teacher: it never sees the student's output,
        and its outputs and scores are the record's, the labels; the student's
        output is returned and recorded in `Record.policy`. A teacher with
        memory between calls (`self`) steps on every call either way.

        Args:
            student: Given a call's visible arguments (a dict from parameter
                name to value, privileged keys excluded), returns what a body
                returns (see the module's docstring); `None`: no change (the
                body stays the policy), so one driver serves every round.
            label: Whether the body still labels each call.
            name: The policy's name in records (default: the student's
                `__name__`, or its type's).

        Yields:
            Nothing: calls made inside the block answer with `student`.

        Raises:
            RuntimeError: If this function already has a policy installed.
        """
        if student is None:
            yield
            return
        if self._installed is not None:
            raise RuntimeError(f"{self.name}: a policy is already installed")
        by = name or getattr(student, "__name__", None) or type(student).__name__
        self._installed = _Installed(student, label, by)
        try:
            yield
        finally:
            self._installed = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run the call's policy (and teacher), recorded inside a `Recorder`.

        Returns:
            The policy's first output's text, checked (`parse`).

        Raises:
            TypeError: If the body or the student returns anything but a
                non-empty list of `ComputedModelOutputThunk`, or that list and
                a list of `Score`.
            ValueError: If a first output is not one of a `Literal`'s values,
                or the scores are not one per output.
        """
        bound = inspect.signature(self._fn).bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        if self.spec.owner is not None:
            arguments.pop(self.spec.owner)
        recorder = _recorder.get()
        if recorder is not None and recorder._keeps(self):
            return self._recorded(recorder, bound, arguments)
        # Not recorded: with no recorder, or a call the recorder does not keep,
        # whose nested calls are not recorded either.
        if recorder is not None:
            recorder._open.append([])
        try:
            labels, answer = self._run(bound, arguments)
        finally:
            if recorder is not None:
                recorder._open.pop()
        return (answer or labels)[0][0].value

    def _run(
        self, bound: inspect.BoundArguments, arguments: dict[str, Any]
    ) -> tuple[Any, Any]:
        """The teacher's outputs and the student's, each checked, or ``None``.

        The body runs unless a student answers alone.  The student receives the
        visible arguments dict (privileged keys excluded, Context objects
        intact).
        """
        installed = self._installed
        labels = None
        if installed is None or installed.label:
            labels = self._outputs(self._fn(*bound.args, **bound.kwargs))
        answer = None
        if installed is not None:
            visible = {
                n: arguments[n] for n in arguments if n not in self.spec.privileged
            }
            answer = self._outputs(installed.student(visible))
        return labels, answer

    def _recorded(
        self, recorder: Recorder, bound: inspect.BoundArguments, arguments: dict
    ) -> Any:
        calls: list[Record] = []
        recorder._open.append(calls)
        try:
            labels, answer = self._run(bound, arguments)
        finally:
            recorder._open.pop()
        completions, scores = labels or answer
        thunks = [_snapshot(c) for c in completions]
        policy = None
        if answer is not None:
            policy = Policy(by=self._installed.by, output=answer[0][0])  # type: ignore[union-attr]
        recorder._open[-1].append(
            Record(
                function=self.name,
                arguments={n: arguments[n] for n in arguments},
                privileged=self.spec.privileged,
                thunks=thunks,
                scores=scores,
                calls=calls,
                policy=policy,
            )
        )
        return (answer or labels)[0][0].value

    def _outputs(
        self, out: Any
    ) -> tuple[list[ComputedModelOutputThunk], list[Score] | None]:
        """A body's or a student's return, checked, its first output's text too."""
        scores = None
        if isinstance(out, tuple) and len(out) == 2:
            out, scores = out
            if not isinstance(scores, list) or not all(
                isinstance(s, Score) for s in scores
            ):
                raise TypeError(f"{self.name}: scores are a list of Score")
        if (
            not isinstance(out, list)
            or not out
            or not all(isinstance(c, ComputedModelOutputThunk) for c in out)
        ):
            raise TypeError(
                f"{self.name}: a body (or a student standing in for it) returns a "
                f"non-empty list of ComputedModelOutputThunk, or (that list, its "
                f"scores); got {type(out).__name__}"
            )
        if scores is not None and len(scores) != len(out):
            raise ValueError(
                f"{self.name}: {len(scores)} scores for {len(out)} outputs"
            )
        self.parse(out[0].value)
        return out, scores


def learnable(fn: Callable | None = None, *, privileged: tuple[str, ...] = ()) -> Any:
    """Mark a function (or a method) as learnable.

    Its signature is the student's interface, its body the teacher (see the
    module's docstring).

    Use as `@learnable`, or `@learnable(privileged=("turn",))` to name the
    parameters the body gets and the student is never told. Privileged facts
    may steer the body and check its output; any fact the output states must be
    recoverable from the visible arguments.

    Args:
        fn: The function.
        privileged: Its privileged parameters.

    Returns:
        The learnable function (`Learnable`).
    """
    if fn is None:
        return lambda f: Learnable(f, privileged)
    return Learnable(fn, privileged)
