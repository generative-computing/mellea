# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Learnable functions: adapter functions whose teacher is a Python body.

A *learnable function* is an ordinary Python function marked ``@learnable``.
It declares an `AdapterFunction` (``mellea.stdlib.components.intrinsic``): a
call to a model, what it reads, which tools it is served with, what it may
output. Its body is the teacher, and can be anything (model calls, an
instruct-validate-repair loop, an RL policy, code). Learning it is emulation: a
student (an adapter) learns to produce what the body produces, given the call.
Learning prescribes no judge and no strategy: how the body earns its output is
the body's own business.

The signature is the call:

* At most one parameter is typed as a ``Context``: the **context argument**,
  the conversation the call takes place in (the call's ``ctx``). The others
  are **structured arguments**: the visible ones are one final user turn, as
  sorted-key JSON.
* ``@learnable(tools=[get_order_status])``: the tools the call is served with
  (``model_options[ModelOption.TOOLS]``, ``tool_calls=True``,
  ``tool_choice="none"``).
* **Privileged parameters** (``@learnable(privileged=("turn",))``) are facts
  the caller knows and the student is never told, given to the body to steer
  and check its output. They are not in the call. **The rule that comes with
  them:** they may steer and check, but any fact the output states must be
  recoverable from the visible arguments, or the student learns to invent it.
* On a method, ``self`` is the teacher's state between calls: hidden state,
  which the student has to infer from its context. It is never recorded.
* The **return annotation is the output**: ``str`` (or no annotation), a
  ``Literal`` of strings (one of them, read from the first token), or a
  Pydantic model (JSON matching its schema). ``Learnable.io_config`` is the
  ``io.yaml`` a trained adapter of the function declares.

A body returns its outputs, not a value: a non-empty
``list[ComputedModelOutputThunk]``, or that list with a ``Score`` for each
(``(outputs, scores)``). The first output is what the call returned; the rest
are candidates not taken. Each output's text is its ``value``; what the body
notes about it goes in its ``meta``.

**The student's input is the call, rendered.** Inside a ``Recorder`` with a
student backend (``Recorder(student=backend)``, ``capture(..., student=)``),
each call is rendered by that backend (``AdapterFunction.render``, the same
rendering its generate path sends) *before* the body runs: the record's
``request``, what ``sft_rows`` and ``preference_rows`` train on. The body's
own model calls do not enter it, so a teacher may generate from anything (its
own instructions, privileged facts) or make no model call at all. Each output
is also kept as a ``Thunk`` snapshot, the teacher's trace: the context and
tools its backend used (empty for an output no backend produced), the full
assistant response, the model and latency.

**Two roles in every call.** The *policy* produces the output the call returns,
which the caller uses, so the rest of the episode follows from it. The
*teacher* labels the call: its outputs and their scores are what training
learns. In a plain call the body is both. ``fn.policy(student)`` makes a
student the policy for the calls inside its block (given the visible arguments,
it returns what a body returns); ``fn.policy(adapter_student(backend))`` serves
the call itself through the backend's adapter-function path. With no teacher,
it is inference (serving, evaluation, a scripted test double); with
``label=True``, the body still runs and labels, on the state the student's
earlier outputs led to: on-policy data. The caller gets the policy's first
output, parsed, on every path, so the driver making the calls cannot tell who
answered, nor whether a ``Recorder`` was listening.

Called inside a ``Recorder``, each call leaves a ``Record``: the call's
arguments, its rendered request, its thunks (the teacher's snapshots, or with
no teacher the policy's), their scores if given, who answered (``policy``), and
the learnable calls its body made (``calls``), so the calls of an episode form
a tree. A recorder may keep only some of a function's calls (``keep``).
Outside a recorder, a learnable function is a plain function call.
"""

from __future__ import annotations

import contextvars
import dataclasses
import functools
import hashlib
import inspect
import json
import types
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, TypeAlias

from ...core import (
    Backend,
    ComputedModelOutputThunk,
    Context,
    ModelOutputThunk,
    RenderedRequest,
)
from ...core.base import AbstractMelleaTool
from ...helpers.annotation_helpers import resolve_signature_annotations
from ...helpers.openai_compatible_helpers import build_tool_calls
from ..components.intrinsic.adapter_function import (
    AdapterFunction,
    OutputSpec,
    Spec,
    _spans_to_openai,
    plain,
    resolve_return,
)


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
            policy's when there is no teacher): the teacher's trace.
        scores: ``None``, or one `Score` per thunk, in order: a call is
            scored whole or not at all.
        calls: The learnable calls its body made, in the order they started.
        policy: ``None`` when the body answered (a plain call); otherwise who
            did and what it returned.
        request: The call as the recorder's student backend renders it,
            before the body ran: the student's input, what training reads.
            ``None`` when recorded without a student backend.
    """

    function: str
    arguments: dict[str, Any]
    privileged: tuple[str, ...]
    thunks: list[Thunk]
    scores: list[Score] | None = None
    calls: list[Record] = field(default_factory=list)
    policy: Policy | None = None
    request: RenderedRequest | None = None

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
            Its fields as JSON-ready data: ``request`` as ``None`` or its
            fields (``messages``, ``tools``, ``tool_choice``, ``constraint``,
            ``adapter``), each thunk as its dataclass dict, ``scores`` as
            ``None`` or a list of ``Score``'s fields, ``calls`` as their own
            rows, and ``policy`` as ``None`` or ``{"by", "completion",
            "meta"}``.
        """
        return {
            "function": self.function,
            "arguments": plain(self.arguments),
            "privileged": list(self.privileged),
            "request": None if self.request is None else self.request.to_json(),
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

    ``student`` is the backend whose rendering defines the student's input
    (the base model's chat template): each call kept is rendered by it before
    its body runs (``Record.request``). Without one, records hold no request,
    and training cannot read them (``sft_rows``). Each thunk is snapshot after
    the call, reading ``_call`` for the context and tools its backend used.

    Args:
        keep: Each function's rule.
        student: The backend that renders each call, or ``None``.
    """

    def __init__(
        self,
        keep: dict[Learnable, Keep] | None = None,
        *,
        student: Backend | None = None,
    ) -> None:
        """Start with no records."""
        self.records: list[Record] = []
        self.student = student
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
class _Installed:
    """A policy installed on a learnable function (`Learnable.policy`)."""

    student: Student
    label: bool
    by: str  # its name in records (`Policy.by`)


class Learnable(AdapterFunction):
    """A function marked `@learnable`: an `AdapterFunction` whose teacher is its body.

    See the module's docstring. Called, it runs its policy (the body, or
    inside `policy` the student given) and returns the policy's first output
    parsed; inside a `Recorder`, it also leaves a `Record`.

    Args:
        fn: The function (or method) marked.
        privileged: Its privileged parameters.
        tools: The tools its call is served with.
    """

    def __init__(
        self,
        fn: Callable,
        privileged: tuple[str, ...] = (),
        tools: Sequence[AbstractMelleaTool] = (),
    ) -> None:
        """Wrap `fn`, its `privileged` parameters named, served with `tools`."""
        if inspect.iscoroutinefunction(fn):
            raise TypeError("@learnable takes a synchronous function")
        super().__init__(fn.__qualname__, tools=tools, teacher=fn)
        self._fn = fn
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
    def spec(self) -> Spec:  # type: ignore[override]
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
        return Spec(context, structured, self._privileged, owner)

    @functools.cached_property
    def interface(self) -> str:
        """The hash of its interface.

        The signature without `self` and the privileged parameters, the
        docstring, and the tools' schemas. The body is not in it.
        """
        sig = inspect.signature(self._fn)
        hidden = {self.spec.owner, *self.spec.privileged}
        visible = [p for n, p in sig.parameters.items() if n not in hidden]
        tools = json.dumps([t.as_json_tool for t in self.tools], sort_keys=True)
        text = (
            f"{sig.replace(parameters=visible)}\n{inspect.getdoc(self._fn) or ''}"
            f"\n{tools}"
        )
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    @functools.cached_property
    def output(self) -> OutputSpec:  # type: ignore[override]
        """Its output, read from the return annotation at first use.

        Raises:
            TypeError: If the return annotation does not resolve, or is not
                text (`str`, or none), a `Literal` of strings, or a Pydantic
                model.
        """
        return OutputSpec.of(resolve_return(self._fn, self.name), self.name)

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
                returns (see the module's docstring); or a student with a
                `bind(fn)` method, given this function first to return that
                callable (`adapter_student`, which serves the call itself);
                `None`: no change (the body stays the policy), so one driver
                serves every round.
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
        bind = getattr(student, "bind", None)
        if callable(bind):  # a student of the function itself (adapter_student)
            student = bind(self)
        by = name or getattr(student, "__name__", None) or type(student).__name__
        self._installed = _Installed(student, label, by)
        try:
            yield
        finally:
            self._installed = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run the call's policy (and teacher), recorded inside a `Recorder`.

        Returns:
            The policy's first output, parsed (`parse`): its text, or for a
            JSON output the Pydantic model.

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
        return self.parse((answer or labels)[0][0].value)

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
            labels = self._teach(bound)
        answer = None
        if installed is not None:
            visible = {
                n: arguments[n] for n in arguments if n not in self.spec.privileged
            }
            answer = self._outputs(installed.student(visible))
        return labels, answer

    def _teach(
        self, bound: inspect.BoundArguments
    ) -> tuple[list[ComputedModelOutputThunk], list[Score] | None]:
        """The body's outputs, checked; fires `adapter_function_invocation_complete`.

        The teacher is an implementation of the adapter function, so its
        calls are reported as the adapter's are (`binding_type="teacher"`).
        """
        from ...backends.adapters.adapter import _fire_invocation_complete

        error: BaseException | None = None
        try:
            return self._outputs(self._fn(*bound.args, **bound.kwargs))
        except BaseException as e:
            error = e
            raise
        finally:
            _fire_invocation_complete(
                name=self.name,
                revision=None,
                binding_type="teacher",
                adapter_type="teacher",
                outcome="success" if error is None else "error",
                error=error,
            )

    def _recorded(
        self, recorder: Recorder, bound: inspect.BoundArguments, arguments: dict
    ) -> Any:
        request = None
        if recorder.student is not None:  # the student's input, before the body
            request = self.render(recorder.student, arguments)
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
                request=request,
            )
        )
        return self.parse((answer or labels)[0][0].value)

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


def learnable(
    fn: Callable | None = None,
    *,
    privileged: tuple[str, ...] = (),
    tools: Sequence[AbstractMelleaTool] = (),
) -> Any:
    """Mark a function (or a method) as learnable: an adapter function its body teaches.

    Its signature declares the call, its body is the teacher (see the
    module's docstring).

    Use as `@learnable`, or `@learnable(tools=[get_order_status],
    privileged=("turn",))` to name the tools the call is served with and the
    parameters the body gets and the student is never told. Privileged facts
    may steer the body and check its output; any fact the output states must
    be recoverable from the visible arguments.

    Args:
        fn: The function.
        privileged: Its privileged parameters.
        tools: The tools its call is served with.

    Returns:
        The learnable function (`Learnable`).
    """
    if fn is None:
        return lambda f: Learnable(f, privileged, tools)
    return Learnable(fn, privileged, tools)


@dataclass(frozen=True)
class AdapterStudent:
    """A policy that serves a learnable function's call through a backend's adapter.

    `fn.policy(adapter_student(backend))`: each call is `fn.call(arguments)`,
    sent through the backend's adapter-function path
    (`act(Intrinsic(name), ctx, model_options=..., tool_calls=...)`), so the
    trained adapter answers exactly the call the records rendered. The
    backend must have the adapter registered (e.g. a Granite Switch checkpoint
    served with `load_embedded_adapters=True`).

    Args:
        backend: The backend serving the adapter.
        name: The adapter's name in the checkpoint, when it is not the
            function's (`fn.name`).
        model_options: Model options over the adapter's `io.yaml` defaults.
    """

    backend: Backend
    name: str | None = None
    model_options: dict[str, Any] | None = None

    def bind(self, fn: AdapterFunction) -> Student:
        """The policy for `fn`'s calls.

        Args:
            fn: The adapter function served.

        Returns:
            A `Student`: given a call's visible arguments, the adapter's
            output.
        """
        from ...stdlib import functional as mfuncs
        from ..components.intrinsic import Intrinsic

        backend, served = self.backend, self.name or fn.name
        options = dict(self.model_options or {})

        def student(visible: dict[str, Any]) -> Returned:
            action, ctx, model_options = fn.call(visible)
            resolve = getattr(backend, "resolve_adapter", None)
            if resolve is not None:
                resolve(served)
            mot, _ = mfuncs.act(
                Intrinsic(served, adapter_types=action.adapter_types),
                ctx,
                backend,
                model_options={**model_options, **options},
                tool_calls=fn.tool_calls,
                strategy=None,
            )
            return [mot]

        student.__name__ = f"adapter:{served}"
        return student


def adapter_student(
    backend: Backend,
    *,
    name: str | None = None,
    model_options: dict[str, Any] | None = None,
) -> AdapterStudent:
    """The trained adapter, served by `backend`, as a learnable function's policy.

    Install it with `fn.policy(adapter_student(backend))`; see `AdapterStudent`.

    Args:
        backend: The backend serving the adapter.
        name: The adapter's name in the checkpoint, when it is not the
            function's.
        model_options: Model options over the adapter's `io.yaml` defaults.

    Returns:
        The policy.
    """
    return AdapterStudent(backend, name, model_options)
