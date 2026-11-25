# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Learnable functions: functions a student can be trained to emulate.

A *learnable function* is an ordinary Python function marked `@learnable`.
Its signature is the student's interface; its body is the teacher, and can be
anything (model calls, code that plans, judges, repair loops). Learning it is
emulation: a student (an adapter) learns to produce what the body produces,
given what the signature lets it see. Learning prescribes no judge and no
strategy: how the body earns its output is the body's own business.

The signature:

* At most one parameter is typed as a `Context`: the **context argument**,
  the conversation the call takes place in. The others are **structured
  arguments**.
* **Privileged parameters** (`@learnable(privileged=("turn",))`) are facts the
  caller knows and the student is never told, given to the body to steer and
  check its output (a question's structured type behind the words heard; what
  someone really said before speech recognition). They are recorded apart from
  the arguments and never reach the learnable view. **The rule that comes with
  them:** they may steer and check, but any fact the output states must be
  recoverable from the visible arguments, or the student learns to invent it.
* On a method, `self` is the teacher's state between calls: hidden state, which
  the student has to infer from its context. It is never part of the view.
* The **return annotation is the output type**, and it is text: `str` (or no
  annotation), or a `Literal` of strings. The output is the text itself, a
  `Literal`'s checked against its values (`Learnable.parse`). Any other type
  raises `TypeError`: a structured result is a tool call, not JSON in the
  content. `Learnable.io_config` is the `io.yaml` a trained adapter of the
  function would declare.

The **learnable view** (`learnable_view`) is exactly what the student is trained
on and served with: the context argument's messages (OpenAI format) and the
tools it declares, then, if the function has structured arguments, one final
user message holding them as sorted-key JSON. Neither the docstring nor
anything privileged is in it. Its format is versioned (`VIEW`).

A body returns its outputs, not a value: a non-empty
`list[ComputedModelOutputThunk]`, or that list with a `Score` for each
(`(outputs, scores)`). The first output is what the call returned; the rest are
candidates not taken. Each output's text is its `value`; what the body notes
about it goes in its `meta`.

**Two roles in every call.** The *policy* produces the output the call returns,
which the caller uses, so the rest of the episode follows from it. The
*teacher* labels the call: its outputs and their scores are what training
learns. In a plain call the body is both. `fn.policy(student)` makes a student
the policy for the calls inside its block (given the learnable view, it returns
what a body returns): with no teacher, it is inference (serving, evaluation, a
scripted test double); with `label=True`, the body still runs and labels, on
the state the student's earlier outputs led to: on-policy data. The caller gets
the policy's first output's text (checked, for a `Literal`) on every path, so
the driver making the calls cannot tell who answered, nor whether a `Recorder`
was listening.

Called inside a `Recorder`, each call leaves a `Record`: the call's view (the
prompt), its outputs (the teacher's, or with no teacher the policy's), their
scores if given, who answered (`policy`), and the learnable calls its body
made (`calls`), so the calls of an episode form a tree. A recorder may keep
only some of a function's calls (`keep`). Outside a recorder, a learnable
function is a plain function call.
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

from ...core import ComputedModelOutputThunk, Context, ModelOutputThunk
from ...helpers.annotation_helpers import resolve_signature_annotations
from ...helpers.openai_compatible_helpers import message_to_openai_message


def _context_to_openai(ctx: Context) -> list[dict[str, Any]]:
    """Fallback for self-checks without a formatter: a direct conversion
    that may miss tool declarations."""
    from ..components.chat import Message

    out = []
    for c in ctx.as_list():
        if isinstance(c, ModelOutputThunk):
            c = Message("assistant", str(c.value or ""))
        if isinstance(c, Message):
            out.append(message_to_openai_message(c))
    return out

VIEW = 1
"""The learnable view's format version (recorded on every record)."""


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
        return _context_to_openai(value)
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


Returned: TypeAlias = (
    list[ComputedModelOutputThunk] | tuple[list[ComputedModelOutputThunk], list[Score]]
)
"""What a body, or a student standing in for it as the policy, returns: its
outputs, or its outputs and a score for each."""

Student: TypeAlias = Callable[[dict[str, Any]], Returned]
"""A policy other than the body: given a call's learnable view, it returns what a
body returns."""


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
        interface: The hash of its interface (`Learnable.interface`).
        view: The learnable view's format version (`VIEW`).
        args: Its visible arguments as passed (Python values).
        privileged: Its privileged arguments as passed.
        prompt: The learnable view: `{"messages", "tools"}`.
        completions: The call's outputs: the teacher's (the body's) when there
            is one, else the policy's; first the output it would return, then
            the candidates not taken.
        scores: `None`, or one `Score` per completion, in order: a call is
            scored whole or not at all.
        calls: The learnable calls its body made, in the order they started.
        policy: `None` when the body answered (a plain call); otherwise who
            did and what it returned. Under `fn.policy(student)` the
            completions are the student's too; under `label=True` they are the
            body's labels, and only `policy` holds what the student returned.
    """

    function: str
    interface: str
    view: int
    args: dict[str, Any]
    privileged: dict[str, Any]
    prompt: dict[str, Any]
    completions: list[ComputedModelOutputThunk]
    scores: list[Score] | None = None
    calls: list[Record] = field(default_factory=list)
    policy: Policy | None = None

    def __post_init__(self) -> None:
        """Check that the scores, if any, are one per completion.

        Raises:
            ValueError: If `scores` is not `None` and its length differs from
                that of `completions`.
        """
        if self.scores is not None and len(self.scores) != len(self.completions):
            raise ValueError(
                f"{self.function}: {len(self.scores)} scores for "
                f"{len(self.completions)} completions"
            )

    def to_json(self) -> dict[str, Any]:
        """The record as written (the standard row), its nested calls inside it.

        Returns:
            Its fields as JSON-ready data: each completion as `{"completion",
            "meta"}`, `scores` as `None` or a list of `Score`'s fields, `calls`
            as their own rows, and `policy` as `None` or `{"by", "completion",
            "meta"}`.
        """
        return {
            "function": self.function,
            "interface": self.interface,
            "view": self.view,
            "args": plain(self.args),
            "privileged": plain(self.privileged),
            "prompt": self.prompt,
            "completions": [
                {"completion": c.value, "meta": plain(c.meta)} for c in self.completions
            ],
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

    Use as a context manager. `records` holds the top-level calls, in the order
    they started; each holds the calls its body made (`Record.calls`). A call
    that raised leaves no record, nor do any of the calls it made. Recorders do
    not nest.

    `keep` records only some of a function's calls: it maps a learnable
    function to a rule (`every(n)`, or any `Keep`), which sees the call's index
    among that function's calls since the recorder started. A function not in
    it is kept every time. A call not kept runs exactly as one kept does and
    returns its output; it leaves no record, and neither do the calls its body
    made. Its learnable view is not rendered unless its policy needs it.

    The learnable view is rendered through the same path the backend uses —
    ``view_for_generation`` → ``to_chat_messages`` →
    ``message_to_openai_message`` — so it captures exactly what the model sees,
    including tools declared on the context's components.

    Args:
        formatter: The backend's formatter (``TemplateFormatter``), used to
            render the learnable view the same way the backend renders the
            prompt.  ``None`` for self-checks that test the learnable
            machinery without a backend; in that case the view falls back to
            a direct conversion that may miss tool declarations.
        keep: Each function's rule.
    """

    def __init__(
        self,
        formatter: Any = None,
        keep: dict[Learnable, Keep] | None = None,
    ) -> None:
        """Start with no records."""
        self.records: list[Record] = []
        # The calls still running, innermost last: each one's `calls`, filling.
        self._open: list[list[Record]] = [self.records]
        self._token: contextvars.Token | None = None
        self._keep = {fn.name: rule for fn, rule in (keep or {}).items()}
        self._seen: Counter[str] = Counter()  # each function's calls so far
        self.formatter = formatter

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
    choices at training/recording time."""

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
            student: Given a call's learnable view, returns what a body returns
                (see the module's docstring); `None`: no change (the body stays
                the policy), so one driver serves every round.
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
            labels, answer = self._run(bound, arguments, None)
        finally:
            if recorder is not None:
                recorder._open.pop()
        return (answer or labels)[0][0].value

    def _run(
        self,
        bound: inspect.BoundArguments,
        arguments: dict[str, Any],
        prompt: dict[str, Any] | None,
    ) -> tuple[Any, Any]:
        """The teacher's outputs and the student's, each checked, or `None`.

        The body runs unless a student answers alone; the view is rendered
        only for a student, from `prompt` when the caller has it already.
        """
        installed = self._installed
        labels = None
        if installed is None or installed.label:
            labels = self._outputs(self._fn(*bound.args, **bound.kwargs))
        answer = None
        if installed is not None:
            view = prompt if prompt is not None else learnable_view(self, arguments)
            answer = self._outputs(installed.student(view))
        return labels, answer

    def _recorded(
        self, recorder: Recorder, bound: inspect.BoundArguments, arguments: dict
    ) -> Any:
        prompt = learnable_view(self, arguments, recorder.formatter)  # as the student reads it, now
        calls: list[Record] = []
        recorder._open.append(calls)
        try:
            labels, answer = self._run(bound, arguments, prompt)
        finally:
            recorder._open.pop()
        completions, scores = labels or answer
        policy = None
        if answer is not None:
            policy = Policy(by=self._installed.by, output=answer[0][0])  # type: ignore[union-attr]
        recorder._open[-1].append(
            Record(
                function=self.name,
                interface=self.interface,
                view=VIEW,
                args={
                    n: arguments[n] for n in arguments if n not in self.spec.privileged
                },
                privileged={n: arguments[n] for n in self.spec.privileged},
                prompt=prompt,
                completions=completions,
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


def learnable_view(
    fn: Learnable, arguments: dict[str, Any], formatter: Any = None
) -> dict[str, Any]:
    """What the student reads for a call of `fn`: `{"messages", "tools"}`.

    The context argument's messages (OpenAI format) and the tools declared on
    its components; then, if `fn` has structured arguments, one final user
    message holding them as sorted-key JSON. No docstring, nothing privileged.
    Format `VIEW`.

    When ``formatter`` is given (the backend's ``TemplateFormatter``), the view
    is rendered through the same path the backend uses, capturing tools from
    the context's components.  Without one (self-checks), the view is a direct
    conversion that may miss tool declarations.

    Args:
        fn: The learnable function.
        arguments: The call's arguments by name (`self` not among them; any
            privileged ones are left out).
        formatter: The backend's formatter, or ``None`` for self-checks.

    Returns:
        The messages and the tools (`None` when the context declares none or
        no formatter was given).

    Raises:
        TypeError: If the context argument is not a `Context`, or a structured
            argument has no JSON form (`plain`).
    """
    spec = fn.spec
    messages: list[dict[str, Any]] = []
    tools = None
    if spec.context is not None:
        ctx = arguments[spec.context]
        if not isinstance(ctx, Context):
            raise TypeError(
                f"{fn.name}: its context argument takes a Context, "
                f"got {type(ctx).__name__}"
            )
        if formatter is not None:
            messages, tools = _render_via_formatter(ctx, formatter)
        else:
            messages = _context_to_openai(ctx)
    if spec.structured:
        args = {n: plain(arguments[n]) for n in spec.structured}
        messages = [
            *messages,
            {"role": "user", "content": json.dumps(args, sort_keys=True)},
        ]
    return {"messages": messages, "tools": tools}


def _render_via_formatter(ctx: Context, formatter: Any) -> tuple[list[dict], list[dict] | None]:
    """Render the context through the backend's formatter, extracting both
    messages and tools from the components — the same path the backend uses."""
    from ...backends.tools import add_tools_from_context_actions, convert_tools_to_json

    # Messages: linearize → format → serialize.
    linearized = ctx.view_for_generation()
    if linearized is None:
        linearized = ctx.as_list()
    chat_messages = formatter.to_chat_messages(linearized)
    messages = [message_to_openai_message(m) for m in chat_messages]

    # Tools: read from the components' TemplateRepresentations.
    tool_dict: dict = {}
    add_tools_from_context_actions(tool_dict, ctx.actions_for_available_tools())
    tools = convert_tools_to_json(tool_dict) if tool_dict else None

    return messages, tools
