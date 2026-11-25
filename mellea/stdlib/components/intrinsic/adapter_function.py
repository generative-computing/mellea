# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`AdapterFunction`: a named capability that behaves like a model.

Every Mellea construct reaches a model through one call,
`Backend.generate_from_context(action, ctx, *, format, model_options,
tool_calls)`. An adapter is expected to behave like a model, so an *adapter
function* is one such call, declared: what it reads (a context, and structured
arguments), which tools it is served with, and what it may output (text, one of
a set of labels, or JSON). `AdapterFunction.call` turns arguments into that
call; a backend's `render_request` renders it, and its generate path serves it.

The call, from the declaration:

| declared by | in the generate call |
|---|---|
| the context argument | `ctx` |
| the function itself | `action`: `Intrinsic(name)`, its `io_config` attached |
| visible structured arguments | one final user turn, sorted-key JSON, appended to `ctx` |
| `tools` | `model_options[ModelOption.TOOLS]`, `tool_calls=True` |
| the output | its `io.yaml`: text, `first_token` labels, or a JSON schema |
| (from `tools`) | `tool_choice`: `"none"` (no adapter function returns tool calls yet) |

Privileged arguments are not in the call: only a teacher sees them.

A *teacher* is an implementation of the call that is not the adapter (another
model, a loop, a policy, code); the *adapter* is the trained weights, served by
a backend. Both are optional: a catalog intrinsic has an adapter and no
teacher, a learnable function (`mellea.stdlib.learnable`) a teacher and, once
trained, an adapter.
"""

from __future__ import annotations

import dataclasses
import enum
import inspect
import json
import types
import typing
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from ....backends.adapters.catalog import AdapterType
from ....backends.model_options import ModelOption
from ....core import Backend, Context, ModelOutputThunk, RenderedRequest, Span
from ....core.base import AbstractMelleaTool
from ....helpers.openai_compatible_helpers import (
    build_tool_calls,
    message_to_openai_message,
)
from .intrinsic import Intrinsic


def _spans_to_openai(spans: list[Span]) -> list[dict[str, Any]]:
    """A flat span list as OpenAI messages.

    `Message` spans are serialized via `message_to_openai_message`;
    `ModelOutputThunk` spans become assistant messages with content,
    tool_calls, and reasoning_content when present. Other spans are skipped.
    """
    from ..chat import Message

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
        TypeError: For any other kind of object (a call's input must be
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


@dataclass(frozen=True)
class Spec:
    """An adapter function's parameters, sorted by who sees them.

    Args:
        context: The context argument's name, or `None`.
        structured: The visible structured arguments' names, in order.
        privileged: The privileged arguments' names: a teacher's only.
        owner: `self`'s name, on a method (the teacher's state), or `None`.
    """

    context: str | None
    structured: tuple[str, ...] = ()
    privileged: tuple[str, ...] = ()
    owner: str | None = None


@dataclass(frozen=True)
class OutputSpec:
    """What an adapter function may output.

    `text`: free text. `labels`: one of a fixed set of strings, read from the
    first token (the `first_token` transformation shared with probes and the
    decision helpers). `json`: a JSON object matching `schema`.

    Args:
        kind: The kind of output.
        labels: The labels, for `labels`.
        schema: The Pydantic model, for `json`.
    """

    kind: Literal["text", "labels", "json"]
    labels: tuple[str, ...] = ()
    schema: type[BaseModel] | None = None

    @classmethod
    def of(cls, annotation: Any, owner: str = "the function") -> OutputSpec:
        """The output a return annotation declares.

        Args:
            annotation: The return annotation (resolved): `str` (or
                `inspect.Signature.empty`), a `Literal` of strings, or a
                Pydantic model.
            owner: The function's name, for the error.

        Returns:
            The output spec.

        Raises:
            TypeError: For any other annotation.
        """
        if annotation is inspect.Signature.empty or annotation is str:
            return cls("text")
        args = typing.get_args(annotation)
        if typing.get_origin(annotation) is Literal and all(
            isinstance(a, str) for a in args
        ):
            return cls("labels", labels=args)
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return cls("json", schema=annotation)
        raise TypeError(
            f"{owner}: returns {annotation!r}; an adapter function returns text "
            f"(str), a Literal of strings, or a Pydantic model."
        )

    def parse(self, text: str, owner: str = "the function") -> Any:
        """The value an output's text stands for.

        Args:
            text: The output's text.
            owner: The function's name, for the error.

        Returns:
            The text (`text`, `labels`), or the Pydantic model it holds
            (`json`).

        Raises:
            ValueError: If the text is not one of the labels, or not JSON
                matching the schema.
        """
        if self.kind == "labels" and text not in self.labels:
            raise ValueError(f"{owner}: {text!r} is not one of {self.labels}")
        if self.kind == "json":
            assert self.schema is not None
            try:
                return self.schema.model_validate_json(text)
            except ValidationError as e:
                raise ValueError(
                    f"{owner}: {text!r} is not a {self.schema.__name__}: {e}"
                ) from e
        return text

    def io_fields(self) -> dict[str, Any]:
        """Its `io.yaml` fields: `response_format`, `transformations`, `output`.

        Returns:
            The fields.
        """
        if self.kind == "labels":
            return {
                "response_format": None,
                "transformations": [
                    {"type": "first_token", "labels": list(self.labels)}
                ],
            }
        if self.kind == "json":
            assert self.schema is not None
            return {
                "response_format": json.dumps(self.schema.model_json_schema()),
                "transformations": None,
            }
        return {
            "response_format": None,
            "transformations": None,
            "output": {"type": "text"},
        }


def _tool_schemas(tools: Sequence[AbstractMelleaTool]) -> list[dict[str, Any]]:
    """The tools' OpenAI-format JSON schemas, in order."""
    return [t.as_json_tool for t in tools]


class AdapterFunction:
    """A named capability that behaves like a model: a declared generate call.

    Args:
        name: The adapter's name in the checkpoint (its `io.yaml` `name`).
        spec: Its parameters, sorted by who sees them.
        output: What it may output.
        tools: The tools it is served with.
        sampling: Its `io.yaml` `parameters` (sampling), or `None`.
        teacher: An implementation of the call that is not the adapter, or
            `None`.
    """

    def __init__(
        self,
        name: str,
        *,
        spec: Spec | None = None,
        output: OutputSpec | None = None,
        tools: Sequence[AbstractMelleaTool] = (),
        sampling: dict[str, Any] | None = None,
        teacher: Callable | None = None,
    ) -> None:
        """Declare an adapter function named `name`."""
        self.name = name
        self._spec = spec if spec is not None else Spec(context=None)
        self._output_spec = output if output is not None else OutputSpec("text")
        self.tools: list[AbstractMelleaTool] = list(tools)
        self.sampling: dict[str, Any] = dict(sampling or {})
        self.teacher = teacher

    @property
    def spec(self) -> Spec:
        """Its parameters, sorted by who sees them."""
        return self._spec

    @property
    def output(self) -> OutputSpec:
        """What it may output."""
        return self._output_spec

    @property
    def tool_calls(self) -> bool:
        """Whether its call exposes tools (it declares some)."""
        return bool(self.tools)

    @property
    def tool_choice(self) -> str | None:
        """Its `tool_choice`: `"none"` with tools (it returns no tool calls), else `None`."""
        return "none" if self.tools else None

    def call(self, arguments: dict[str, Any]) -> tuple[Intrinsic, Context, dict]:
        """The generate call for these arguments: `(action, ctx, model_options)`.

        Pass it on with `tool_calls=self.tool_calls`. Privileged arguments, if
        given, are ignored.

        Args:
            arguments: The call's arguments by name: the context argument
                and the structured ones (privileged ones may be among them).

        Returns:
            The action (`Intrinsic(name)`, this function's `io_config`
            attached), the context (the context argument, then one user turn
            holding the structured arguments as sorted-key JSON, if there are
            any), and the model options (its tools).

        Raises:
            KeyError: If the context argument or a structured argument is
                missing.
            TypeError: If the context argument is not a `Context`, or a
                structured argument has no JSON form (`plain`).
        """
        from ...context import ChatContext
        from ..chat import Message

        spec = self.spec
        ctx: Context
        if spec.context is not None:
            ctx = arguments[spec.context]
            if not isinstance(ctx, Context):
                raise TypeError(
                    f"{self.name}: its context argument takes a Context, "
                    f"got {type(ctx).__name__}"
                )
        else:
            ctx = ChatContext()
        if spec.structured:
            visible = {n: plain(arguments[n]) for n in spec.structured}
            ctx = ctx.add(Message("user", json.dumps(visible, sort_keys=True)))
        action = Intrinsic(
            self.name,
            adapter_types=(AdapterType.ALORA, AdapterType.LORA),
            io_config=self.io_config(),
        )
        model_options: dict[str, Any] = {}
        if self.tools:
            model_options[ModelOption.TOOLS] = list(self.tools)
        return action, ctx, model_options

    def render(self, backend: Backend, arguments: dict[str, Any]) -> RenderedRequest:
        """The request `backend` sends for this call: what the adapter receives.

        Args:
            backend: The backend whose rendering defines the input (the base
                model's chat template).
            arguments: The call's arguments by name (see `call`).

        Returns:
            The rendered request.
        """
        action, ctx, model_options = self.call(arguments)
        return backend.render_request(
            action, ctx, model_options=model_options, tool_calls=self.tool_calls
        )

    def parse(self, text: str) -> Any:
        """The value an output's text stands for (`OutputSpec.parse`).

        Args:
            text: An output's text.

        Returns:
            The text, or for a JSON output the Pydantic model it holds.
        """
        return self.output.parse(text, self.name)

    def io_config(self) -> dict[str, Any]:
        """Its `io.yaml`: name, output, transformations, tools, tool_choice, sampling.

        Returns:
            The configuration, ready for `yaml.safe_dump`. `tools` and
            `tool_choice` appear when it declares tools, `parameters` when it
            declares sampling.
        """
        config: dict[str, Any] = {"name": self.name, "model": None}
        config.update(self.output.io_fields())
        if self.tools:
            config["tools"] = _tool_schemas(self.tools)
            config["tool_choice"] = self.tool_choice
        if self.sampling:
            config["parameters"] = dict(self.sampling)
        return config


def resolve_return(fn: Callable, owner: str) -> Any:
    """`fn`'s return annotation, resolved alone (PEP 563 strings included).

    Only the return annotation is resolved, so a parameter typed with a name
    that exists only for the type checker does not stop it.

    Args:
        fn: The function.
        owner: Its name, for the error.

    Returns:
        The annotation, `inspect.Signature.empty` if there is none.

    Raises:
        TypeError: If it does not resolve.
    """
    ret = inspect.signature(fn).return_annotation
    if isinstance(ret, str):
        target = inspect.unwrap(fn)
        only = types.SimpleNamespace(__annotations__={"return": ret})
        try:
            ret = typing.get_type_hints(
                only, globalns=getattr(target, "__globals__", {})
            )["return"]
        except Exception as e:
            raise TypeError(
                f"{owner}: its return annotation {ret!r} does not resolve: {e}"
            ) from e
    return ret
