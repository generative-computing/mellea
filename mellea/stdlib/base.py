"""Basic stdlib data structures."""

from __future__ import annotations

import abc
import base64
import binascii
import datetime
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Protocol, runtime_checkable

from PIL import Image as PILImage

from mellea.helpers.fancy_logger import FancyLogger


class CBlock:
    """A `CBlock` is a block of content that can serve as input to or output from an LLM."""

    def __init__(self, value: str | None, meta: dict[str, Any] | None = None):
        """Initializes the CBlock with a string and some metadata."""
        if value is not None and not isinstance(value, str):
            raise TypeError("value to a Cblock should always be a string or None")
        self._underlying_value = value
        if meta is None:
            meta = {}
        self._meta = meta

    @property
    def value(self) -> str | None:
        """Gets the value of the block."""
        return self._underlying_value

    @value.setter
    def value(self, v: str):
        """Sets the value of the block."""
        self._underlying_value = v

    def __str__(self):
        """Stringifies the block."""
        return self.value if self.value else ""

    def __repr__(self):
        """Provides a python-parsable representation of the block (usually)."""
        return f"CBlock({self.value}, {self._meta.__repr__()})"


class ImageBlock:
    """A `ImageBlock` represents an image (as base64 PNG)."""

    def __init__(self, value: str, meta: dict[str, Any] | None = None):
        """Initializes the ImageBlock with a base64 PNG string representation and some metadata."""
        assert self.is_valid_base64_png(value), (
            "Invalid base64 string representation of image."
        )
        self._value = value
        self._meta = {} if meta is None else meta

    @staticmethod
    def is_valid_base64_png(s: str) -> bool:
        """Checks if a string is a valid base64 string [AIA PAI Nc Hin R v1.0]."""
        try:
            # Check if the string has a data URI prefix and remove it.
            if "data:" in s and "base64," in s:
                s = s.split("base64,")[1]

            # Add padding if necessary
            s = s.strip()
            mod4 = len(s) % 4
            if mod4 > 0:
                s = s + "=" * (4 - mod4)

            # Attempt to decode the Base64 string
            decoded_data = base64.b64decode(s, validate=True)

            # The official PNG signature is 8 bytes long.
            png_signature = b"\x89PNG\r\n\x1a\n"

            if decoded_data.startswith(png_signature):
                return True
            else:
                return False

            return True
        except (binascii.Error, ValueError):
            return False

    @staticmethod
    def pil_to_base64(image: PILImage.Image) -> str:
        """Converts a PIL image to a base64 string representation."""
        img_io = BytesIO()
        image.save(img_io, "PNG")
        return base64.b64encode(img_io.getvalue()).decode("utf-8")

    @classmethod
    def from_pil_image(
        cls, image: PILImage.Image, meta: dict[str, Any] | None = None
    ) -> ImageBlock:
        """Converts a PIL image to a base64 string representation."""
        image_base64 = cls.pil_to_base64(image)
        return cls(image_base64, meta)

    def __str__(self):
        """Stringifies the block."""
        return self._value

    def __repr__(self):
        """Provides a python-parsable representation of the block (usually)."""
        return f"ImageBlock({self._value}, {self._meta.__repr__()})"


@runtime_checkable
class Component(Protocol):
    """A `Component` is a composite data structure that is intended to be represented to an LLM."""

    def parts(self) -> list[Component | CBlock]:
        """The set of all the constituent parts of the `Component`."""
        raise NotImplementedError("parts isn't implemented by default")

    def format_for_llm(self) -> TemplateRepresentation | str:
        """Formats the `Component` into a `TemplateRepresentation` or string.

        Returns: a `TemplateRepresentation` or string
        """
        raise NotImplementedError("format_for_llm isn't implemented by default")


def get_images_from_component(c: Component) -> None | list[ImageBlock]:
    """Gets images from a `Component` if they are present and a non-empty list, otherwise returns None."""
    if hasattr(c, "images"):
        imgs = c.images
        if imgs is not None:
            assert isinstance(imgs, list), "images field must be a list."
            assert all(isinstance(im, ImageBlock) for im in imgs), (
                "all elements of images list must be ImageBlocks."
            )
            if len(imgs) == 0:
                return None
            else:
                return imgs
        else:
            return None
    else:
        return None


class ModelOutputThunk(CBlock):
    """A `ModelOutputThunk` is a special type of `CBlock` that we know came from a model's output. It is possible to instantiate one without the output being computed yet."""

    def __init__(
        self,
        value: str | None,
        meta: dict[str, Any] | None = None,
        parsed_repr: CBlock | Component | Any | None = None,
        tool_calls: dict[str, ModelToolCall] | None = None,
    ):
        """Initializes as a cblock, optionally also with a parsed representation from an output formatter."""
        super().__init__(value, meta)
        self.parsed_repr: CBlock | Component | Any | None = parsed_repr
        self.tool_calls = tool_calls

    def is_computed(self):
        """Returns true only if this Thunk has already been filled."""
        return self.value is not None


def blockify(s: str | CBlock | Component) -> CBlock | Component:
    """`blockify` is a helper function that turns raw strings into CBlocks."""
    # noinspection PyUnreachableCode
    match s:
        case str():
            return CBlock(s)
        case CBlock():
            return s
        case Component():
            return s
        case _:
            raise Exception("Type Error")


@dataclass
class ContextTurn:
    """A turn of model input and model output."""

    model_input: CBlock | Component | None
    output: ModelOutputThunk | None


class Context(abc.ABC):
    """A `Context` is used to track the state of a `MelleaSession`."""

    is_chat_context: bool = False

    @abc.abstractmethod
    def reset(self):
        """Resets the context to a fresh state.

        Note: resetting a context does NOT free memory or clear cache. For this reason, you probably want to be calling this method from a `Session`.
        """
        ...

    @abc.abstractmethod
    def insert(
        self,
        value: CBlock | Component,
        *,
        key: Any | None = None,
        generate_logs: list[GenerateLog] | None = None,
    ):
        """Each `Context` must define its own semantics for inserting something into the context.

        Args:
            value (CBlock | Component): the thing to insert.
            key (Optional[Any]): a key by which the value is indexed to. This is optional and only needed for fairly sophisticated Context types. Note that this is NOT necessarily a key that can be used for KV cache lookups!
            generate_logs: Adding log information about the insertion. Should only be used for output objects.
        """
        ...

    @abc.abstractmethod
    def insert_turn(
        self, turn: ContextTurn, *, generate_logs: list[GenerateLog] | None = None
    ):
        """Insert a turn into the chat history.

        Args:
            turn: the turn to insert.
            generate_logs: Adding log information about the insertion. Will be bound to the output part of the turn.

        Returns:
            None
        """
        ...

    @abc.abstractmethod
    def copy(self) -> Context:
        """Produces a deep copy of the current Context's contents, allowing for branch-and-merge style semantics over a Context."""
        ...

    @abc.abstractmethod
    def _hash_for_kv_cache(self):
        """A `Context` is responsible for maintaining a hash representation of itself. This hash is used by backends to refer to a Context's state."""
        ...

    @abc.abstractmethod
    def render_for_generation(self) -> list[Component | CBlock] | None:
        """Provides a linear list of context components to use for generation, or None if that is not possible to construct."""
        ...

    @abc.abstractmethod
    def actions_for_available_tools(self) -> list[Component | CBlock] | None:
        """Provides a list of actions to extract tools from for use with during generation, or None if that's not possible.

        Can be used to make the available tools differ from the tools of all the actions in the context.
        """
        ...

    @abc.abstractmethod
    def full_event_log(self) -> list[Component | CBlock]:
        """Provides a list of all events stored in the context."""
        ...

    @abc.abstractmethod
    def last_output(self) -> ModelOutputThunk | None:
        """The last output thunk of the context."""
        ...

    @abc.abstractmethod
    def last_turn(self) -> ContextTurn | None:
        """The last input/output turn of the context."""
        ...

    @property
    @abc.abstractmethod
    def logs(self) -> list[list[GenerateLog] | None]:
        """Returns a list of all logs in the context."""
        ...

    @abc.abstractmethod
    def get_logs_by_index(self, index: int) -> list[GenerateLog] | None:
        """Returns a `GenerateLog` for the given index."""
        ...

    @abc.abstractmethod
    def last_output_and_logs(
        self, all_intermediate_results: bool = False
    ) -> tuple[ModelOutputThunk | None, list[GenerateLog] | None | GenerateLog]:
        """Returns a `ModelOutputThunk` for the last output and the corresponding `GenerateLog`.

        Args:
            all_intermediate_results: if False (default), only returns the Log for the that led to the final output, if True, a list of all intermediate results (including the final one) is returned.
        """
        ...


class BasicContext(Context, abc.ABC):
    """Implementing some common functionality for Contexts."""

    _ctx: list[CBlock | Component | ModelOutputThunk] = []
    _log_ctx: list[list[GenerateLog] | None] = []

    def __init__(self):
        """Constructs a basic context."""
        self._ctx = []
        self._log_ctx = []

    def actions_for_available_tools(self) -> list[Component | CBlock] | None:
        """Provides a list of actions to extract tools from for use with during generation, or None if that's not possible.

        Can be used to make the available tools differ from the tools of all the actions in the context.
        In most cases, this will just be the same context as `render_for_generation`.
        """
        return self.render_for_generation()

    def last_output(self):
        """The last output thunk of the context."""
        for c in self._ctx[::-1]:
            if isinstance(c, ModelOutputThunk):
                return c
        return None

    @property
    def logs(self) -> list[list[GenerateLog] | None]:
        """Returns a list of all logs in the context."""
        return list(self._log_ctx)

    def get_logs_by_index(self, index: int) -> list[GenerateLog] | None:
        """Returns the log of a given index from the context."""
        try:
            return self._log_ctx[index]
        except IndexError:
            FancyLogger.get_logger().warn(f"Index {index} for logs is out of range")
            return None

    def last_output_and_logs(
        self, all_intermediate_results: bool = False
    ) -> tuple[ModelOutputThunk | None, list[GenerateLog] | GenerateLog | None]:
        """The last output thunk of the context and the corresponding log."""
        last: ModelOutputThunk | None = None
        last_i = 0
        for c in self._ctx[::-1]:
            last_i -= 1
            if isinstance(c, ModelOutputThunk):
                last = c
                break
        if last is None:
            return None, None
        else:
            logs = self.get_logs_by_index(last_i)
            if all_intermediate_results or logs is None:
                # return everything
                return last, logs
            else:
                log = [log for log in logs if log.is_final_result]
                # if there is only one log in history, this should be the one.
                if len(log) == 0:
                    if len(logs) == 1:
                        FancyLogger.get_logger().warn(
                            f"No final result found for log {logs[0]}. Returning the only result."
                        )
                        log = logs
                    else:
                        FancyLogger.get_logger().warn(
                            f"No final result found for log {logs[0]}. Could not decide which log to return. Returning None."
                        )
                        return last, None
                assert len(log) == 1, (
                    f"Found multiple/none final results for logs: {len(log)}, "
                )
                return last, log[0]

    def full_event_log(self) -> list[Component | CBlock]:
        """Returns the underlying _ctx."""
        return self._ctx

    def last_turn(self):
        """The last input/output turn of the context."""
        if len(self._ctx) == 0:
            return None
        last_element = self._ctx[-1]
        if isinstance(last_element, ModelOutputThunk):
            if len(self._ctx) >= 2:
                # assuming that the last two elements are input and output
                return ContextTurn(self._ctx[-2], last_element)
            else:
                # if self._ctx is of size 1 and only element is output element, return partial turn without an input.
                return ContextTurn(None, last_element)
        else:
            # if the last element is input element, return partial turn without output
            return ContextTurn(last_element, None)

    def __str__(self):
        """Pretty prints the context. For debugging."""
        return f"{self.__class__.__name__} := \n" + "\n".join(
            [f"    {c!s}" for c in self._ctx]
        )


class LinearContext(BasicContext):
    """Initializes a linear context with unbounded window_size and is_chat=True by default."""

    def __init__(
        self,
        *,
        window_size: int | None = None,
        log_window_size: int | None = 10,
        is_chat_context=True,
    ):
        """Initializes a linear context with unbounded window_size (log_window_size = 10) and is_chat=True by default."""
        super().__init__()
        self.window_size = window_size
        self._log_window_size = log_window_size
        self.is_chat_context = is_chat_context

    def reset(self):
        """Resets the context to a fresh state.

        Note: resetting a context does NOT free memory or clear cache. For this reason, you probably want to be calling this method from a `Session`.
        """
        self._ctx = []

    def insert(
        self,
        value: CBlock | Component,
        *,
        key: Any | None = None,
        generate_logs: list[GenerateLog] | None = None,
    ):
        """Inserts into the context and then shifts the window forward if necessary."""
        self._ctx.append(value)
        self._log_ctx.append(generate_logs)
        if self.window_size is not None and len(self._ctx) > self.window_size:
            del self._ctx[0]
        if (
            self._log_window_size is not None
            and len(self._log_ctx) > self._log_window_size
        ):
            del self._log_ctx[0]

    def insert_turn(
        self, turn: ContextTurn, *, generate_logs: list[GenerateLog] | None = None
    ):
        """Insert a turn into the context."""
        if turn.model_input:
            self.insert(turn.model_input, generate_logs=None)
        if turn.output:
            self.insert(turn.output, generate_logs=generate_logs)

    def render_for_generation(self) -> list[Component | CBlock] | None:
        """Returns the underlying _ctx list for generation."""
        return self._ctx

    def is_chat_history(self):
        FancyLogger.get_logger().warning(
            "is_chat_history doesn't work properly, because ModelOutputThunks are not Messages."
        )
        """Returns true if everything in the LinearContext is a chat `Message`."""
        return all(
            str(type(x)) == "Message" for x in self._ctx
        )  # sic: avoids circular import.

    def _hash_for_kv_cache(self):
        """Constructs a hash that corresponds to the string contents of the KV cache associated with this context."""
        assert False, "not supported yet."

    def copy(self):
        """Constructs a deep copy of this Context."""
        return deepcopy(self)


class SimpleContext(BasicContext):
    """A `SimpleContext` is a context in which each interaction is a separate and independent turn. The history of all previous turns is NOT saved.

    This context is intended for applications where each LLM call is (mostly) a stand-alone request. Patterns like instruct-validate-repair fall into this category.

    It is possible for a single turn to have many different CBlocks/Components. This can happen for a variety of reasons:
    1. Instruct/Repair is actually up to 3 (not 4!) turns: a system, a user, an assistant, and then the ALora output.
    2. It's possible to have a Component with a bunch of other stuff in it. We haven't decided how to represent this in Span world yet, but it's possible that one approach would be to have any causal dependency structure represented in terms of a linearization or poset.
    """

    def __init__(self):
        """Initializes a SimpleContext which contains at max one turn. with is_chat_context=True."""
        super().__init__()
        self.is_chat_context = True

    def render_for_generation(self) -> list[Component | CBlock] | None:
        """Uses _ctx ordering."""
        return []

    def reset(self):
        """Resets the context to a fresh state.

        Note: resetting a context does NOT free memory or clear cache. For this reason, you probably want to be calling this method from a `Session`.
        """
        self._ctx = []

    def insert(
        self,
        value: CBlock | Component,
        *,
        key: Any | None = None,
        generate_logs: list[GenerateLog] | None = None,
    ):
        """Adds the value to the context."""
        assert key is None
        self._ctx = [value]
        self._log_ctx = [generate_logs]

    def insert_turn(
        self, turn: ContextTurn, *, generate_logs: list[GenerateLog] | None = None
    ):
        """Removes the previous turn and starts a new one."""
        self.reset()
        self._ctx = [x for x in [turn.model_input, turn.output] if x]
        self._log_ctx = [None, generate_logs]

    def _hash_for_kv_cache(self):
        """Constructs a hash that corresponds to the string contents of the KV cache associated with this context."""
        assert False, "not supported yet."

    def copy(self):
        """Constructs a deep copy of this Context."""
        return deepcopy(self)


@dataclass
class TemplateRepresentation:
    """Representing a component as a set of important attributes that can be consumed by the formatter."""

    obj: Any
    args: dict[
        str,
        str | Component | CBlock | Iterable | Mapping | TemplateRepresentation | None,
    ]
    tools: dict[str, Callable] | None = (
        None  # the key must be the name of the function.
    )
    fields: list[Any] | None = None
    template: str | None = None
    template_order: list[str] | None = None
    images: list[ImageBlock] | None = None


@dataclass
class GenerateLog:
    """A dataclass for capturing log entries.

    GenerateLog provides a structured way to include various details in log entries, making it useful for maintaining detailed
    records of events or operations where context and additional data are significant.
    """

    date: datetime.datetime | None = None
    prompt: str | list[dict] | None = None
    backend: str | None = None
    model_options: dict[str, Any] | None = None
    model_output: Any | None = None
    action: Component | CBlock | None = None
    result: ModelOutputThunk | None = None
    is_final_result: bool | None = False
    extra: dict[str, Any] | None = None


@dataclass
class ModelToolCall:
    """A dataclass for capturing the tool calls a model wants to make.

    Provides a unified way to call tools post generation.
    """

    name: str
    func: Callable
    args: Mapping[str, Any]

    def call_func(self) -> Any:
        """A helper function for calling the function/tool represented by this object."""
        return self.func(**self.args)
