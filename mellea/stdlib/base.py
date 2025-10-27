"""Basic stdlib data structures."""

from __future__ import annotations

import abc
import asyncio
import base64
import binascii
import datetime
import enum
from collections.abc import Callable, Coroutine, Iterable, Mapping
from copy import copy, deepcopy
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Protocol, TypeVar, runtime_checkable

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


# TODO: JAL. Make this a CBlock?
# TODO: JAL. Add support for passing in docs as model options?
class Document:
    def __init__(self, text: str, title: str | None = None):
        self.text = text
        self.title = title


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
        imgs = c.images  # type: ignore
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


class GenerateType(enum.Enum):
    """Used to track what functions can be used to extract a value from a ModelOutputThunk."""

    NONE = None
    ASYNC = 1
    SYNC = 2


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

        # Set computed to True if a value is passed in.
        self._computed: bool = True if value is not None else False

        # Additional fields that should be standardized across apis.
        self.tool_calls = tool_calls
        self._thinking: str | None = None

        # Used for tracking generation.
        self._context: list[Component | CBlock] | None = None
        self._action: Component | CBlock | None = None
        self._model_options: dict[str, Any] | None = None

        # Used for async and async streaming.
        self._async_queue: asyncio.Queue = asyncio.Queue(maxsize=20)
        self._chunk_size = 3  # Minimum number of chunks to stream at a single time.

        # _generate and _generate_type are linked. _generate will determine
        # what gets set for _generate_type. _generate_type determines what
        # function(s) can be used to get the value of the ModelOutputThunk.
        self._generate: asyncio.Task[None] | None = None
        self._generate_type: GenerateType = GenerateType.NONE
        self._generate_extra: asyncio.Task[Any] | None = (
            None  # Currently only used by hf.
        )
        self._process: Callable[[ModelOutputThunk, Any], Coroutine] | None = None
        self._post_process: Callable[[ModelOutputThunk], Coroutine] | None = None

        self._generate_log: GenerateLog | None = None

    def is_computed(self):
        """Returns true only if this Thunk has already been filled."""
        return self._computed

    @property
    def value(self) -> str | None:
        """Gets the value of the block."""
        if not self._computed:
            return None
        return self._underlying_value

    @value.setter
    def value(self, v: str):
        """Sets the value of the block."""
        self._underlying_value = v

    async def avalue(self) -> str:
        """Returns the value of the ModelOutputThunk. Can be used for both async streaming and async non-streaming.

        Raises:
            Exception: Propagates any errors from the underlying inference engine api request.
            RuntimeError: If called when the ModelOutputThunk's generate function is not async compatible.
        """
        if self._computed:
            assert self.value  # If computed, the value cannot be None.
            return self.value

        if not self._generate_type == GenerateType.ASYNC:
            raise RuntimeError(
                f"Cannot use `ModelOutputThunk.avalue()` when the generate function is using `{self._generate_type.name}`"
            )

        while not self._computed:
            await self.astream()

        assert self.value is not None  # If computed, the value cannot be None.
        return self.value

    # If we require a function that returns only the new chunks of data, we can implement that similarly.
    async def astream(self) -> str:
        """Returns the ModelOutputThunk's partial value including the next chunk(s). Can be used for both async streaming and async non-streaming.

        Returns the value of the ModelOutputThunk if streaming is done.

        **Note**: Be careful with calling this function. Only call it from one location at a time. This means you shouldn't pass a ModelOutputThunk to
        multiple coroutines/tasks and call astream from those coroutines/tasks simultaneously. We have considered solutions to this but are waiting until
        we see this error happen in a real use case.

        Raises:
            Exception: Propagates any errors from the underlying inference engine api request.
            RuntimeError: If called when the ModelOutputThunk's generate function is not async compatible.
        """
        if self._computed:
            assert self.value is not None  # If computed, the value cannot be None.
            return self.value

        if not self._generate_type == GenerateType.ASYNC:
            raise RuntimeError(
                f"Cannot use `ModelOutputThunk.astream()` when the generate function is using `{self._generate_type.name}`"
            )

        # Type of the chunk depends on the backend.
        chunks: list[Any | None] = []
        while True:
            try:
                item = self._async_queue.get_nowait()
                chunks.append(item)
            except asyncio.QueueEmpty:
                # We've exhausted the current items in the queue.
                break

        # Make sure we always get the minimum chunk size.
        while len(chunks) <= self._chunk_size:
            if len(chunks) > 0:
                if chunks[-1] is None or isinstance(chunks[-1], Exception):
                    break  # Hit sentinel value or an error.
                # We could switch to relying on the `done` / `finish_reason` field of chunks,
                # but that forces us to know about the chunk type here. Prefer sentinel values
                # for now.

            item = await self._async_queue.get()
            chunks.append(item)

        # Process the sentinel value if it's there.
        if chunks[-1] is None:
            chunks.pop()  # Remove the sentinel value.
            self._computed = True

            # Shouldn't be needed, but cancel the Tasks this ModelOutputThunk relied on.
            if self._generate is not None:
                self._generate.cancel()
            if self._generate_extra is not None:
                # Covers an hf edge case. The task is done generating anything useful but isn't `done` yet.
                await self._generate_extra
                self._generate_extra.cancel()

            # If ModelOutputThunks get too bulky, we can do additional cleanup here
            # and set fields to None.

        elif isinstance(chunks[-1], Exception):
            # For now, just re-raise the exception.
            # It's possible that we hit this error after already streaming some
            # chunks. We should investigate allowing recovery in the future.
            raise chunks[-1]

        for chunk in chunks:
            assert self._process is not None
            await self._process(self, chunk)

        if self._computed:
            assert self._post_process is not None
            await self._post_process(self)

        return self._underlying_value  # type: ignore

    def __repr__(self):
        """Provides a python-parsable representation (usually).

        Differs from CBlock because `._meta` can be very large for ModelOutputThunks.
        """
        return f"ModelOutputThunk({self.value})"

    def __copy__(self):
        """Returns a shallow copy of the ModelOutputThunk. A copied ModelOutputThunk cannot be used for generation; don't copy over fields associated with generating."""
        copied = ModelOutputThunk(
            self._underlying_value, self._meta, self.parsed_repr, self.tool_calls
        )

        # Check if the parsed_repr needs to be changed. A ModelOutputThunk's parsed_repr can point to
        # itself if the parsing didn't result in a new representation. It makes sense to update the
        # parsed_repr to the copied ModelOutputThunk in that case.
        if self.parsed_repr is self:
            copied.parsed_repr = copied

        copied._computed = self._computed
        copied._thinking = self._thinking
        copied._action = self._action
        copied._context = self._context
        copied._generate_log = self._generate_log
        copied._model_options = self._model_options
        return copied

    def __deepcopy__(self, memo):
        """Returns a deep copy of the ModelOutputThunk. A copied ModelOutputThunk cannot be used for generation; don't copy over fields associated with generation. Similar to __copy__ but creates deepcopies of _meta, parsed_repr, and most other fields that are objects."""
        # Use __init__ to initialize all fields. Modify the fields that need to be copied/deepcopied below.
        deepcopied = ModelOutputThunk(self._underlying_value)
        memo[id(self)] = deepcopied

        # TODO: We can tweak what gets deepcopied here. ModelOutputThunks should be immutable (unless generating),
        # so this __deepcopy__ operation should be okay if it needs to be changed to be a shallow copy.

        # Check if the parsed_repr needs to be changed. A ModelOutputThunk's parsed_repr can point to
        # itself if the parsing didn't result in a new representation. It makes sense to update the
        # parsed_repr to the deepcopied ModelOutputThunk in that case.
        if self.parsed_repr is self:
            deepcopied.parsed_repr = deepcopied
        else:
            deepcopied.parsed_repr = deepcopy(self.parsed_repr)

        deepcopied._meta = deepcopy(self._meta)
        deepcopied.tool_calls = deepcopy(self.tool_calls)
        deepcopied._computed = self._computed
        deepcopied._thinking = self._thinking
        deepcopied._action = deepcopy(self._action)
        deepcopied._context = copy(
            self._context
        )  # The items in a context should be immutable.
        deepcopied._generate_log = copy(self._generate_log)
        deepcopied._model_options = copy(self._model_options)
        return deepcopied


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


ContextT = TypeVar("ContextT", bound="Context")


class Context(abc.ABC):
    """A `Context` is used to track the state of a `MelleaSession`.

    A context is immutable. Every alteration leads to a new context.
    """

    _previous: Context | None
    _data: Component | CBlock | None
    _is_root: bool
    _is_chat_context: bool = True

    def __init__(self):
        """Constructs a new root context with no content."""
        self._previous = None
        self._data = None
        self._is_root = True

    # factory functions below this line.

    @classmethod
    def from_previous(
        cls: type[ContextT], previous: Context, data: Component | CBlock
    ) -> ContextT:
        """Constructs a new context from an existing context."""
        assert isinstance(previous, Context), (
            "Cannot create a new context from a non-Context object."
        )
        assert data is not None, "Cannot create a new context from None data."

        x = cls()
        x._previous = previous
        x._data = data
        x._is_root = False
        x._is_chat_context = previous._is_chat_context
        return x

    @classmethod
    def reset_to_new(cls: type[ContextT]) -> ContextT:
        """Returns an empty context for convenience."""
        return cls()

    # Internal functions below this line.

    @property
    def is_root_node(self) -> bool:
        """Returns whether this context is the root context node."""
        return self._is_root

    @property
    def previous_node(self) -> Context | None:
        """Returns the context node from which this context node was created.

        Internal use: Users should not need to use this property.
        """
        return self._previous

    @property
    def node_data(self) -> Component | CBlock | None:
        """Returns the data associated with this context node.

        Internal use: Users should not need to use this property.
        """
        return self._data

    @property
    def is_chat_context(self) -> bool:
        """Returns whether this context is a chat context."""
        return self._is_chat_context

    # User functions below this line.

    def as_list(self, last_n_components: int | None = None) -> list[Component | CBlock]:
        """Returns a list of the last n components in the context sorted from FIRST TO LAST.

        If `last_n_components` is `None`, then all components are returned.
        """
        context_list: list[Component | CBlock] = []
        current_context: Context = self

        last_n_count = 0
        while not current_context.is_root_node and (
            last_n_components is None or last_n_count < last_n_components
        ):
            data = current_context.node_data
            assert data is not None, "Data cannot be None (except for root context)."
            assert data not in context_list, (
                "There might be a cycle in the context tree. That is not allowed."
            )
            context_list.append(data)
            last_n_count += 1

            current_context = current_context.previous_node  # type: ignore
            assert current_context is not None, (
                "Previous context cannot be None (except for root context)."
            )

        context_list.reverse()
        return context_list

    def actions_for_available_tools(self) -> list[Component | CBlock] | None:
        """Provides a list of actions to extract tools from for use with during generation, or None if that's not possible.

        Can be used to make the available tools differ from the tools of all the actions in the context. Can be overwritten by subclasses.
        """
        return self.view_for_generation()

    def last_output(self, check_last_n_components: int = 3) -> ModelOutputThunk | None:
        """The last output thunk of the context."""
        for c in self.as_list(last_n_components=check_last_n_components)[::-1]:
            if isinstance(c, ModelOutputThunk):
                return c
        return None

    def last_turn(self):
        """The last input/output turn of the context.

        This can be partial. If the last event is an input, then the output is None.
        """
        history = self.as_list(last_n_components=2)

        if len(history) == 0:
            return None
        last_element = history[-1]
        if isinstance(last_element, ModelOutputThunk):
            if len(history) >= 2:
                # assuming that the last two elements are input and output
                return ContextTurn(history[-2], last_element)
            else:
                # if self._ctx is of size 1 and only element is output element, return partial turn without an input.
                return ContextTurn(None, last_element)
        else:
            # if the last element is input element, return partial turn without output
            return ContextTurn(last_element, None)

    # Abstract methods below this line.

    @abc.abstractmethod
    def add(self, c: Component | CBlock) -> Context:
        """Returns a new context obtained by adding `c` to this context."""
        # something along ....from_previous(self, c)
        ...

    @abc.abstractmethod
    def view_for_generation(self) -> list[Component | CBlock] | None:
        """Provides a linear list of context components to use for generation, or None if that is not possible to construct."""
        ...


class ChatContext(Context):
    """Initializes a chat context with unbounded window_size and is_chat=True by default."""

    def __init__(self, *, window_size: int | None = None):
        """Constructs a new chat context."""
        super().__init__()
        self._window_size = window_size

    def add(self, c: Component | CBlock) -> ChatContext:
        """Add a new component/cblock to the context. Returns the new context."""
        new = ChatContext.from_previous(self, c)
        new._window_size = self._window_size
        return new

    def view_for_generation(self) -> list[Component | CBlock] | None:
        """Returns the context in a linearized form. Uses the window_size set during initialization."""
        return self.as_list(self._window_size)


class SimpleContext(Context):
    """A `SimpleContext` is a context in which each interaction is a separate and independent turn. The history of all previous turns is NOT saved.."""

    def add(self, c: Component | CBlock) -> SimpleContext:
        """Add a new component/cblock to the context. Returns the new context."""
        return SimpleContext.from_previous(self, c)

    def view_for_generation(self) -> list[Component | CBlock] | None:
        """Returns an empty list."""
        return []


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
