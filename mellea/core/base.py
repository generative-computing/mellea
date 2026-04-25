"""Foundational data structures for mellea's generative programming model.

Defines the building blocks that flow through every layer of the library: ``CBlock``
(a content block wrapping a string value), ``Component`` (an abstract composable
generative unit), ``ModelOutputThunk`` (a lazily-evaluated model response),
``Context`` and ``ContextTurn`` (stateful conversation history containers),
``TemplateRepresentation`` (the structured rendering of a component for prompt
templates), ``ImageBlock``, and ``ModelToolCall``. Understanding these types is
the starting point for building custom components or sampling strategies.
"""

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
from typing import Any, Generic, Literal, Protocol, TypeVar, runtime_checkable

import typing_extensions
from PIL import Image as PILImage

from ..plugins.manager import has_plugins, invoke_hook
from ..plugins.types import HookType


class CBlock:
    """A `CBlock` is a block of content that can serve as input to or output from an LLM.

    Args:
        value (str | None): The underlying string content of the block.
        meta (dict[str, Any] | None): Optional metadata about this block (e.g., the inference engine's
            completion object). Defaults to an empty dict.
        cache (bool): If ``True``, the inference engine may store the KV cache for this block. Experimental.

    """

    def __init__(
        self,
        value: str | None,
        meta: dict[str, Any] | None = None,
        *,
        cache: bool = False,
    ):
        """Initialize CBlock with a string value and optional metadata."""
        if value is not None and not isinstance(value, str):
            raise TypeError("value to a Cblock should always be a string or None")
        self._underlying_value = value
        self.cache = cache
        if meta is None:
            meta = {}
        self._meta = meta

    @property
    def value(self) -> str | None:
        """Gets the value of the block."""
        return self._underlying_value

    @value.setter
    def value(self, v: str) -> None:
        """Sets the value of the block."""
        self._underlying_value = v

    def __str__(self) -> str:
        """Stringifies the block."""
        return self.value if self.value else ""

    def __repr__(self) -> str:
        """Provides a python-parsable representation of the block (usually)."""
        return f"CBlock({self.value}, {self._meta.__repr__()})"


class ImageBlock(CBlock):
    """A `ImageBlock` represents an image (as base64 PNG).

    Args:
        value (str): A valid base64-encoded PNG string (with or without a data URI prefix).
        meta (dict[str, Any] | None): Optional metadata to associate with this image block.

    """

    def __init__(self, value: str, meta: dict[str, Any] | None = None):
        """Initialize ImageBlock with a base64-encoded PNG string, validating the encoding.

        Raises:
            AssertionError: If ``value`` is not a valid base64-encoded PNG string.
        """
        assert self.is_valid_base64_png(value), (
            "Invalid base64 string representation of image."
        )
        super().__init__(value, meta)

    @staticmethod
    def is_valid_base64_png(s: str) -> bool:
        """Checks whether a string is a valid base64-encoded PNG image.

        Strips any data URI prefix before decoding. Adds padding characters if
        necessary to make the base64 string a valid length.

        Args:
            s (str): The string to validate, optionally prefixed with a data URI header.

        Returns:
            bool: ``True`` if the string decodes to a PNG image, ``False`` otherwise.
        """
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
        """Converts a PIL image to a base64-encoded PNG string.

        Args:
            image (PILImage.Image): The PIL image to encode.

        Returns:
            str: A base64-encoded string of the image serialised as PNG.
        """
        img_io = BytesIO()
        image.save(img_io, "PNG")
        return base64.b64encode(img_io.getvalue()).decode("utf-8")

    @classmethod
    def from_pil_image(
        cls, image: PILImage.Image, meta: dict[str, Any] | None = None
    ) -> ImageBlock:
        """Creates an ``ImageBlock`` from a PIL image object.

        Converts the image to a base64-encoded PNG string and wraps it in a new
        ``ImageBlock`` instance.

        Args:
            image (PILImage.Image): The PIL image to encode.
            meta (dict[str, Any] | None): Optional metadata to associate with the block.

        Returns:
            ImageBlock: A new ``ImageBlock`` containing the base64-encoded PNG.
        """
        image_base64 = cls.pil_to_base64(image)
        return cls(image_base64, meta)

    def __repr__(self) -> str:
        """Provides a python-parsable representation of the block (usually)."""
        return f"ImageBlock({self.value}, {self._meta.__repr__()})"


S = typing_extensions.TypeVar("S", default=Any, covariant=True)
"""Used for class definitions for Component and ModelOutputThunk; also used for functions that don't accept CBlocks. Defaults to `Any`."""

C = typing_extensions.TypeVar("C", default=str)
"""Used for component typing in function parameters where the function takes a Component[C] and/or CBlock and can return a ModelOutputThunk[C]. Defaults to `str`."""


class ComponentParseError(Exception):
    """Raised by `Component.parse()` when the underlying parsing method throws an exception."""


@runtime_checkable
class Component(Protocol, Generic[S]):
    """A `Component` is a composite data structure that is intended to be represented to an LLM."""

    def parts(self) -> list[Component | CBlock]:
        """Returns the set of all constituent sub-components and content blocks of this ``Component``.

        Returns:
            list[Component | CBlock]: A list of child ``Component`` or ``CBlock`` objects that make
            up this component. The list may be empty for leaf components.

        Raises:
            NotImplementedError: If the concrete subclass has not overridden this method.
        """
        raise NotImplementedError("parts isn't implemented by default")

    def format_for_llm(self) -> TemplateRepresentation | str:
        """Formats the ``Component`` into a ``TemplateRepresentation`` or plain string for LLM consumption.

        Returns:
            TemplateRepresentation | str: A structured ``TemplateRepresentation`` (for components
            with tools, fields, or templates) or a plain string for simple components.

        Raises:
            NotImplementedError: If the concrete subclass has not overridden this method.
        """
        raise NotImplementedError("format_for_llm isn't implemented by default")

    def parse(self, computed: ModelOutputThunk) -> S:
        """Parses the expected type ``S`` from a given ``ModelOutputThunk``.

        Delegates to the component's underlying ``_parse`` method and wraps any
        exception in a ``ComponentParseError`` for uniform error handling.

        Args:
            computed (ModelOutputThunk): The model output thunk whose value should be parsed.

        Returns:
            S: The parsed result produced by ``_parse``, typed according to the component's type parameter.

        Raises:
            ComponentParseError: If the underlying ``_parse`` call raises any exception.
        """
        try:
            return self._parse(computed)
        except Exception as e:
            raise ComponentParseError(f"component parsing failed: {e}")

    def _parse(self, computed: ModelOutputThunk) -> S:
        """Components can define a return type that is parsed from the text output of an LLM."""
        raise NotImplementedError("parse isn't implemented by default")


class GenerateType(enum.Enum):
    """Used to track what functions can be used to extract a value from a ModelOutputThunk.

    Attributes:
        NONE (None): No generation function has been set; the thunk is either already computed or uninitialized.
        ASYNC (int): The generation function is async-compatible; ``avalue``/``astream`` may be used.
        SYNC (int): The generation function is synchronous only; async extraction methods are unavailable.
    """

    NONE = None
    ASYNC = 1
    SYNC = 2


@dataclass
class GenerationMetadata:
    """Backend execution metadata attached to every ModelOutputThunk.

    Fields are populated as generation progresses; see individual field docstrings for timing.

    Args:
        usage: Token usage dict with 'prompt_tokens', 'completion_tokens', 'total_tokens'.
        model: Model identifier that generated the output.
        provider: Provider name (e.g. 'openai', 'ollama', 'huggingface', 'watsonx').
        ttfb_ms: Time to first token in milliseconds; None for non-streaming.
        streaming: Whether this generation used streaming mode.
    """

    usage: dict[str, Any] | None = None
    """Token usage following OpenAI API standard.

    Core fields: 'prompt_tokens', 'completion_tokens', 'total_tokens'.
    May include 'prompt_tokens_details' (nested dict; 'cached_tokens' = tokens served
    from prompt cache) and 'completion_tokens_details'. Anthropic models also populate
    'cache_creation_input_tokens' (tokens written to prompt cache).
    """

    model: str | None = None
    """Model identifier that generated the output (e.g. 'gpt-4', 'llama2:7b', 'meta-llama/Llama-2-7b-hf')."""

    provider: str | None = None
    """Provider name (e.g. 'openai', 'ollama', 'huggingface', 'watsonx')."""

    ttfb_ms: float | None = None
    """Time to first token in milliseconds.

    Set when the first chunk is received from the backend.
    None for non-streaming requests or when not measured.
    """

    streaming: bool = False
    """Whether this generation used streaming mode.

    Set from model options at the start of astream().
    """


class ModelOutputThunk(CBlock, Generic[S]):
    """A `ModelOutputThunk` is a special type of `CBlock` that we know came from a model's output. It is possible to instantiate one without the output being computed yet.

    Args:
        value (str | None): The raw model output string, or ``None`` if not yet computed.
        meta (dict[str, Any] | None): Optional metadata from the inference engine (e.g., completion object).
        parsed_repr (S | None): An already-parsed representation to attach; set when re-wrapping existing output.
        tool_calls (dict[str, ModelToolCall] | None): Tool calls returned by the model alongside the text output.

    """

    def __init__(
        self,
        value: str | None,
        meta: dict[str, Any] | None = None,
        parsed_repr: S | None = None,
        tool_calls: dict[str, ModelToolCall] | None = None,
    ):
        """Initialize ModelOutputThunk with an optional pre-computed value and metadata."""
        super().__init__(value, meta)

        self.parsed_repr: S | None = parsed_repr
        """Will be non-`None` once computed."""

        # Set computed to True if a value is passed in.
        self._computed: bool = True if value is not None else False

        # Additional fields that should be standardized across apis.
        self.tool_calls = tool_calls
        self._thinking: str | None = None
        self.generation: GenerationMetadata = GenerationMetadata()
        """Backend execution metadata populated during generation."""

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
        self._on_computed: Callable[[ModelOutputThunk], Coroutine] | None = None

        self._start: datetime.datetime | None = None
        self._first_chunk_received: bool = False
        self._generate_log: GenerateLog | None = None

    def _record_ttfb(self) -> None:
        """Record time-to-first-byte if streaming and not yet recorded."""
        if (
            self.generation.streaming
            and not self._first_chunk_received
            and self._start is not None
        ):
            self.generation.ttfb_ms = (
                datetime.datetime.now() - self._start
            ).total_seconds() * 1000
            self._first_chunk_received = True

    def _copy_from(self, other: ModelOutputThunk) -> None:
        """Copy computed-output fields from *other* into *self*.

        This is used when a hook replaces the MOT: callers already hold a
        reference to *self*, so we swap the output-relevant state in-place
        rather than replacing the object.
        """
        self._underlying_value = other._underlying_value
        self._meta = other._meta
        self.parsed_repr = other.parsed_repr
        self.tool_calls = other.tool_calls
        self._thinking = other._thinking
        self.generation = other.generation
        self._generate_log = other._generate_log

    def is_computed(self) -> bool:
        """Returns true only if this Thunk has already been filled.

        Returns:
            ``True`` if the thunk value has been set, ``False`` otherwise.
        """
        return self._computed

    @property
    def value(self) -> str | None:
        """Gets the value of the block."""
        if not self._computed:
            return None
        return self._underlying_value

    @value.setter
    def value(self, v: str) -> None:
        """Sets the value of the block."""
        self._underlying_value = v

    async def avalue(self) -> str:
        """Returns the fully resolved value of the ModelOutputThunk, awaiting generation if necessary.

        Can be used for both async streaming and async non-streaming backends. If the
        thunk is already computed the value is returned immediately.

        Returns:
            str: The complete text output from the model.

        Raises:
            Exception: Propagates any errors from the underlying inference engine api request.
            RuntimeError: If called when the ModelOutputThunk's generate function is not async compatible.
        """
        if self._computed:
            assert self.value is not None  # If computed, the value cannot be None.
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
        """Returns only the NEW text fragment (delta) received since the last call.

        This method is designed for streaming consumption where you want incremental
        updates. Each call returns only the newly received content, not the accumulated
        text. When streaming is complete, subsequent calls will raise RuntimeError.

        **Note**: Be careful with calling this function. Only call it from one location at a time. This means you shouldn't pass a ModelOutputThunk to
        multiple coroutines/tasks and call astream from those coroutines/tasks simultaneously. We have considered solutions to this but are waiting until
        we see this error happen in a real use case.

        Returns:
            str: Only the new text fragment received since the last call (delta), not the
                accumulated text. Returns empty string if no new content is available yet.

        Raises:
            Exception: Propagates any errors from the underlying inference engine api request.
            RuntimeError: If called when the ModelOutputThunk's generate function is not async compatible,
                or if called after the thunk is already computed.
        """
        if self._computed:
            raise RuntimeError(
                "Streaming has finished and MOT is computed. Subsequent calls to mot.astream() are not permitted."
            )

        do_set_computed = False
        # Use string directly to avoid importing ModelOption from backends into core (circular import).
        # ModelOption.STREAM is defined in mellea/backends/model_options.py.
        self.generation.streaming = bool(
            (self._model_options or {}).get("@@@stream@@@", False)
        )

        if not self._generate_type == GenerateType.ASYNC:
            raise RuntimeError(
                f"Cannot use `ModelOutputThunk.astream()` when the generate function is using `{self._generate_type.name}`"
            )
        # Beginning value
        beginning_length = (
            0 if self._underlying_value is None else len(str(self._underlying_value))
        )  # type: ignore

        # Type of the chunk depends on the backend.
        chunks: list[Any | None] = []
        while True:
            try:
                item = self._async_queue.get_nowait()
                chunks.append(item)
                self._record_ttfb()
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
            self._record_ttfb()

        # Process the sentinel value if it's there.
        if chunks[-1] is None:
            chunks.pop()  # Remove the sentinel value.
            do_set_computed = True

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
            # Close any open telemetry span before propagating the error.
            # We can't call full post_process here (it assumes success invariants),
            # but we must not leak the span.
            span = self._meta.get("_telemetry_span")
            if span is not None:
                from ..telemetry import end_backend_span, set_span_error

                set_span_error(span, chunks[-1])
                end_backend_span(span)
                del self._meta["_telemetry_span"]

            # Fire generation_error hook (FIRE_AND_FORGET — does not block the raise)
            if has_plugins(HookType.GENERATION_ERROR):
                from ..plugins.hooks.generation import GenerationErrorPayload

                err_payload = GenerationErrorPayload(
                    exception=chunks[-1], model_output=self
                )
                await invoke_hook(HookType.GENERATION_ERROR, err_payload)

            raise chunks[-1]

        for chunk in chunks:
            assert self._process is not None
            await self._process(self, chunk)

        if do_set_computed:
            assert self._underlying_value is not None
            self._computed = True

            assert self._post_process is not None
            await self._post_process(self)

            match self._action:
                case Component():
                    self.parsed_repr = self._action._parse(self)
                case CBlock():
                    assert self.value is not None, (
                        "value must be non-None since this thunk is computed"
                    )
                    self.parsed_repr = self.value  # type: ignore
                case _:
                    raise ValueError(
                        "attempted to astream from a model output thunk with no ._action set"
                    )
            assert self.parsed_repr is not None, (
                "enforce constraint that a computed ModelOutputThunk has a non-None parsed_repr"
            )

            # --- generation_post_call hook ---
            if has_plugins(HookType.GENERATION_POST_CALL):
                from ..plugins.hooks.generation import GenerationPostCallPayload

                glog = self._generate_log
                prompt = glog.prompt if glog and glog.prompt else ""
                latency_ms = (
                    (datetime.datetime.now() - self._start).total_seconds() * 1000
                    if self._start
                    else -1
                )
                post_payload = GenerationPostCallPayload(
                    prompt=prompt, model_output=self, latency_ms=latency_ms
                )
                await invoke_hook(HookType.GENERATION_POST_CALL, post_payload)
                # NOTE: If we allow generation_post_call to modify the model output thunk, we need to
                # set the value and copy over fields here.
                # replacement = await invoke_hook(...)
                # if replacement is not None and replacement is not self:
                #     self._copy_from(replacement)

        return (
            self._underlying_value
            if beginning_length == 0
            else self._underlying_value[beginning_length:]  # type: ignore
        )

    def __repr__(self) -> str:
        """Provides a python-parsable representation (usually).

        Differs from CBlock because `._meta` can be very large for ModelOutputThunks.
        """
        return f"ModelOutputThunk({self.value})"

    def __copy__(self) -> ModelOutputThunk:
        """Returns a shallow copy of the ModelOutputThunk. A copied ModelOutputThunk cannot be used for generation; don't copy over fields associated with generating."""
        copied = ModelOutputThunk(
            self._underlying_value, self._meta, self.parsed_repr, self.tool_calls
        )

        # Check if the parsed_repr needs to be changed. A ModelOutputThunk's parsed_repr can point to
        # itself if the parsing didn't result in a new representation. It makes sense to update the
        # parsed_repr to the copied ModelOutputThunk in that case.
        if self.parsed_repr is self:
            copied.parsed_repr = copied  # type: ignore

        copied._computed = self._computed
        copied._thinking = self._thinking
        copied._action = self._action
        copied._context = self._context
        copied._generate_log = self._generate_log
        copied._model_options = self._model_options
        copied.generation = copy(self.generation)
        return copied

    def __deepcopy__(self, memo: dict) -> ModelOutputThunk:
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
        deepcopied.generation = deepcopy(self.generation)
        return deepcopied


class ComputedModelOutputThunk(ModelOutputThunk[S]):
    """A `ComputedModelOutputThunk` is a `ModelOutputThunk` that is guaranteed to be computed.

    This subclass provides a clear type distinction between thunks that may need awaiting
    and those that are already computed. It should be returned from synchronous functions
    and sampling strategies to indicate that no awaiting is needed.

    Uses zero-copy class reassignment: calling `ComputedModelOutputThunk(thunk)` reassigns
    the thunk's `__class__` to `ComputedModelOutputThunk` without creating a new object.

    Args:
        thunk: A fully-computed ``ModelOutputThunk`` whose class will be reassigned.
    """

    def __new__(cls, thunk: ModelOutputThunk[S]) -> ComputedModelOutputThunk[S]:
        """Convert the ModelOutputThunk into a ComputedModelOutputThunk."""
        thunk.__class__ = cls
        return thunk  # type: ignore[return-value]

    def __init__(self, thunk: ModelOutputThunk[S]) -> None:
        """A `ComputedModelOutputThunk` is a `ModelOutputThunk` that is guaranteed to be computed.

        Uses zero-copy class reassignment: calling `ComputedModelOutputThunk(thunk)` reassigns
        the thunk's `__class__` to `ComputedModelOutputThunk` without creating a new object.
        """
        # Call the underlying value. It's already been cast as a ComputedModelOutputThunk, so it's .is_computed() value is always True.
        if not self._computed:
            raise ValueError(
                "ComputedModelOutputThunk requires a computed ModelOutputThunk; but ._computed is False."
            )
        if self.value is None:
            raise ValueError("ComputedModelOutputThunk requires a non-None value.")

    async def avalue(self) -> str:
        """Return the value of the thunk. Use .value instead.

        Returns:
            The computed string value.
        """
        assert self.value is not None, "ComputedModelOutputThunk value cannot be None"
        return self.value

    async def astream(self) -> str:
        """Cannot astream from ComputedModelOutputThunks. Use .value instead.

        Returns:
            Never returns; always raises.

        Raises:
            RuntimeError: Always, because computed thunks do not support streaming.
        """
        raise RuntimeError(
            "Cannot stream from a ComputedModelOutputThunk. "
            "This thunk is already fully computed and does not support streaming."
        )

    @property
    def value(self) -> str:
        """Gets the value of the block."""
        return self._underlying_value  # type: ignore

    @value.setter
    def value(self, v: str):
        """Sets the value of the block."""
        self._underlying_value = v

    def is_computed(self) -> Literal[True]:
        """Returns `True` since thunk is always computed.

        Returns:
            Always ``True``.
        """
        return True


@dataclass
class ContextTurn:
    """A turn of model input and model output.

    Args:
        model_input (CBlock | Component | None): The input component or content block for this turn,
            or ``None`` for an output-only partial turn.
        output (ModelOutputThunk | None): The model's output thunk for this turn,
            or ``None`` for an input-only partial turn.

    """

    model_input: CBlock | Component | None
    output: ModelOutputThunk | None


ContextT = TypeVar("ContextT", bound="Context")


class Context(abc.ABC):
    """A `Context` is used to track the state of a `MelleaSession`.

    A context is immutable. Every alteration leads to a new context.

    Attributes:
        is_root_node (bool): ``True`` when this context is the root (empty) node of the linked list.
        previous_node (Context | None): The context node from which this one was created,
            or ``None`` for the root node.
        node_data (Component | CBlock | None): The data associated with this context node,
            or ``None`` for the root node.
        is_chat_context (bool): Whether this context operates in chat (multi-turn) mode.
    """

    _previous: Context | None
    _data: Component | CBlock | None
    _is_root: bool
    _is_chat_context: bool = True

    def __init__(self) -> None:
        """Constructs a new root context with no content."""
        self._previous = None
        self._data = None
        self._is_root = True

    # factory functions below this line.

    @classmethod
    def from_previous(
        cls: type[ContextT], previous: Context, data: Component | CBlock
    ) -> ContextT:
        """Constructs a new context node linked to an existing context node.

        Args:
            previous (Context): The existing context to extend.
            data (Component | CBlock): The component or content block to associate with the new node.

        Returns:
            ContextT: A new context instance whose ``previous_node`` is ``previous``.
        """
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
        """Returns a new empty (root) context.

        Returns:
            ContextT: A freshly initialised root context with no data or history.
        """
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
        """Returns a list of context components sorted from earliest (first) to most recent (last).

        If `last_n_components` is `None`, then all components are returned.

        Args:
            last_n_components (int | None): Maximum number of most-recent components to include.
                Pass ``None`` to return the full history.

        Returns:
            list[Component | CBlock]: Components in chronological order (oldest first).
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
        """Provides a list of actions to extract tools from for use during generation.

        Returns ``None`` if it is not possible to construct such a list. Can be used to make
        the available tools differ from the tools of all the actions in the context. Can be
        overridden by subclasses.

        Returns:
            list[Component | CBlock] | None: The list of actions whose tools should be made
            available during generation, or ``None`` if unavailable.
        """
        return self.view_for_generation()

    def last_output(self, check_last_n_components: int = 3) -> ModelOutputThunk | None:
        """Returns the most recent ``ModelOutputThunk`` found within the last N context components.

        Args:
            check_last_n_components (int): Number of most-recent components to search through.
                Defaults to 3.

        Returns:
            ModelOutputThunk | None: The most recent output thunk, or ``None`` if none is found
            within the searched components.
        """
        for c in self.as_list(last_n_components=check_last_n_components)[::-1]:
            if isinstance(c, ModelOutputThunk):
                return c
        return None

    def last_turn(self) -> ContextTurn | None:
        """The last input/output turn of the context.

        This can be partial. If the last event is an input, then the output is None.

        Returns:
            The most recent turn, or ``None`` if the context is empty.
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
        """Returns a new context obtained by appending ``c`` to this context.

        Args:
            c (Component | CBlock): The component or content block to add to the context.

        Returns:
            Context: A new context node with ``c`` as its data and this context as its previous node.
        """
        # something along ....from_previous(self, c)
        ...

    @abc.abstractmethod
    def view_for_generation(self) -> list[Component | CBlock] | None:
        """Provides a linear list of context components to use for generation.

        Returns ``None`` if it is not possible to construct such a list (e.g., the context
        is in an inconsistent state). Concrete subclasses define the ordering and filtering logic.

        Returns:
            list[Component | CBlock] | None: An ordered list of components suitable for passing
            to a backend, or ``None`` if generation is not currently possible.
        """
        ...


class AbstractMelleaTool(abc.ABC):
    """Abstract base class for Mellea Tool.

    Attributes:
        name (str): The unique name used to identify the tool in JSON descriptions and tool-call dispatch.
        as_json_tool (dict[str, Any]): A JSON-serialisable description of the tool, compatible with
            the function-calling schemas expected by supported inference backends.
    """

    name: str
    """Name of the tool."""

    @abc.abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Executes the tool with the provided arguments and returns the result.

        Args:
            *args: Positional arguments forwarded to the tool implementation.
            **kwargs: Keyword arguments forwarded to the tool implementation.

        Returns:
            Any: The result produced by the tool; the concrete type depends on the implementation.
        """

    @property
    @abc.abstractmethod
    def as_json_tool(self) -> dict[str, Any]:
        """Provides a JSON description for Mellea Tool."""


@dataclass
class TemplateRepresentation:
    """Representing a component as a set of important attributes that can be consumed by the formatter.

    Args:
        obj (Any): The original component object being represented.
        args (dict): Named arguments extracted from the component for template substitution.
        tools (dict[str, AbstractMelleaTool] | None): Tools available for this representation,
            keyed by the tool's function name. Defaults to ``None``.
        fields (list[Any] | None): An optional ordered list of field values for positional templates.
        template (str | None): An optional Jinja2 template string to use when rendering.
        template_order (list[str] | None): An optional ordering hint for template sections/keys.
        images (list[ImageBlock] | None): Optional list of image blocks associated with this representation.

    """

    obj: Any
    args: dict[
        str,
        str | Component | CBlock | Iterable | Mapping | TemplateRepresentation | None,
    ]
    tools: dict[str, AbstractMelleaTool] | None = (
        None  # the key must be the name of the function.
    )
    fields: list[Any] | None = None
    template: str | None = None
    template_order: list[str] | None = None
    images: list[ImageBlock] | None = None


@dataclass
class GenerateLog:
    """A dataclass for capturing log entries for a single generation call.

    GenerateLog provides a structured way to include various details in log entries, making it useful for maintaining detailed
    records of events or operations where context and additional data are significant.

    Args:
        date (datetime.datetime | None): Timestamp when the generation was logged.
        prompt (str | list[dict] | None): The prompt string or chat-message list sent to the model.
        backend (str | None): Identifier of the inference backend used for this generation.
        model_options (dict[str, Any] | None): Model configuration options applied to this call.
        model_output (Any | None): The raw output returned by the backend API.
        action (Component | CBlock | None): The component or block that triggered the generation.
        result (ModelOutputThunk | None): The ``ModelOutputThunk`` produced by this generation call.
        is_final_result (bool | None): Whether this log entry corresponds to the definitive final result.
        extra (dict[str, Any] | None): Arbitrary extra metadata to attach to the log entry.

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

    Args:
        name (str): The name of the tool the model requested to call.
        func (AbstractMelleaTool): The ``AbstractMelleaTool`` instance that will be invoked.
        args (Mapping[str, Any]): The keyword arguments the model supplied for the tool call.

    """

    name: str
    func: AbstractMelleaTool
    args: Mapping[str, Any]

    def call_func(self) -> Any:
        """Invokes the tool represented by this object and returns the result.

        Returns:
            Any: The value returned by ``func.run(**args)``; the concrete type depends on the tool.
        """
        return self.func.run(**self.args)


def blockify(s: str | CBlock | Component) -> CBlock | Component:
    """Turn a raw string into a ``CBlock``, leaving ``CBlock`` and ``Component`` objects unchanged.

    Args:
        s: A plain string, ``CBlock``, or ``Component`` to normalise.

    Returns:
        A ``CBlock`` wrapping ``s`` if it was a string; otherwise ``s`` unchanged.

    Raises:
        Exception: If ``s`` is not a ``str``, ``CBlock``, or ``Component``.
    """
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


def get_images_from_component(c: Component) -> None | list[ImageBlock]:
    """Return the images attached to a ``Component``, or ``None`` if absent or empty.

    Args:
        c: The ``Component`` whose ``images`` attribute is inspected.

    Returns:
        A non-empty list of ``ImageBlock`` objects if the component has an
        ``images`` attribute with at least one element; ``None`` otherwise.
    """
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
