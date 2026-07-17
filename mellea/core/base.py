# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Foundational data structures for mellea's generative programming model.

Defines the building blocks that flow through every layer of the library: `CBlock`
(a content block wrapping a string value), `Component` (an abstract composable
generative unit), `ModelOutputThunk` (a lazily-evaluated model response that is
intentionally *not* a `CBlock` subtype — keeping them separate makes Python
structural pattern matching order-insensitive),
`Context` and `ContextTurn` (stateful conversation history containers),
`TemplateRepresentation` (the structured rendering of a component for prompt
templates), `ImageBlock`, and `ModelToolCall`. Understanding these types is
the starting point for building custom components or sampling strategies.
"""

from __future__ import annotations

import abc
import asyncio
import base64
import binascii
import datetime
import enum
import logging
import threading
from collections import OrderedDict
from collections.abc import Callable, Coroutine, Iterable, Mapping
from copy import copy, deepcopy
from dataclasses import dataclass, field
from io import BytesIO
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    ParamSpec,
    Protocol,
    TypeVar,
    runtime_checkable,
)

if TYPE_CHECKING:
    import torch

import requests
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
        cache (bool): If `True`, the inference engine may store the KV cache for this block. Experimental.

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
            AssertionError: If `value` is not a valid base64-encoded PNG string.
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
            bool: `True` if the string decodes to a PNG image, `False` otherwise.
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
        """Creates an `ImageBlock` from a PIL image object.

        Converts the image to a base64-encoded PNG string and wraps it in a new
        `ImageBlock` instance.

        Args:
            image (PILImage.Image): The PIL image to encode.
            meta (dict[str, Any] | None): Optional metadata to associate with the block.

        Returns:
            ImageBlock: A new `ImageBlock` containing the base64-encoded PNG.
        """
        image_base64 = cls.pil_to_base64(image)
        return cls(image_base64, meta)

    def __repr__(self) -> str:
        """Provides a python-parsable representation of the block (usually)."""
        return f"ImageBlock({self.value}, {self._meta.__repr__()})"


class ImageUrlBlock(CBlock):
    """An `ImageUrlBlock` represents an image as a URL.

    Use this when the image is hosted remotely and you want to pass the URL
    directly to backends that support it (e.g. OpenAI). Backends that only
    accept base64-encoded images (e.g. Ollama) download and encode the image
    automatically, so the URL never has to be converted by hand.

    Args:
        value (str): A URL string pointing to the image.
        meta (dict[str, Any] | None): Optional metadata to associate with this image block.

    """

    def __init__(self, value: str, meta: dict[str, Any] | None = None):
        """Initialize ImageUrlBlock with a URL string.

        Raises:
            ValueError: If `value` does not look like a URL (does not start
                with `http://` or `https://`).
        """
        if not value.startswith(("http://", "https://")):
            raise ValueError(
                f"ImageUrlBlock requires an http:// or https:// URL; got: {value!r}"
            )
        super().__init__(value, meta)

    def resolve_base64(self) -> str:
        """Return the image as a base64-encoded PNG, downloading it once per URL.

        Backends that cannot pass a URL through (e.g. Ollama) need the image
        as base64. The download and encode result is memoized in a process-wide,
        URL-keyed cache, so re-using the URL across conversation turns — even
        via a freshly reconstructed block — does not re-fetch the image. This
        call is blocking; async callers should offload it with
        `asyncio.to_thread`.

        Returns:
            str: The base64-encoded PNG representation of the image at the URL.

        Raises:
            ValueError: If the response exceeds the size cap or the image
                cannot be downloaded or decoded.
        """
        return _cached_download_image_as_base64(str(self.value))

    def __repr__(self) -> str:
        """Provides a python-parsable representation of the block (usually)."""
        return f"ImageUrlBlock({self.value}, {self._meta.__repr__()})"


class AudioBlock(CBlock):
    """An `AudioBlock` represents audio as base64 data.

    The `format` parameter is optional when `value` is a data URI — the format
    is extracted automatically from the MIME type (e.g. `data:audio/wav;base64,...`
    yields `"wav"`). When `value` is raw base64, `format` is required.

    Args:
        value (str): A valid base64-encoded audio string, optionally with a data URI
            prefix (e.g. `"data:audio/wav;base64,..."`) or raw base64.
        format (str | None): The audio format, such as `"wav"` or `"mp3"`. If `None`,
            the format is auto-detected from the data URI prefix. Required when `value`
            is raw base64 with no data URI prefix.
        meta (dict[str, Any] | None): Optional metadata to associate with this audio block.

    """

    def __init__(
        self, value: str, format: str | None = None, meta: dict[str, Any] | None = None
    ):
        """Initialize AudioBlock with a base64-encoded audio string and format.

        When `value` is a data URI, `format` is auto-detected from the MIME type if
        not provided. When `value` is raw base64, `format` must be supplied explicitly.

        Raises:
            AssertionError: If `value` is not a valid base64-encoded audio string.
            ValueError: If `format` cannot be determined (not provided and not
                derivable from a data URI prefix), or if an explicit `format` is
                an empty string.
        """
        assert self.is_valid_base64_audio(value), (
            "Invalid base64 string representation of audio."
        )
        if format is None and "data:" in value and "base64," in value:
            mime = value.split(";")[0].split("data:")[1]  # e.g. "audio/wav"
            subtype = mime.split("/")[-1]  # e.g. "wav"
            # Normalise non-standard / legacy MIME subtypes to canonical format tokens.
            # OpenAI Chat Completions input_audio.format accepts only: wav, mp3.
            # Some mime subtypes need to be mapped (in particular mpeg -> mp3)
            # Unknown subtypes are passed through unchanged — other backends may
            # accept additional formats (e.g. flac, ogg) that OpenAI does not.
            _MIME_SUBTYPE_TO_OPENAI_FORMAT: dict[str, str] = {
                "x-wav": "wav",
                "wave": "wav",
                "mpeg": "mp3",
                "x-mpeg": "mp3",
                "x-mp3": "mp3",
                "x-flac": "flac",
            }
            format = _MIME_SUBTYPE_TO_OPENAI_FORMAT.get(subtype, subtype)
        if not (format and format.strip()):
            raise ValueError(
                "AudioBlock format must be a non-empty string (e.g. 'wav'). "
                "Pass it explicitly or use a data URI: data:audio/wav;base64,..."
            )
        super().__init__(value, meta)
        self.format = format

    @staticmethod
    def is_valid_base64_audio(s: str) -> bool:
        """Checks whether a string is valid base64-encoded audio payload data.

        Strips any data URI prefix before decoding. Adds padding characters if
        necessary to make the base64 string a valid length.

        Args:
            s (str): The string to validate, optionally prefixed with a data URI header.

        Returns:
            bool: `True` if the string decodes as base64, `False` otherwise.
        """
        try:
            if "data:" in s and "base64," in s:
                s = s.split("base64,")[1]

            s = s.strip()
            mod4 = len(s) % 4
            if mod4 > 0:
                s = s + "=" * (4 - mod4)

            base64.b64decode(s, validate=True)
            return True
        except (binascii.Error, ValueError):
            return False

    def __repr__(self) -> str:
        """Provides a python-parsable representation of the block (usually)."""
        return f"AudioBlock({self.value}, {self.format}, {self._meta.__repr__()})"


class AudioUrlBlock(CBlock):
    """An `AudioUrlBlock` represents audio as a URL.

    Args:
        value (str): A URL string pointing to the audio.
        format (str): The audio format, such as `"wav"` or `"mp3"`.
        meta (dict[str, Any] | None): Optional metadata to associate with this audio block.

    """

    def __init__(self, value: str, format: str, meta: dict[str, Any] | None = None):
        """Initialize AudioUrlBlock with a URL string and declared format.

        Raises:
            ValueError: If ``value`` does not look like a URL (does not start
                with ``http://`` or ``https://``), or if ``format`` is empty.
        """
        if not value.startswith(("http://", "https://")):
            raise ValueError(
                f"AudioUrlBlock requires an http:// or https:// URL; got: {value!r}"
            )
        if not format.strip():
            raise ValueError("AudioUrlBlock format must be a non-empty string.")
        super().__init__(value, meta)
        self.format = format

    def __repr__(self) -> str:
        """Provides a python-parsable representation of the block (usually)."""
        return f"AudioUrlBlock({self.value}, {self.format}, {self._meta.__repr__()})"


_IMAGE_DOWNLOAD_TIMEOUT_S: float = 10.0
"""Socket timeout (seconds) applied to image URL downloads."""

_IMAGE_DOWNLOAD_MAX_BYTES: int = 25 * 1024 * 1024
"""Maximum accepted size (bytes) of a downloaded image body."""

_IMAGE_CACHE_MAX_ENTRIES: int = 128
"""Maximum number of URL -> base64 entries retained by the download cache."""

_image_base64_cache: OrderedDict[str, str] = OrderedDict()
"""Process-wide LRU cache mapping image URLs to their base64-encoded PNG."""

_image_base64_cache_lock = threading.Lock()
"""Guards `_image_base64_cache` against concurrent `asyncio.to_thread` callers."""


def _cached_download_image_as_base64(url: str) -> str:
    """Download an image as base64, memoizing the result per URL.

    Wraps `_download_image_as_base64` with a bounded, thread-safe LRU cache
    keyed on the URL so the same image is fetched only once regardless of how
    many `ImageUrlBlock` instances reference it. The download runs outside the
    lock so concurrent fetches of distinct URLs still proceed in parallel;
    only the small cache read/write is serialized.

    Args:
        url: An `http://` or `https://` URL pointing to an image.

    Returns:
        str: The base64-encoded PNG representation of the image at the URL.

    Raises:
        ValueError: If the response exceeds the size cap or the image cannot
            be downloaded or decoded.
    """
    with _image_base64_cache_lock:
        cached = _image_base64_cache.get(url)
        if cached is not None:
            _image_base64_cache.move_to_end(url)  # mark as most-recently used
            return cached

    # Download outside the lock; two callers racing on a cold URL may both
    # fetch, but the result is identical and the last writer simply wins.
    encoded = _download_image_as_base64(url)

    with _image_base64_cache_lock:
        _image_base64_cache[url] = encoded
        _image_base64_cache.move_to_end(url)
        while len(_image_base64_cache) > _IMAGE_CACHE_MAX_ENTRIES:
            _image_base64_cache.popitem(last=False)  # evict least-recently used
    return encoded


def _download_image_as_base64(url: str) -> str:
    """Download an image from a URL and return it as a base64-encoded PNG string.

    Fetches the bytes at `url`, loads them through PIL to confirm they are a
    real image, and re-encodes the result as a base64 PNG so the output is
    consistent with `ImageBlock`'s expected format.

    A timeout bounds slow responses and the body is streamed with a size cap
    to guard against memory-exhaustion. This function is blocking; async
    callers should offload it with `asyncio.to_thread`.

    Args:
        url: An `http://` or `https://` URL pointing to an image.

    Returns:
        str: The base64-encoded PNG representation of the downloaded image.

    Raises:
        ValueError: If the response exceeds the size cap or the image cannot
            be downloaded or decoded.
    """
    try:
        with requests.get(  # scheme validated by caller
            url, timeout=_IMAGE_DOWNLOAD_TIMEOUT_S, stream=True
        ) as response:
            response.raise_for_status()
            declared = response.headers.get("Content-Length")
            if declared is not None and int(declared) > _IMAGE_DOWNLOAD_MAX_BYTES:
                raise ValueError(
                    f"Image at {url!r} exceeds the {_IMAGE_DOWNLOAD_MAX_BYTES}-byte limit"
                )
            # Stream so an undeclared/lying Content-Length can't exhaust memory.
            raw = response.raw.read(_IMAGE_DOWNLOAD_MAX_BYTES + 1, decode_content=True)
        if len(raw) > _IMAGE_DOWNLOAD_MAX_BYTES:
            raise ValueError(
                f"Image at {url!r} exceeds the {_IMAGE_DOWNLOAD_MAX_BYTES}-byte limit"
            )
        image = PILImage.open(BytesIO(raw))
    except (requests.RequestException, OSError, ValueError) as e:
        raise ValueError(
            f"Failed to download or decode image from URL {url!r}: {e}"
        ) from e
    return ImageBlock.pil_to_base64(image)


def make_image_block(
    src: str | PILImage.Image,
    *,
    convert_to_base64: bool = False,
    meta: dict[str, Any] | None = None,
) -> ImageBlock | ImageUrlBlock:
    """Create the appropriate image block from any supported image source.

    Dispatches on the type and shape of `src` so callers don't have to know
    whether they need an `ImageBlock` (base64-encoded) or an `ImageUrlBlock`
    (URL-referenced):

    - A PIL image is encoded to a base64 PNG and returned as an `ImageBlock`.
    - An `http://`/`https://` URL is returned as an `ImageUrlBlock`, unless
      `convert_to_base64=True`, in which case the image is downloaded and
      returned as an `ImageBlock`.
    - A base64-encoded PNG string (with or without a data URI prefix) is
      returned as an `ImageBlock`.

    Args:
        src: The image source — a PIL image, an image URL, or a base64-encoded
            PNG string.
        convert_to_base64: If `True` and `src` is a URL, download the image and
            return an `ImageBlock` instead of an `ImageUrlBlock`. Ignored for
            non-URL sources.
        meta: Optional metadata to associate with the returned block.

    Returns:
        ImageBlock | ImageUrlBlock: An `ImageBlock` for PIL images, base64
        strings, and downloaded URLs; an `ImageUrlBlock` for URLs when
        `convert_to_base64` is `False`.

    Raises:
        ValueError: If `src` is a string that is neither a valid URL nor a
            valid base64-encoded PNG, or if a URL download fails.
        TypeError: If `src` is not a PIL image or a string.
    """
    if isinstance(src, PILImage.Image):
        return ImageBlock.from_pil_image(src, meta)

    if isinstance(src, str):
        if src.startswith(("http://", "https://")):
            if convert_to_base64:
                return ImageBlock(_download_image_as_base64(src), meta)
            return ImageUrlBlock(src, meta)
        if ImageBlock.is_valid_base64_png(src):
            return ImageBlock(src, meta)
        raise ValueError(
            f"make_image_block could not interpret string source; expected an "
            f"http(s) URL or a base64-encoded PNG, got: {src!r}"
        )

    raise TypeError(
        f"make_image_block expects a PIL image or a string source, got: {type(src)!r}"
    )


S = typing_extensions.TypeVar("S", default=Any, covariant=True)
"""Used for class definitions for Component and ModelOutputThunk; also used for functions that don't accept CBlocks. Defaults to `Any`."""

C = typing_extensions.TypeVar("C", default=str)
"""Used for component typing in function parameters where the function takes a Component[C] and/or CBlock and can return a ModelOutputThunk[C]. Defaults to `str`."""


class ComponentParseError(Exception):
    """Raised by `Component.parse()` when the underlying parsing method throws an exception."""


@runtime_checkable
class Component(Protocol, Generic[S]):
    """A `Component` is a composite data structure that is intended to be represented to an LLM."""

    def parts(self) -> list[Component | CBlock | ModelOutputThunk]:
        """Returns the set of all constituent sub-components and content blocks of this `Component`.

        Returns:
            list[Component | CBlock | ModelOutputThunk]: A list of child `Component`, `CBlock`,
            or `ModelOutputThunk` objects that make up this component. The list may be empty for
            leaf components.

        Raises:
            NotImplementedError: If the concrete subclass has not overridden this method.
        """
        raise NotImplementedError("parts isn't implemented by default")

    def format_for_llm(self) -> TemplateRepresentation | str:
        """Formats the `Component` into a `TemplateRepresentation` or plain string for LLM consumption.

        Returns:
            TemplateRepresentation | str: A structured `TemplateRepresentation` (for components
            with tools, fields, or templates) or a plain string for simple components.

        Raises:
            NotImplementedError: If the concrete subclass has not overridden this method.
        """
        raise NotImplementedError("format_for_llm isn't implemented by default")

    def parse(self, computed: ModelOutputThunk) -> S:
        """Parses the expected type `S` from a given `ModelOutputThunk`.

        Delegates to the component's underlying `_parse` method and wraps any
        exception in a `ComponentParseError` for uniform error handling.

        Args:
            computed (ModelOutputThunk): The model output thunk whose value should be parsed.

        Returns:
            S: The parsed result produced by `_parse`, typed according to the component's type parameter.

        Raises:
            ComponentParseError: If the underlying `_parse` call raises any exception.
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
        ASYNC (int): The generation function is async-compatible; `avalue`/`astream` may be used.
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
        model: Requested model identifier.
        provider: Provider name (e.g. 'openai', 'ollama', 'huggingface', 'watsonx').
        ttfb_ms: Time to first token in milliseconds; None for non-streaming.
        streaming: Whether this generation used streaming mode.
        response_model: Model identifier reported on the response; may differ from the requested model.
        finish_reasons: Finish reason(s) reported on the response (typically one per choice).
        response_id: Provider-assigned identifier for the response.
        logits: Per-token processed logit scores (post-LogitsProcessor); None if not requested or unavailable.
        raw_logits: Per-token raw LM-head logits (pre-LogitsProcessor); None if not requested or unavailable.
    """

    usage: dict[str, Any] | None = None
    """Token usage following OpenAI API standard.

    Core fields: 'prompt_tokens', 'completion_tokens', 'total_tokens'.
    May include optional breakdown fields like 'completion_tokens_details'
    and 'prompt_tokens_details' (nested dicts with per-category token counts
    for reasoning, audio, caching, etc.).
    """

    model: str | None = None
    """Requested model identifier (e.g. 'gpt-4', 'llama2:7b', 'meta-llama/Llama-2-7b-hf').

    What the caller asked for. See `response_model` for what the provider
    actually served, which may differ for version aliases or routed deployments.
    """

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

    response_model: str | None = None
    """Model identifier reported on the response.

    May differ from the request-side `model` when the provider routes to a
    different deployment (e.g. fine-tunes, version aliases). `None` when the
    backend response does not surface a model field.
    """

    finish_reasons: list[str] | None = None
    """Finish reason(s) reported on the response.

    Typically a one-element list (single-choice completions). `None` when the
    backend response does not surface a finish reason.
    """

    response_id: str | None = None
    """Provider-assigned response identifier.

    Useful for cross-referencing logs/traces with provider-side records.
    `None` when the backend response does not carry an id.
    """

    logits: tuple[torch.Tensor, ...] | None = None
    """Per-token processed logit scores from the backend (post-LogitsProcessor).

    Populated when `ModelOption.LOGITS=True` and the backend supports it.
    These are logits after the LogitsProcessor chain (temperature, top-k/top-p,
    repetition penalty, etc.). For the HuggingFace backend this is a tuple of
    1-D tensors of shape `(vocab_size,)`, one per generated token. `None` if not
    requested, if the backend does not support logits, or when
    `ModelOption.STREAM=True`.
    """

    raw_logits: tuple[torch.Tensor, ...] | None = None
    """Per-token raw LM-head logits from the backend (pre-LogitsProcessor).

    Populated when `ModelOption.RAW_LOGITS=True` and the backend supports it.
    These are the unprocessed logits straight from the LM head, before any
    LogitsProcessor transforms. For the HuggingFace backend this is a tuple of
    1-D tensors of shape `(vocab_size,)`, one per generated token. `None` if not
    requested, if the backend does not support raw logits, or when
    `ModelOption.STREAM=True`.
    """


@dataclass
class RawProviderResponse:
    """Backend-native response payload from the provider's SDK.

    Reading these fields couples your code to a specific provider's response
    shape. For portable access prefer `mot.value`, `mot.parsed_repr`,
    `mot.tool_calls`, or `mot.generation`.

    Args:
        provider: Name of the provider that produced `response`; the same value
            as `mot.generation.provider`. Read it to know how to interpret
            `response`.
        response: Full SDK response object. Shape depends on `provider`.
        streamed_chunks: Per-chunk SDK objects from streaming responses.
            `None` for non-streaming requests.
    """

    provider: str | None = None
    response: Any | None = None
    streamed_chunks: list[Any] | None = None


@dataclass
class _CallInfo:
    """Originating-call data for a `ModelOutputThunk`.

    Preserved across `__copy__` / `__deepcopy__` because retries and sampling
    routinely need to re-issue or inspect the call that produced a thunk.

    Args:
        action: The component or block whose generation produced this thunk.
        context: The context passed to the originating generate call.
        model_options: Model options passed to the originating generate call.
        generation_id: Mellea-side hook correlation ID; distinct from the
            provider-assigned `GenerationMetadata.response_id`.
    """

    action: Component | CBlock | ModelOutputThunk | None = None
    context: list[Component | CBlock | ModelOutputThunk] | None = None
    model_options: dict[str, Any] | None = None
    generation_id: str | None = None


@dataclass
class _GenerationState:
    """In-flight computation machinery for a `ModelOutputThunk`.

    Reset to a fresh empty instance on `__copy__` / `__deepcopy__` — a copied
    thunk is a distinct (non-generating) object and must not share queues,
    tasks, or thread signals with the original.

    Args:
        queue: Single-consumer queue feeding `astream()` during generation.
        chunk_size: Minimum number of chunks to stream at a single time.
        first_chunk_received: Whether the first streamed chunk has arrived
            (gates time-to-first-byte recording).
        generate: The task driving generation. Linked to `generate_type`.
        generate_type: Determines which functions can resolve the thunk's value.
        generate_extra: Auxiliary generation task; currently only used by hf.
        cancel_hook: Optional cooperative-cancel hook called before asyncio task
            cancellation. Backends that run generation in a thread (e.g. Hugging
            Face via `asyncio.to_thread`) set this to a non-blocking callable
            (e.g. `threading.Event.set`) so the thread receives a stop signal
            before the task wrapper is cancelled. Must be non-blocking;
            exceptions are logged and suppressed. Copied thunks reset this to
            `None` — each computation owns its own thread signal.
        process: Backend coroutine that folds a streamed chunk into the thunk.
        post_process: Backend coroutine run once after the value is complete.
        on_computed: Coroutine run when the thunk becomes computed.
        start: Wall-clock start time of generation, for latency metrics.
    """

    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=20))
    chunk_size: int = 3
    first_chunk_received: bool = False
    generate: asyncio.Task[None] | None = None
    generate_type: GenerateType = GenerateType.NONE
    generate_extra: asyncio.Task[Any] | None = None
    cancel_hook: Callable[[], None] | None = None
    process: Callable[[ModelOutputThunk, Any], Coroutine] | None = None
    post_process: Callable[[ModelOutputThunk], Coroutine] | None = None
    on_computed: Callable[[ModelOutputThunk], Coroutine] | None = None
    start: datetime.datetime | None = None


class ModelOutputThunk(Generic[S]):
    """A `ModelOutputThunk` represents a lazily-evaluated model response. It is possible to instantiate one without the output being computed yet.

    Unlike `CBlock`, `ModelOutputThunk` is not a content-block input type — it is
    always an output.  The two classes are intentionally kept separate so that
    Python structural pattern matching (`match`/`case`) can distinguish them
    without order-sensitivity bugs.

    Args:
        value (str | None): The raw model output string, or `None` if not yet computed.
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
        if value is not None and not isinstance(value, str):
            raise TypeError(
                "value to a ModelOutputThunk should always be a string or None"
            )
        self._underlying_value = value
        self.cache: bool = False
        if meta is None:
            meta = {}
        self._meta = meta

        self.parsed_repr: S | None = parsed_repr
        """Will be non-`None` once computed."""

        # Set computed to True if a value is passed in.
        self._computed: bool = True if value is not None else False
        self._cancelled: bool = False

        # Additional fields that should be standardized across apis.
        self.tool_calls = tool_calls
        self.thinking: str | None = None
        self.generation: GenerationMetadata = GenerationMetadata()
        """Backend execution metadata populated during generation."""

        self.raw: RawProviderResponse = RawProviderResponse()
        """Backend-native provider response populated during generation."""

        # Originating-call data, preserved across copies. See `_CallInfo`.
        self._call = _CallInfo()

        # In-flight computation machinery, reset on copy. See `_GenerationState`.
        self._gen = _GenerationState()

        self._generate_log: GenerateLog | None = None
        # Soft-failure cause recorded by backends that return a placeholder
        # MOT instead of raising. Sibling to `_cancelled`.
        self._error: Exception | None = None

    def _record_ttfb(self) -> None:
        """Record time-to-first-byte if streaming and not yet recorded."""
        if (
            self.generation.streaming
            and not self._gen.first_chunk_received
            and self._gen.start is not None
        ):
            self.generation.ttfb_ms = (
                datetime.datetime.now() - self._gen.start
            ).total_seconds() * 1000
            self._gen.first_chunk_received = True

    async def cancel_generation(self, error: Exception | None = None) -> None:
        """Cancel an in-progress streaming generation, drain the queue, and fire the `generation_error` hook.

        Safe to call at any point during streaming. After this method returns,
        `is_computed()` is `True` and `value` contains whatever text was
        accumulated before cancellation.  Calling on an already-computed MOT
        is a no-op.

        Draining the internal queue after cancellation is necessary to release
        any `asyncio.Queue.put()` call that the generation task was blocked on
        (queue maxsize=20).

        Args:
            error: Optional cause attached to the `generation_error` hook
                payload.  When provided, this exception is delivered to
                subscribed plugins (e.g. the tracing plugin records it on the
                in-flight span) so observers reflect the actual reason for
                cancellation.  When `None`, a generic
                `RuntimeError("Generation cancelled")` is used.

        Raises:
            asyncio.CancelledError: Re-raised when the *calling* task itself is
                being cancelled (`asyncio.current_task().cancelling() > 0`).
                This prevents external cancellation (e.g. `asyncio.wait_for`
                timeout) from being silently absorbed while awaiting the inner
                generation task.
        """
        if self._computed:
            return

        def _drain() -> None:
            while not self._gen.queue.empty():
                try:
                    self._gen.queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

        # Signal any backend thread before cancelling the asyncio task wrapper
        # so the thread can stop cooperatively instead of running to completion.
        if self._gen.cancel_hook is not None:
            try:
                self._gen.cancel_hook()
            except Exception as hook_exc:
                logging.getLogger(__name__).warning(
                    "cancel_generation: cancel_hook raised (suppressed): %r", hook_exc
                )

        if self._gen.generate is not None and not self._gen.generate.done():
            self._gen.generate.cancel()

        if self._gen.generate_extra is not None and not self._gen.generate_extra.done():
            self._gen.generate_extra.cancel()

        # Drain before awaiting — unblocks any put() the task is stuck on.
        _drain()

        if self._gen.generate is not None:
            try:
                await self._gen.generate
            except asyncio.CancelledError:
                # Re-raise if the *outer* task is being cancelled (Python 3.11+
                # task.cancelling() > 0) so we don't silently absorb external
                # cancellation. For the inner task's own CancelledError (the
                # expected result of .cancel() above), cancelling() is 0.
                cur = asyncio.current_task()
                if cur is not None and cur.cancelling() > 0:
                    raise
            except Exception:
                pass

        if self._gen.generate_extra is not None:
            try:
                await self._gen.generate_extra
            except asyncio.CancelledError:
                cur = asyncio.current_task()
                if cur is not None and cur.cancelling() > 0:
                    raise
            except Exception:
                pass

        # Drain again for any final item the task put before terminating.
        _drain()

        if has_plugins(HookType.GENERATION_ERROR):
            from ..plugins.hooks.generation import GenerationErrorPayload

            recorded: Exception = (
                error if error is not None else RuntimeError("Generation cancelled")
            )
            await invoke_hook(
                HookType.GENERATION_ERROR,
                GenerationErrorPayload(
                    exception=recorded,
                    model_output=self,
                    generation_id=self._call.generation_id,
                ),
            )

        if self._underlying_value is None:
            self._underlying_value = ""
        self._cancelled = True
        self._computed = True

    @property
    def cancelled(self) -> bool:
        """`True` if :meth:`cancel_generation` ran to completion on this MOT.

        A normally-completed MOT leaves this `False`; only an actual
        cancellation via :meth:`cancel_generation` flips it.  Consumers holding
        a computed MOT can use this to distinguish a genuine result from one
        cut short (for example by a streaming requirement failure).
        """
        return self._cancelled

    @property
    def error(self) -> Exception | None:
        """Soft-failure cause recorded by the backend, or `None` on success.

        Sibling to `cancelled`: `error` is involuntary (the backend produced
        an unusable result without raising); `cancelled` is voluntary (a
        consumer stopped the generation via `cancel_generation`). The two
        are recorded independently.
        """
        return self._error

    @property
    def generate_log(self) -> GenerateLog | None:
        """The `GenerateLog` recorded for this generation.

        Returned by reference; mutating the returned object mutates the MOT's
        log.
        """
        return self._generate_log

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
        self.thinking = other.thinking
        self.generation = other.generation
        self.raw = other.raw
        self._generate_log = other._generate_log
        self._cancelled = other._cancelled
        self._error = other._error
        # _gen.cancel_hook is deliberately not copied: _copy_from swaps output
        # state, not backend-thread plumbing, which is tied to the original
        # computation.
        self._gen.cancel_hook = None

    def is_computed(self) -> bool:
        """Returns true only if this Thunk has already been filled.

        Returns:
            `True` if the thunk value has been set, `False` otherwise.
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

        if not self._gen.generate_type == GenerateType.ASYNC:
            raise RuntimeError(
                f"Cannot use `ModelOutputThunk.avalue()` when the generate function is using `{self._gen.generate_type.name}`"
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
            (self._call.model_options or {}).get("@@@stream@@@", False)
        )

        if not self._gen.generate_type == GenerateType.ASYNC:
            raise RuntimeError(
                f"Cannot use `ModelOutputThunk.astream()` when the generate function is using `{self._gen.generate_type.name}`"
            )
        # Beginning value
        beginning_length = (
            0 if self._underlying_value is None else len(str(self._underlying_value))
        )  # type: ignore

        # Type of the chunk depends on the backend.
        chunks: list[Any | None] = []
        while True:
            try:
                item = self._gen.queue.get_nowait()
                chunks.append(item)
                self._record_ttfb()
            except asyncio.QueueEmpty:
                # We've exhausted the current items in the queue.
                break

        # Make sure we always get the minimum chunk size.
        while len(chunks) <= self._gen.chunk_size:
            if len(chunks) > 0:
                if chunks[-1] is None or isinstance(chunks[-1], Exception):
                    break  # Hit sentinel value or an error.
                # We could switch to relying on the `done` / `finish_reason` field of chunks,
                # but that forces us to know about the chunk type here. Prefer sentinel values
                # for now.

            item = await self._gen.queue.get()
            chunks.append(item)
            self._record_ttfb()

        # Process the sentinel value if it's there.
        if chunks[-1] is None:
            chunks.pop()  # Remove the sentinel value.
            do_set_computed = True

            # Shouldn't be needed, but cancel the Tasks this ModelOutputThunk relied on.
            if self._gen.generate is not None:
                self._gen.generate.cancel()
            if self._gen.generate_extra is not None:
                # Covers an hf edge case. The task is done generating anything useful but isn't `done` yet.
                await self._gen.generate_extra
                self._gen.generate_extra.cancel()

            # If ModelOutputThunks get too bulky, we can do additional cleanup here
            # and set fields to None.

        elif isinstance(chunks[-1], Exception):
            # Fire generation_error hook (FIRE_AND_FORGET — does not block the raise)
            if has_plugins(HookType.GENERATION_ERROR):
                from ..plugins.hooks.generation import GenerationErrorPayload

                err_payload = GenerationErrorPayload(
                    exception=chunks[-1],
                    model_output=self,
                    generation_id=self._call.generation_id,
                )
                await invoke_hook(HookType.GENERATION_ERROR, err_payload)

            raise chunks[-1]

        for chunk in chunks:
            assert self._gen.process is not None
            await self._gen.process(self, chunk)

        if do_set_computed:
            assert self._underlying_value is not None
            self._computed = True

            assert self._gen.post_process is not None
            await self._gen.post_process(self)

            match self._call.action:
                case Component():
                    self.parsed_repr = self._call.action._parse(self)
                case CBlock():
                    assert self.value is not None, (
                        "value must be non-None since this thunk is computed"
                    )
                    self.parsed_repr = self.value  # type: ignore
                case ModelOutputThunk():
                    assert self.value is not None, (
                        "value must be non-None since this thunk is computed"
                    )
                    self.parsed_repr = self.value  # type: ignore
                case _:
                    raise ValueError(
                        "attempted to astream from a model output thunk with no originating action set"
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
                    (datetime.datetime.now() - self._gen.start).total_seconds() * 1000
                    if self._gen.start
                    else -1
                )
                post_payload = GenerationPostCallPayload(
                    prompt=prompt,
                    model_output=self,
                    latency_ms=latency_ms,
                    generation_id=self._call.generation_id,
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

    def __str__(self) -> str:
        """Stringifies the thunk value."""
        return self.value if self.value else ""

    def __repr__(self) -> str:
        """Provides a python-parsable representation (usually).

        Differs from CBlock because `._meta` can be very large for ModelOutputThunks.
        """
        return f"ModelOutputThunk({self.value})"

    def __copy__(self) -> ModelOutputThunk:
        """Returns a shallow copy of the ModelOutputThunk.

        Copies are post-generation: `_call` (originating-call data) is preserved
        while `_gen` (in-flight machinery) is left as the fresh instance from
        `__init__`. Copying an uncomputed thunk raises.

        Raises:
            RuntimeError: If the thunk has not been computed.
        """
        if not self._computed:
            raise RuntimeError(
                "Cannot copy an uncomputed ModelOutputThunk; copies are post-generation."
            )
        copied = ModelOutputThunk(
            self._underlying_value, self._meta, self.parsed_repr, self.tool_calls
        )

        # Check if the parsed_repr needs to be changed. A ModelOutputThunk's parsed_repr can point to
        # itself if the parsing didn't result in a new representation. It makes sense to update the
        # parsed_repr to the copied ModelOutputThunk in that case.
        if self.parsed_repr is self:
            copied.parsed_repr = copied  # type: ignore

        copied._computed = self._computed
        copied._cancelled = self._cancelled
        copied._error = self._error
        copied.thinking = self.thinking
        copied._generate_log = self._generate_log
        # _call is preserved; _gen is left as the fresh _GenerationState() from
        # __init__ so the copy shares no queues, tasks, or thread signals.
        copied._call = copy(self._call)
        copied.generation = copy(self.generation)
        copied.raw = copy(self.raw)
        return copied

    def __deepcopy__(self, memo: dict) -> ModelOutputThunk:
        """Returns a deep copy of the ModelOutputThunk.

        Like `__copy__` but deep-copies `_meta`, `parsed_repr`, and the
        originating-call data. `_gen` is left as the fresh instance from
        `__init__`. Copying an uncomputed thunk raises.

        Raises:
            RuntimeError: If the thunk has not been computed.
        """
        if not self._computed:
            raise RuntimeError(
                "Cannot deepcopy an uncomputed ModelOutputThunk; copies are post-generation."
            )
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
        deepcopied._cancelled = self._cancelled
        deepcopied._error = self._error
        deepcopied.thinking = self.thinking
        deepcopied._generate_log = copy(self._generate_log)
        # action is deep-copied; context and model_options are shallow-copied
        # (their items are immutable). _gen is left as the fresh instance from
        # __init__ so the copy shares no queues, tasks, or thread signals.
        deepcopied._call = _CallInfo(
            action=deepcopy(self._call.action),
            context=copy(self._call.context),
            model_options=copy(self._call.model_options),
            generation_id=self._call.generation_id,
        )
        deepcopied.generation = deepcopy(self.generation)
        deepcopied.raw = deepcopy(self.raw)
        return deepcopied


class ComputedModelOutputThunk(ModelOutputThunk[S]):
    """A `ComputedModelOutputThunk` is a `ModelOutputThunk` that is guaranteed to be computed.

    This subclass provides a clear type distinction between thunks that may need awaiting
    and those that are already computed. It should be returned from synchronous functions
    and sampling strategies to indicate that no awaiting is needed.

    Uses zero-copy class reassignment: calling `ComputedModelOutputThunk(thunk)` reassigns
    the thunk's `__class__` to `ComputedModelOutputThunk` without creating a new object.

    Args:
        thunk: A fully-computed `ModelOutputThunk` whose class will be reassigned.
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
        """The raw string value produced by the model.

        When `format=` was passed to the generating call, this is a JSON
        string matching the declared schema — not a parsed model instance.
        Use `MyModel.model_validate_json(str(result))` to get a typed instance.
        """
        return self._underlying_value  # type: ignore

    @value.setter
    def value(self, v: str):
        """Sets the value of the block."""
        self._underlying_value = v

    def is_computed(self) -> Literal[True]:
        """Returns `True` since thunk is always computed.

        Returns:
            Always `True`.
        """
        return True


@dataclass
class ContextTurn:
    """A turn of model input and model output.

    Args:
        model_input (CBlock | Component | ModelOutputThunk | None): The input component or content block for this turn,
            or `None` for an output-only partial turn.
        output (ModelOutputThunk | None): The model's output thunk for this turn,
            or `None` for an input-only partial turn.

    """

    model_input: CBlock | Component | ModelOutputThunk | None
    output: ModelOutputThunk | None


ContextT = TypeVar("ContextT", bound="Context")


class Context(abc.ABC):
    """A `Context` is used to track the state of a `MelleaSession`.

    A context is immutable. Every alteration leads to a new context.

    Attributes:
        is_root_node (bool): `True` when this context is the root (empty) node of the linked list.
        previous_node (Context | None): The context node from which this one was created,
            or `None` for the root node.
        node_data (Component | CBlock | ModelOutputThunk | None): The data associated with this context node,
            or `None` for the root node.
        is_chat_context (bool): Whether this context operates in chat (multi-turn) mode.
    """

    _previous: Context | None
    _data: Component | CBlock | ModelOutputThunk | None
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
        cls: type[ContextT],
        previous: Context,
        data: Component | CBlock | ModelOutputThunk,
    ) -> ContextT:
        """Constructs a new context node linked to an existing context node.

        Args:
            previous (Context): The existing context to extend.
            data (Component | CBlock | ModelOutputThunk): The component, content block, or model output to associate with the new node.

        Returns:
            ContextT: A new context instance whose `previous_node` is `previous`.
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

    def new_instance(self) -> Context:
        """Return a new empty root context, preserving any subclass configuration.

        The base implementation calls `reset_to_new()`, which returns a bare
        instance with no history and no config.  Subclasses that carry
        configuration (e.g. `ChatContext` with `model_id` and `window_size`)
        should override this to propagate their config into the fresh instance.

        Returns:
            Context: A freshly initialised root context of the same type.
        """
        return self.reset_to_new()

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
    def node_data(self) -> Component | CBlock | ModelOutputThunk | None:
        """Returns the data associated with this context node.

        Internal use: Users should not need to use this property.
        """
        return self._data

    @property
    def is_chat_context(self) -> bool:
        """Returns whether this context is a chat context."""
        return self._is_chat_context

    # User functions below this line.

    def as_list(
        self, last_n_components: int | None = None
    ) -> list[Component | CBlock | ModelOutputThunk]:
        """Returns a list of context components sorted from earliest (first) to most recent (last).

        If `last_n_components` is `None`, then all components are returned.

        Args:
            last_n_components (int | None): Maximum number of most-recent components to include.
                Pass `None` to return the full history.

        Returns:
            list[Component | CBlock | ModelOutputThunk]: Components in chronological order (oldest first).
        """
        context_list: list[Component | CBlock | ModelOutputThunk] = []
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

    def actions_for_available_tools(
        self,
    ) -> list[Component | CBlock | ModelOutputThunk] | None:
        """Provides a list of actions to extract tools from for use during generation.

        Returns `None` if it is not possible to construct such a list. Can be used to make
        the available tools differ from the tools of all the actions in the context. Can be
        overridden by subclasses.

        Returns:
            list[Component | CBlock | ModelOutputThunk] | None: The list of actions whose tools should be made
            available during generation, or `None` if unavailable.
        """
        return self.view_for_generation()

    def last_output(self, check_last_n_components: int = 3) -> ModelOutputThunk | None:
        """Returns the most recent `ModelOutputThunk` found within the last N context components.

        Args:
            check_last_n_components (int): Number of most-recent components to search through.
                Defaults to 3.

        Returns:
            ModelOutputThunk | None: The most recent output thunk, or `None` if none is found
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
            The most recent turn, or `None` if the context is empty.
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
    def add(self, c: Component | CBlock | ModelOutputThunk) -> Context:
        """Returns a new context obtained by appending `c` to this context.

        Args:
            c (Component | CBlock | ModelOutputThunk): The component, content block, or model output to add to the context.

        Returns:
            Context: A new context node with `c` as its data and this context as its previous node.
        """
        # something along ....from_previous(self, c)
        ...

    @abc.abstractmethod
    def view_for_generation(self) -> list[Component | CBlock | ModelOutputThunk] | None:
        """Provides a linear list of context components to use for generation.

        Returns `None` if it is not possible to construct such a list (e.g., the context
        is in an inconsistent state). Concrete subclasses define the ordering and filtering logic.

        Returns:
            list[Component | CBlock | ModelOutputThunk] | None: An ordered list of components suitable for passing
            to a backend, or `None` if generation is not currently possible.
        """
        ...


P = ParamSpec("P")
R = TypeVar("R")


class AbstractMelleaTool(abc.ABC, Generic[P, R]):
    """Abstract base class for Mellea Tool with parameter and return type support.

    Type parameters:
        P: Parameter specification for the tool's callable (via ParamSpec)
        R: Return type of the tool

    Attributes:
        name (str): The unique name used to identify the tool in JSON descriptions and tool-call dispatch.
        as_json_tool (dict[str, Any]): A JSON-serialisable description of the tool, compatible with
            the function-calling schemas expected by supported inference backends.
    """

    name: str
    """Name of the tool."""

    @abc.abstractmethod
    def run(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Executes the tool with the provided arguments and returns the result.

        Args:
            *args: Positional arguments forwarded to the tool implementation.
            **kwargs: Keyword arguments forwarded to the tool implementation.

        Returns:
            R: The result produced by the tool; the concrete type depends on the implementation.
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
            keyed by the tool's function name. Defaults to `None`.
        fields (list[Any] | None): An optional ordered list of field values for positional templates.
        template (str | None): An optional Jinja2 template string to use when rendering.
        template_order (list[str] | None): An optional ordering hint for template sections/keys.
        images (list[ImageBlock | ImageUrlBlock] | None): Optional list of image blocks associated with this representation.
        audio (list[AudioBlock | AudioUrlBlock] | None): Optional list of audio blocks associated
            with this representation.

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
    images: list[ImageBlock | ImageUrlBlock] | None = None
    audio: list[AudioBlock | AudioUrlBlock] | None = None


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
        action (Component | CBlock | ModelOutputThunk | None): The component or block that triggered the generation.
        result (ModelOutputThunk | None): The `ModelOutputThunk` produced by this generation call.
        is_final_result (bool | None): Whether this log entry corresponds to the definitive final result.
        extra (dict[str, Any] | None): Arbitrary extra metadata to attach to the log entry.

    """

    date: datetime.datetime | None = None
    prompt: str | list[dict] | None = None
    backend: str | None = None
    model_options: dict[str, Any] | None = None
    model_output: Any | None = None
    action: Component | CBlock | ModelOutputThunk | None = None
    result: ModelOutputThunk | None = None
    is_final_result: bool | None = False
    extra: dict[str, Any] | None = None


@dataclass
class ModelToolCall:
    """A dataclass for capturing the tool calls a model wants to make.

    Provides a unified way to call tools post generation.

    Args:
        name (str): The name of the tool the model requested to call.
        func (AbstractMelleaTool): The `AbstractMelleaTool` instance that will be invoked.
        args (Mapping[str, Any]): The keyword arguments the model supplied for the tool call.

    """

    name: str
    func: AbstractMelleaTool
    args: Mapping[str, Any]

    def call_func(self) -> Any:
        """Invokes the tool represented by this object and returns the result.

        Returns:
            Any: The value returned by `func.run(**args)`; the concrete type depends on the tool.
        """
        return self.func.run(**self.args)


def blockify(
    s: str | CBlock | Component | ModelOutputThunk,
) -> CBlock | Component | ModelOutputThunk:
    """Turn a raw string into a `CBlock`, leaving `CBlock`, `Component`, and `ModelOutputThunk` objects unchanged.

    Args:
        s: A plain string, `CBlock`, `Component`, or `ModelOutputThunk` to normalise.

    Returns:
        A `CBlock` wrapping `s` if it was a string; otherwise `s` unchanged.

    Raises:
        Exception: If `s` is not a `str`, `CBlock`, `Component`, or `ModelOutputThunk`.
    """
    # noinspection PyUnreachableCode
    match s:
        case str():
            return CBlock(s)
        case CBlock():
            return s
        case Component():
            return s
        case ModelOutputThunk():
            return s
        case _:
            raise Exception("Type Error")


def get_images_from_component(c: Component) -> None | list[ImageBlock | ImageUrlBlock]:
    """Return the images attached to a `Component`, or `None` if absent or empty.

    Args:
        c: The `Component` whose `images` attribute is inspected.

    Returns:
        A non-empty list of `ImageBlock` or `ImageUrlBlock` objects if the
        component has an `images` attribute with at least one element;
        `None` otherwise.
    """
    if hasattr(c, "images"):
        imgs = c.images  # type: ignore
        if imgs is not None:
            assert isinstance(imgs, list), "images field must be a list."
            assert all(isinstance(im, (ImageBlock, ImageUrlBlock)) for im in imgs), (
                "all elements of images list must be ImageBlock or ImageUrlBlock."
            )
            if len(imgs) == 0:
                return None
            else:
                return imgs
        else:
            return None
    else:
        return None


def get_audio_from_component(c: Component) -> None | list[AudioBlock | AudioUrlBlock]:
    """Return the audio attached to a `Component`, or `None` if absent or empty.

    Args:
        c: The `Component` whose `audio` attribute is inspected.

    Returns:
        A non-empty list of ``AudioBlock`` or ``AudioUrlBlock`` objects if the
        component has an ``audio`` attribute with at least one element;
        ``None`` otherwise.
    """
    if hasattr(c, "audio"):
        audio = c.audio  # type: ignore
        if audio is not None:
            assert isinstance(audio, list), "audio field must be a list."
            assert all(isinstance(a, (AudioBlock, AudioUrlBlock)) for a in audio), (
                "all elements of audio list must be AudioBlock or AudioUrlBlock."
            )
            if len(audio) == 0:
                return None
            else:
                return audio
        else:
            return None
    else:
        return None
