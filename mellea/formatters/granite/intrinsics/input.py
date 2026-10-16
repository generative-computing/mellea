# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Classes and functions that implement common aspects of input processing for intrinsics."""

# Standard
import json
import pathlib
import string

# Where AloraActivationError lives without creating an import cycle: the
# adapters package imports formatters for other purposes, but not this module.
from ....backends.adapters._core import AloraActivationError
from ....core.utils import MelleaLogger

# First Party
from ..base.io import ChatCompletionRewriter
from ..base.optional import import_optional
from ..base.types import (
    ChatCompletion,
    DocumentMessage,
    SystemMessage,
    UserMessage,
    VLLMExtraBody,
)

# Local
from .constants import TOP_LOGPROBS, TOP_LOGPROBS_MAX
from .util import make_config_dict

_MISSING_STATE = object()
"""Sentinel distinguishing "no `state` kwarg passed" from `state=None`."""

_SYSTEM_REPLACEMENT_WARNED: set[tuple[str, str]] = set()
"""(adapter name, dropped system prompt) pairs already warned about.

The rewriter is constructed per call and has no backend identity, so "warn
once per backend" is approximated by warning once per distinct
(adapter, replaced prompt) pair per process."""


class _InvocationMarkerError(AloraActivationError):
    """The invocation marker does not occur exactly once in the rewritten request.

    A private subclass so the marker-count guard (D6) raises the existing
    "declared invocation sequence absent" class; every catch site for
    `AloraActivationError` and `ValueError` keeps working; while carrying
    the marker-count message the PEFT-flavoured base `__str__` cannot. The
    base class's four PEFT-specific attributes are present but empty; the
    single-string `args` shape keeps pickle/deepcopy working.
    """

    def __init__(self, message: str) -> None:
        """Initialize with the marker-count message."""
        self.capability_name = ""
        self.base_model_name = ""
        self.qualified_name = ""
        self.invocation_tokens = ()
        ValueError.__init__(self, message)

    def __str__(self) -> str:
        """Return the marker-count message."""
        return str(self.args[0])


def _needs_logprobs(transformations: list | None) -> bool:
    """Check whether input processing for a model needs to specify logprobs.

    Subroutine to check whether input processing for a model needs to specify logprobs
    in the chat completion arguments.

    Args:
        transformations (list | None): Contents of the `transformations` field from
            the intrinsic's YAML configuration.

    Returns:
        bool: `True` if this intrinsic produces a field for which logprobs must be
            enabled for downstream result decoding to succeed.
    """
    if transformations is None:
        return False
    return any(t["type"] in ("likelihood", "first_token") for t in transformations)


def _first_token_labels(transformations: list | None) -> list[str] | None:
    """Extract the configured label pool of a `first_token` transformation.

    Args:
        transformations (list | None): Contents of the `transformations` field from
            the intrinsic's YAML configuration.

    Returns:
        list[str] | None: The `labels` list of the first `first_token`
        transformation, or `None` when the config declares none.
    """
    if transformations is None:
        return None
    for t in transformations:
        if t["type"] == "first_token":
            labels = t.get("labels")
            return labels if isinstance(labels, list) else None
    return None


def _format_fields(template: str) -> set[str]:
    """Return the names of the placeholders a format template references.

    Args:
        template (str): A `str.format`-style template.

    Returns:
        set[str]: Names of all `{name}` placeholders in *template*.
    """
    return {
        field_name
        for _, field_name, _, _ in string.Formatter().parse(template)
        if field_name is not None
    }


def _render_state_value(state: object) -> str:
    """Render a decision `state` value the way the trained layout expects.

    Mirrors the reference implementation's `text_of` plus empty sentinel:
    strings pass through; dicts and lists render as compact JSON with
    `ensure_ascii=False`; `""`, `None`, `{}`, and `[]` render as `"(empty)"`.

    Args:
        state (object): The state value passed by the caller.

    Returns:
        str: Rendered state text.
    """
    if state in ("", None, {}, []):
        return "(empty)"
    if isinstance(state, (dict, list)):
        return json.dumps(state, ensure_ascii=False)
    return str(state)


def sentence_delimiter(tag: str, sentence_num: int) -> str:
    """Return a tag string that identifies the beginning of the indicated sentence.

    Args:
        tag: Tag string prefix, e.g. `"i"` or `"c"`.
        sentence_num: Zero-based index of the sentence.

    Returns:
        Tag string (including trailing space) that identifies the beginning of
        the indicated sentence in sentence-tagged text.
    """
    return f"<{tag}{sentence_num}> "


def mark_sentence_boundaries(
    split_strings: list[list[str]], tag_prefix: str, index: int = 0
) -> tuple[list[str], int]:
    """Modify input strings by inserting sentence boundary markers.

    Modify one or more input strings by inserting a tag in the form
    `<[prefix][number]>`
    at the location of each sentence boundary.

    Args:
        split_strings: Input string(s), pre-split into sentences.
        tag_prefix: String to place before the number part of each tagged
            sentence boundary.
        index: Starting index for sentence numbering. Defaults to 0. Pass a
            non-zero value to continue numbering from a prior call.

    Returns:
        tuple[list[str], int]: Tuple of (list of input strings with all sentence
        boundaries marked, next available index after the last sentence).
    """
    result: list[str] = []
    for sentences in split_strings:
        to_concat = []
        for sentence in sentences:
            to_concat.append(f"{sentence_delimiter(tag_prefix, index)}{sentence}")
            index += 1
        result.append(" ".join(to_concat))
    return result, index


def move_documents_to_message(
    chat_completion: ChatCompletion | dict, how: str = "string"
) -> ChatCompletion | dict:
    """Move RAG documents from extra_body to first message.

    By convention, our canned JSON requests place RAG documents in extra_body/documents.
    Some models do not accept this parameter.
    This function edits a request by putting the documents into the first turn of the
    messages.

    Args:
        chat_completion: A chat completion request as dataclass or parsed JSON.
        how: How to serialize the documents; supported values are `"string"`,
            `"json"`, and `"roles"`.

    Returns:
        A copy of `chat_completion` with any documents under `extra_body`
        moved to the first message. Returned type will be the same as the input type.
        May return original object if no edits are necessary.

    Raises:
        TypeError: If `chat_completion` is not a :class:`ChatCompletion` or
            `dict`.
        ValueError: If `how` is not one of `"string"`, `"json"`, or
            `"roles"`.
    """
    if isinstance(chat_completion, ChatCompletion):
        should_return_dataclass = True
    elif isinstance(chat_completion, dict):
        should_return_dataclass = False
        chat_completion = ChatCompletion.model_validate(chat_completion)
    else:
        raise TypeError(
            f"Unexpected type '{type(chat_completion)}' for 'chat_completion' "
            f"argument. Should be ChatCompletion or dict."
        )

    if (
        chat_completion.extra_body is not None
        and chat_completion.extra_body.documents is not None
    ):
        docs_list = chat_completion.extra_body.documents

        doc_message_text = ""
        if how == "string":
            doc_text = "\n\n".join(
                [f"[Document {d.doc_id}]\n{d.text}" for d in docs_list]
            )
            doc_message_text = (
                "You have access to the following documents:\n\n" + doc_text
            )
        elif how == "json":
            doc_message_text = json.dumps([d.model_dump() for d in docs_list])
        elif how == "roles":
            doc_roles = []
            for doc in docs_list:
                doc_role = DocumentMessage(
                    role=f"document {doc.doc_id}", content=doc.text
                )
                doc_roles.append(doc_role)
        else:
            raise ValueError(f"Unknown document serialization method '{how}'")

        if how == "roles":
            new_messages = doc_roles + chat_completion.messages
        else:
            new_messages = [
                UserMessage(content=doc_message_text),
                *chat_completion.messages,
            ]

        # Round-trip through parsed JSON so that extra_body.documents will be unset
        new_extra_body = VLLMExtraBody.model_validate(
            {
                k: v
                for k, v in chat_completion.extra_body.model_dump().items()
                if k != "documents"
            }
        )
        chat_completion = chat_completion.model_copy(
            update={"messages": new_messages, "extra_body": new_extra_body}
        )

    if should_return_dataclass:
        return chat_completion
    return chat_completion.model_dump()


class IntrinsicsRewriter(ChatCompletionRewriter):
    """General-purpose chat completion rewriter for intrinsics.

    General-purpose chat completion rewriter for use with models that implement
    LLM intrinsics. Reads parameters of the model's input and output formats
    from a YAML configuration file and edits the input chat completion appropriately.

    Args:
        config_file (str | pathlib.Path | None): Path to the YAML configuration file for the
            target intrinsic. Mutually exclusive with `config_dict`.
        config_dict (dict | None): Inline configuration dictionary. Mutually exclusive with
            `config_file`.
        model_name (str | None): Optional model name used to locate model-specific overrides
            within the configuration.

    Attributes:
        config (dict): Parsed YAML configuration file for the target intrinsic.
        response_format (dict): JSON Schema of the expected response format.
        parameters (dict): Additional parameters (key-value pairs) that this
            rewriter adds to all chat completion requests.
        extra_body_parameters (dict): Extended vLLM-specific parameters that go
            under the `extra_body` element of each request. These are merged
            with any existing `extra_body` content in incoming requests.
        instruction (str | None): Optional instruction template. When present,
            a new user message is appended with the formatted instruction.
        sentence_boundaries (dict[str, str] | None): Optional sentence-boundary
            marking specification, mapping location strings (`"last_message"`
            or `"documents"`) to marker prefixes (e.g. `"c"` produces
            `<c0>`, `<c1>`, …).
        docs_as_message (str | None): Optional specification for moving
            documents from `extra_body/documents` to a user message at the
            start of the messages list. Value must be `"string"`, `"json"`,
            or `"roles"`.
        system_prompt (str | None): Optional trained system prompt. When
            present, all system-role messages in the incoming request are
            dropped and this prompt is inserted as the first message.
        state_template (str | None): Optional template for the state turn,
            rendered as a user message immediately before the instruction
            message from the reserved `state` format kwarg.
        invocation_text (str | None): Optional activation marker. When present,
            the marker must occur exactly once across the final messages.
    """

    config: dict
    """Parsed YAML configuration file for the target intrinsic."""

    response_format: dict
    """JSON Schema of expected response format"""

    parameters: dict
    """Additional parameters (key-value pairs) that this rewriter adds to all chat
    completion requests."""

    extra_body_parameters: dict
    """Extended vLLM-specific parameters that go under the `extra_body` element of
    the parameters field. These parameters need to be merged with any `extra_body`
    content that is present in incoming requests."""

    instruction: str | None
    """Optional instruction template. If present, a new user message will be added with
    the indicated instruction."""

    sentence_boundaries: dict[str, str] | None
    """
    Optional sentence boundary marking specification, as a mapping from sentence
    location (i.e. "last_message", "documents") to marker string (i.e. "c" for the
    first sentence to be marked with "<c0>").
    """

    docs_as_message: str | None
    """
    Optional specification for moving documents from `extra_body/documents` to a
    user message at the beginning of the messages list. Value specifies how to serialize
    the documents into the message: "string" or "json".
    """

    system_prompt: str | None
    """Optional trained system prompt; replaces all incoming system messages."""

    state_template: str | None
    """Optional template for the state turn, taking the reserved `state` kwarg."""

    invocation_text: str | None
    """Optional activation marker that must appear exactly once per request."""

    def __init__(
        self,
        /,
        config_file: str | pathlib.Path | None = None,
        config_dict: dict | None = None,
        model_name: str | None = None,
    ):
        """Initialize IntrinsicsRewriter from a YAML config file or dict."""
        config = make_config_dict(config_file, config_dict)
        if config is None:
            raise ValueError("config cannot be None")
        self.config = config

        if self.config["parameters"] is not None and not isinstance(
            self.config["parameters"], dict
        ):
            raise TypeError(
                f"'parameters' field must be null (~ in YAML) or contain a mapping "
                f"from chat completion parameter name to value. Current value "
                f"{self.config['parameters']}"
            )

        # Split out parameters that go in extra_body
        self.parameters = self.config["parameters"] or {}
        self.extra_body_parameters = {}
        if "extra_body" in self.parameters:
            self.extra_body_parameters.update(self.parameters["extra_body"])
            del self.parameters["extra_body"]

        # Check if we're supposed to override model name
        if model_name is not None:
            self.parameters["model"] = model_name
        elif self.config["model"]:
            self.parameters["model"] = self.config["model"]

        # Compute additional parameters we need to add to every request
        if _needs_logprobs(self.config["transformations"]):
            self.parameters["logprobs"] = True
            if "top_logprobs" not in self.parameters:
                # Size the request to the label pool for one-token readouts,
                # capped at the deployment maximum; a value configured in the
                # io.yaml `parameters` always wins over our sizing.
                labels = _first_token_labels(self.config["transformations"])
                self.parameters["top_logprobs"] = (
                    min(len(labels), TOP_LOGPROBS_MAX) if labels else TOP_LOGPROBS
                )
        self.instruction = self.config["instruction"]

        # Trained prompt layout of one-token decision capabilities.
        self.system_prompt = self.config["system_prompt"]
        self.state_template = self.config["state_template"]
        self.invocation_text = self.config["invocation_text"]
        self._first_token = (
            _first_token_labels(self.config["transformations"]) is not None
        )

        # `state` is a reserved format kwarg: it is consumed by
        # `state_template` rendering, never by the instruction template. Reject
        # an instruction that references it at construction time, so the
        # mismatch surfaces on load rather than as a per-call format error.
        if self.instruction is not None and "state" in _format_fields(self.instruction):
            raise ValueError(
                "The intrinsic's 'instruction' template references the reserved "
                "placeholder '{state}'. State content is rendered by the "
                "'state_template' field instead; remove '{state}' from the "
                "instruction template."
            )

        if self.config["response_format"] is not None:
            self.extra_body_parameters["structured_outputs"] = {
                "json": self.config["response_format"]
            }

        self.sentence_boundaries = self.config["sentence_boundaries"]
        if self.sentence_boundaries:
            # Sentence boundary detection requires nltk
            with import_optional("nltk"):
                # Third Party
                import nltk
            self.sentence_splitter = nltk.tokenize.punkt.PunktSentenceTokenizer()
            if not isinstance(self.sentence_boundaries, dict):
                raise TypeError(
                    f"'sentence_boundaries', if present, must be a mapping from "
                    f"location (last_message/documents) to mapping prefix string. "
                    f"Received {self.sentence_boundaries}."
                )
            for k, v in self.sentence_boundaries.items():
                if k not in ("last_message", "documents", "all_but_last_message"):
                    raise ValueError(
                        f"Unexpected location '{k}' in 'sentence_boundaries' field. "
                        f"Value should be 'last_message', 'documents', or "
                        f"'all_but_last_message'."
                    )
                if not isinstance(v, str):
                    raise TypeError(
                        f"Prefix string for location '{k}' in sentence_boundaries "
                        f"field set to {v}, which is not a string."
                    )

        self.docs_as_message = self.config["docs_as_message"]
        valid_docs_as_message = ["string", "json", "roles"]
        if self.docs_as_message and self.docs_as_message not in valid_docs_as_message:
            raise ValueError(
                f"docs_as_message parameter set to '{self.docs_as_message}', which is "
                f"not one of the valid options {valid_docs_as_message}"
            )

    def _mark_sentence_boundaries(
        self, chat_completion: ChatCompletion
    ) -> ChatCompletion:
        """Subroutine of :func:`_transform()` that handles sentence boundary detection.

        Should be applied BEFORE adding any instruction messages to the input.

        Args:
            chat_completion (ChatCompletion): Argument to `_transform()`.

        Returns:
            ChatCompletion: Copy of the original chat completion with sentence
                boundaries marked in the last message and in documents.
        """
        # Mark sentence boundaries in the last message.
        # last_message uses its own numbering starting from 0, independent of
        # the numbering used for documents and conversation history.
        if self.sentence_boundaries and "last_message" in self.sentence_boundaries:
            messages = chat_completion.messages.copy()  # Do not modify input!
            last_message_as_sentences = list(
                self.sentence_splitter.tokenize(messages[-1].content)
            )
            last_message_tag = self.sentence_boundaries["last_message"]
            if last_message_tag:
                rewritten_texts, _ = mark_sentence_boundaries(
                    [last_message_as_sentences], last_message_tag
                )
                messages[-1].content = rewritten_texts[0]
                chat_completion = chat_completion.model_copy(
                    update={"messages": messages}
                )

        # documents and all_but_last_message share a continuous numbering.
        index = 0

        # Mark sentence boundaries in documents if present
        if (
            chat_completion.extra_body
            and chat_completion.extra_body.documents
            and self.sentence_boundaries
            and "documents" in self.sentence_boundaries
        ):
            docs_as_sentences = [
                list(self.sentence_splitter.tokenize(d.text))
                for d in chat_completion.extra_body.documents
            ]
            # The documents input to the model consists of the original documents
            # with each sentence boundary marked with <c0>, <c1>, ... <ck-1>,
            # where `k` is the number of sentences in ALL documents.
            documents_tag = self.sentence_boundaries["documents"]
            if documents_tag:
                rewritten_texts, index = mark_sentence_boundaries(
                    docs_as_sentences, documents_tag, index
                )
                rewritten_docs = [
                    doc.model_copy(update={"text": text})
                    for doc, text in zip(
                        chat_completion.extra_body.documents,
                        rewritten_texts,
                        strict=True,
                    )
                ]
                # Don't modify original input
                extra_body = chat_completion.extra_body.model_copy(
                    update={"documents": rewritten_docs}
                )
                chat_completion = chat_completion.model_copy(
                    update={"extra_body": extra_body}
                )

        # Mark sentence boundaries in conversation history if requested.
        # Uses the same numbering as documents, continuing from where they left off.
        if (
            self.sentence_boundaries
            and "all_but_last_message" in self.sentence_boundaries
        ):
            history_tag = self.sentence_boundaries["all_but_last_message"]
            if history_tag:
                messages = chat_completion.messages.copy()  # Do not modify input!
                for i, message in enumerate(messages[:-1]):
                    msg_as_sentences = list(
                        self.sentence_splitter.tokenize(message.content)
                    )
                    rewritten_texts, index = mark_sentence_boundaries(
                        [msg_as_sentences], history_tag, index
                    )
                    messages[i] = message.model_copy(
                        update={"content": rewritten_texts[0]}
                    )
                chat_completion = chat_completion.model_copy(
                    update={"messages": messages}
                )

        return chat_completion

    def _reject_decision_incompatibilities(
        self, chat_completion: ChatCompletion
    ) -> None:
        """Reject tools and documents on one-token decision calls.

        Both would be appended to the trained system prompt by the chat
        template of a Switch-style model, changing the layout the decision
        adapter was trained on.

        Args:
            chat_completion (ChatCompletion): Argument to `_transform()`.

        Raises:
            ValueError: If the request carries tools or RAG documents.
        """
        if chat_completion.tools:
            raise ValueError(
                "Decision calls do not support tools: a tool list would be "
                "appended to the trained system prompt by the chat template "
                "and break the one-token readout. Remove tools from the "
                "context and model options before calling a decision "
                "helper."
            )
        documents = (
            chat_completion.extra_body.documents
            if chat_completion.extra_body is not None
            else None
        )
        if documents:
            raise ValueError(
                "Decision calls do not support documents: a document list "
                "would be appended to the trained system prompt by the chat "
                "template and break the one-token readout. Pass the document "
                "text as the state instead."
            )

    def _replace_system_messages(self, messages: list, system_prompt: str) -> list:
        """Drop all system-role messages and insert the configured one at index 0.

        Args:
            messages (list): Incoming messages.
            system_prompt (str): The configured system prompt to insert.

        Returns:
            list: New message list with the configured system prompt first.
        """
        dropped = [m for m in messages if m.role == "system"]
        if dropped:
            self._warn_system_replacement(dropped)
        kept = [m for m in messages if m.role != "system"]
        return [SystemMessage(content=system_prompt), *kept]

    def _warn_system_replacement(self, dropped: list) -> None:
        """Warn, once per adapter and replaced prompt, that a system prompt was replaced.

        Args:
            dropped (list): The system messages that are being dropped.
        """
        adapter_key = str(self.parameters.get("model") or "unknown")
        for message in dropped:
            key = (adapter_key, message.content)
            if key in _SYSTEM_REPLACEMENT_WARNED:
                continue
            _SYSTEM_REPLACEMENT_WARNED.add(key)
            MelleaLogger.get_logger().warning(
                "Intrinsic '%s' replaced a caller-supplied system prompt with "
                "the trained system prompt from its configuration. The "
                "caller's system prompt is ignored for this capability.",
                adapter_key,
            )

    def _transform(
        self, chat_completion: ChatCompletion, /, **kwargs
    ) -> ChatCompletion:
        edits: dict = {}

        if self._first_token:
            self._reject_decision_incompatibilities(chat_completion)

        if self.sentence_boundaries:
            chat_completion = self._mark_sentence_boundaries(chat_completion)

        if self.docs_as_message:
            # Note that it's important for this transformation to happen after sentence
            # boundaries are inserted into the documents.
            chat_completion = move_documents_to_message(  # type: ignore[assignment]
                chat_completion, self.docs_as_message
            )

        # The reserved `state` kwarg is consumed by `state_template` rendering
        # and never reaches the instruction template's format arguments.
        state = kwargs.pop("state", _MISSING_STATE)
        if state is not _MISSING_STATE and self.state_template is None:
            raise ValueError(
                "A 'state' argument was passed, but the intrinsic's "
                "configuration declares no 'state_template' to render it. This "
                "is a configuration error: add a 'state_template' field or "
                "stop passing 'state'."
            )

        if (
            self.instruction is not None
            or self.state_template is not None
            or self.system_prompt is not None
        ):
            # Generate and append new user message of instructions
            messages = chat_completion.messages.copy()  # Do not modify input!
            if self.system_prompt is not None:
                messages = self._replace_system_messages(messages, self.system_prompt)
            format_args = kwargs.copy()
            if len(messages) > 0:
                format_args["last_message"] = messages[-1].content
            if state is not _MISSING_STATE:
                try:
                    state_str = self.state_template.format(  # type: ignore[union-attr]
                        state=_render_state_value(state)
                    )
                except KeyError as e:
                    raise ValueError(
                        f"Missing argument for intrinsic's state template: {e}"
                    ) from e
                messages.append(UserMessage(content=state_str))
            if self.instruction is not None:
                try:
                    instruction_str = self.instruction.format(**format_args)
                except KeyError as e:
                    raise ValueError(
                        f"Missing argument for intrinsic's instruction string: {e}"
                    ) from e
                messages.append(UserMessage(content=instruction_str))
            edits["messages"] = messages
        edits.update(self.parameters)

        # Marker guard: the activation marker must occur exactly once across
        # the final messages. A count of zero means the adapter would never
        # activate; a count above one (usually state, question, or option
        # content containing the marker) means it would activate at the wrong
        # position.
        final_messages = edits.get("messages", chat_completion.messages)
        if self.state_template is not None and not final_messages:
            raise ValueError(
                "The intrinsic's rewriter produced no messages. A "
                "state_template capability requires at least the state and "
                "instruction turns."
            )
        if self.invocation_text is not None:
            marker_count = sum(
                m.content.count(self.invocation_text) for m in final_messages
            )
            if marker_count != 1:
                raise _InvocationMarkerError(
                    f"Invocation marker {self.invocation_text!r} appears "
                    f"{marker_count} times across the final messages, expected "
                    f"exactly 1. The adapter would not activate correctly. "
                    f"State, question, or option content containing the marker "
                    f"is the usual cause."
                )

        extra_body_dict = (
            chat_completion.extra_body.model_dump()
            if chat_completion.extra_body
            else {}
        )
        extra_body_dict.update(self.extra_body_parameters)
        if len(extra_body_dict) > 0:
            edits["extra_body"] = VLLMExtraBody.model_validate(extra_body_dict)

        return chat_completion.model_copy(update=edits)
