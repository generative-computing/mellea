# SPDX-License-Identifier: Apache-2.0

"""Classes and functions that implement common aspects of input processing for intrinsics."""

# Standard
import json
import pathlib

# First Party
from ..base.io import ChatCompletionRewriter
from ..base.types import ChatCompletion, DocumentMessage, UserMessage, VLLMExtraBody
from ..base.util import import_optional

# Local
from .constants import TOP_LOGPROBS
from .util import make_config_dict


def _needs_logprobs(transformations: list | None) -> bool:
    """Check whether input processing for a model needs to specify logprobs.

    Subroutine to check whether input processing for a model needs to specify logprobs
    in the chat completion arguments.

    :param transformations: Contents of the field by the same name in the YAML file
    :type transformations: list
    :return: ``True`` if this intrinsic produces a field for which logprobs need to be
        enabled for downstream result decoding to succeed.
    :rtype: bool
    """
    if transformations is None:
        return False
    return any(t["type"] == "likelihood" for t in transformations)


def sentence_delimiter(tag, sentence_num) -> str:
    """Return a tag string that identifies the beginning of the indicated sentence.

    :param tag: tag string, i.e. "i" or "c"
    :param sentence_num: index of sentence
    :return: Tag string (including trailing space) that identifies the beginning of
        the indicated sentence in sentence-tagged text.
    :rtype: str
    """
    return f"<{tag}{sentence_num}> "


def mark_sentence_boundaries(
    split_strings: list[list[str]], tag_prefix: str
) -> list[str]:
    """Modify input strings by inserting sentence boundary markers.

    Modify one or more input strings by inserting a tag in the form
    ``<[prefix][number]>``
    at the location of each sentence boundary.

    :param split_strings: Input string(s), pre-split into sentences
    :param tag_prefix: String to place before the number part of each tagged
        sentence boundary.

    :returns: List of input strings with all sentence boundaries marked.
    """
    index = 0
    result: list[str] = []
    for sentences in split_strings:
        to_concat = []
        for sentence in sentences:
            to_concat.append(f"{sentence_delimiter(tag_prefix, index)}{sentence}")
            index += 1
        result.append(" ".join(to_concat))
    return result


def move_documents_to_message(
    chat_completion: ChatCompletion | dict, how: str = "string"
) -> ChatCompletion | dict:
    """Move RAG documents from extra_body to first message.

    By convention, our canned JSON requests place RAG documents in extra_body/documents.
    Some models do not accept this parameter.
    This function edits a request by putting the documents into the first turn of the
    messages.

    :param chat_completion: A chat completion request as dataclass or parsed JSON
    :param how: How to serialize the documents; supported values are "string" and "json"

    :returns: A copy of ``chat_completion`` with any documents under ``extra_body``
        moved to the first message. Returned type will be the same as the input type.
        May return original object if no edits are necessary.
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
    """

    config: dict
    """Parsed YAML configuration file for the target intrinsic."""

    response_format: dict
    """JSON Schema of expected response format"""

    parameters: dict
    """Additional parameters (key-value pairs) that this rewriter adds to all chat
    completion requests."""

    extra_body_parameters: dict
    """Extended vLLM-specific parameters that go under the ``extra_body`` element of
    the parameters field. These parameters need to be merged with any ``extra_body``
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
    Optional specification for moving documents from ``extra_body/documents`` to a
    user message at the beginning of the messages list. Value specifies how to serialize
    the documents into the message: "string" or "json".
    """

    def __init__(
        self,
        /,
        config_file: str | pathlib.Path | None = None,
        config_dict: dict | None = None,
        model_name: str | None = None,
    ):
        """Initialize the IntrinsicsRewriter.

        :param config_file: (optional) Location of YAML configuration file
        :param config_dict: (optional) Contents of YAML configuration file that the
            caller has parsed by calling ``yaml.safe_load()``.
        :param model_name: (optional) model name to put into chat completion requests,
         or ``None`` to use default model name from the configuration
        """
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
            self.parameters["top_logprobs"] = TOP_LOGPROBS
        self.instruction = self.config["instruction"]

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
                if k not in ("last_message", "documents"):
                    raise ValueError(
                        f"Unexpected location '{k}' in 'sentence_boundaries' field. "
                        f"Value should be 'last_message' or 'documents'."
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

        :param chat_completion: Argument to :func:`_transform()`
        :type chat_completion: ChatCompletion
        :return: Copy of original chat completion with sentence boundaries marked in
            the last message and in documents.
        :rtype: ChatCompletion
        """
        # Mark sentence boundaries in the last message.
        if self.sentence_boundaries and "last_message" in self.sentence_boundaries:
            messages = chat_completion.messages.copy()  # Do not modify input!
            last_message_as_sentences = list(
                self.sentence_splitter.tokenize(messages[-1].content)
            )
            last_message_tag = self.sentence_boundaries["last_message"]
            if last_message_tag:
                rewritten_last_message_text = mark_sentence_boundaries(
                    [last_message_as_sentences], last_message_tag
                )[0]
                messages[-1].content = rewritten_last_message_text
                chat_completion = chat_completion.model_copy(
                    update={"messages": messages}
                )

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
                rewritten_docs = [
                    doc.model_copy(update={"text": text})
                    for doc, text in zip(
                        chat_completion.extra_body.documents,
                        mark_sentence_boundaries(docs_as_sentences, documents_tag),
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
        return chat_completion

    def _transform(
        self, chat_completion: ChatCompletion, /, **kwargs
    ) -> ChatCompletion:
        edits: dict = {}

        if self.sentence_boundaries:
            chat_completion = self._mark_sentence_boundaries(chat_completion)

        if self.docs_as_message:
            # Note that it's important for this transformation to happen after sentence
            # boundaries are inserted into the documents.
            chat_completion = move_documents_to_message(  # type: ignore[assignment]
                chat_completion, self.docs_as_message
            )

        if self.instruction is not None:
            # Generate and append new user message of instructions
            messages = chat_completion.messages.copy()  # Do not modify input!
            format_args = kwargs.copy()
            if len(messages) > 0:
                format_args["last_message"] = messages[-1].content
            try:
                instruction_str = self.instruction.format(**format_args)
            except KeyError as e:
                raise ValueError(
                    f"Missing argument for intrinsic's instruction string: {e}"
                ) from e
            messages.append(UserMessage(content=instruction_str))
            edits["messages"] = messages
        edits.update(self.parameters)

        extra_body_dict = (
            chat_completion.extra_body.model_dump()
            if chat_completion.extra_body
            else {}
        )
        extra_body_dict.update(self.extra_body_parameters)
        if len(extra_body_dict) > 0:
            edits["extra_body"] = VLLMExtraBody.model_validate(extra_body_dict)

        return chat_completion.model_copy(update=edits)
