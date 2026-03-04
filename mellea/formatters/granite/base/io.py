# SPDX-License-Identifier: Apache-2.0

"""Input and output processing for chat completions-like APIs.

Classes and functions that implement common aspects of input and output string
processing for chat completions-like APIs.
"""

# Standard
import abc

# Third Party
import pydantic

# Local
from .types import AssistantMessage, ChatCompletion, ChatCompletionResponse, Document


class InputProcessor(abc.ABC):
    """Interface for generic input processors.

    An input processor exposes an API to transform a chat completion request into a string prompt.
    """

    @abc.abstractmethod
    def transform(
        self, chat_completion: ChatCompletion, add_generation_prompt: bool = True
    ) -> str:
        """Convert the structured representation of the inputs to a completion request.

        Converts the structured representation of the inputs to a completion request into
        the string representation of the tokens that should be sent to the model to
        implement said request.

        :param chat_completion: Structured representation of the inputs
        :param add_generation_prompt: If ``True``, the returned prompt string will
            contain a prefix of the next assistant response for use as a prompt to a
            generation request. Otherwise, the prompt will only contain the messages and
            documents in ``input``.

        :returns: String that can be passed to the model's tokenizer to create a prompt
            for generation.
        """


class OutputProcessor(abc.ABC):
    """Base class for generic output processors.

    An output processor exposes an API to transform model output into a structured
    representation of the information.

    This interface is generic; see individual classes for more specific arguments.
    """

    @abc.abstractmethod
    def transform(
        self, model_output: str, chat_completion: ChatCompletion | None = None
    ) -> AssistantMessage:
        """Convert the model output into a structured representation.

        Convert the model output generated into a structured representation of the
        information.

        :param model_output: String output of the a generation request, potentially
            incomplete if it was a streaming request
        :param chat_completion: The chat completion request that produced
            ``model_output``. Parameters of the request can determine how the output
            should be decoded.

        :returns: The parsed output so far, as an instance of :class:`AssistantMessage`
            possibly with model-specific extension fields.
        """


class ChatCompletionRewriter(abc.ABC):
    """Base class for objects that rewrite a chat completion request.

    Base class for objects that rewrite a chat completion request into another chat
    completion request.
    """

    def transform(
        self, chat_completion: ChatCompletion | str | dict, /, **kwargs
    ) -> ChatCompletion:
        """Rewrite a chat completion request into another one.

        Rewrite a chat completion request into another chat completion request.
        Does not modify the original :class:`ChatCompletion` object.

        :param chat_completion: Original chat completion request, either as a dataclass
            or as the JSON representation of one.

        :returns: Rewritten copy of ``chat_completion``.
        """
        if isinstance(chat_completion, str):
            chat_completion = ChatCompletion.model_validate_json(chat_completion)
        if isinstance(chat_completion, dict):
            chat_completion = ChatCompletion.model_validate(chat_completion)

        if not isinstance(chat_completion, ChatCompletion):
            raise TypeError(
                f"chat_completion argument must be either a ChatCompletion "
                f"object or the JSON representation of one. Received type "
                f"'{type(chat_completion)}'."
            )
        return self._transform(chat_completion, **kwargs)

    @abc.abstractmethod
    def _transform(
        self, chat_completion: ChatCompletion, /, **kwargs
    ) -> ChatCompletion:
        """Subclasses must implement this internal hook to transform input requests.

        :param chat_completion: Description Original chat completion request
        :type chat_completion: ChatCompletion
        :param kwargs: Description
        :return: Description
        :rtype: ChatCompletion
        """


class ChatCompletionResultProcessor(abc.ABC):
    """Base class for chat completion result processors.

    Base class for objects that convert the raw json result of a chat completion request
    into a JSON object with model-specific postprocessing applied.
    """

    def transform(
        self,
        chat_completion_response: ChatCompletionResponse | dict | pydantic.BaseModel,
        chat_completion: ChatCompletion | None = None,
    ) -> ChatCompletionResponse:
        """Parse the result of a chat completion.

        :param chat_completion_response: Response to a chat completion request as
            a parsed dataclass, parsed JSON value, or OpenAI dataclass
        :param chat_completion: The chat completion request that produced
            ``chat_completion_response``. Required by some implementations in order
            to decode references to part of the original request.

        :returns: Rewritten copy of ``chat_completion``.
        """
        # Convert from over-the-wire format if necessary
        if isinstance(chat_completion_response, dict):
            chat_completion_response = ChatCompletionResponse.model_validate(
                chat_completion_response
            )
        elif not isinstance(chat_completion_response, ChatCompletionResponse):
            if isinstance(chat_completion_response, pydantic.BaseModel):
                # Got another library's dataclass. Attempt to convert to our own.
                chat_completion_response = ChatCompletionResponse.model_validate(
                    chat_completion_response.model_dump()
                )
            else:
                raise TypeError(
                    f"Received unexpected type {type(chat_completion_response)=}"
                )
        return self._transform_impl(chat_completion_response, chat_completion)

    @abc.abstractmethod
    def _transform_impl(
        self,
        chat_completion_response: ChatCompletionResponse,
        chat_completion: ChatCompletion | None,
    ) -> ChatCompletionResponse:
        """Transform the completion response.

        Subclasses must override this method with an implementation of
        :func:`transform()`.
        """


class Retriever(abc.ABC):
    """Base class for document retrievers.

    Provides APIs for searching by text snippet and for inserting new documents.
    """

    @abc.abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> list[Document]:
        """Retrieve the top k matches of a query from the corpus.

        :param query: Query string to use for lookup
        :param top_k: Number of top-k results to return

        :returns: Pyarrow table with fields: "id", "title", "url", "begin", "end",
            "text", "score"
        """
