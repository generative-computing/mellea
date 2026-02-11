"""LangChain-compatible chat model that wraps Mellea."""

from collections.abc import AsyncIterator, Iterator
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from pydantic import Field

from .message_conversion import langchain_to_mellea_messages, mellea_to_langchain_result

try:
    from mellea import MelleaSession
    from mellea.backends import ModelOption
    from mellea.core import Context
    from mellea.stdlib.requirements import check, req
    from mellea.stdlib.sampling import RejectionSamplingStrategy
except ImportError:
    # Fallback for type hints if mellea is not installed
    MelleaSession = Any  # type: ignore
    ModelOption = Any  # type: ignore
    Context = Any  # type: ignore
    req = None  # type: ignore
    check = None  # type: ignore
    RejectionSamplingStrategy = None  # type: ignore


class MelleaChatModel(BaseChatModel):
    """LangChain chat model that uses Mellea as the backend.

    This allows LangChain applications to use Mellea's generative
    programming capabilities through the standard LangChain interface.

    Example:
        ```python
        from mellea import start_session
        from mellea_langchain import MelleaChatModel

        # Create Mellea session
        m = start_session()

        # Create LangChain chat model
        chat_model = MelleaChatModel(mellea_session=m)

        # Use with LangChain
        from langchain_core.messages import HumanMessage
        response = chat_model.invoke([HumanMessage(content="Hello!")])
        ```
    """

    mellea_session: Any = Field(description="The Mellea session to use for generation")
    model_name: str = Field(default="mellea", description="Name to identify this model")
    streaming: bool = Field(default=False, description="Whether to stream responses by default")

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True

    def __init__(
        self,
        mellea_session: Any,
        model_name: str = "mellea",
        streaming: bool = False,
        **kwargs: Any,
    ):
        """Initialize the Mellea chat model.

        Args:
            mellea_session: Configured Mellea session
            model_name: Name to identify this model
            streaming: Whether to stream by default
            **kwargs: Additional LangChain model parameters
        """
        super().__init__(
            mellea_session=mellea_session,
            model_name=model_name,
            streaming=streaming,
            **kwargs,
        )

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "MelleaChatModel":
        """Bind tools to this chat model.

        This method creates a new instance of the chat model with tools attached.
        The tools will be passed to Mellea during generation, enabling the model
        to call functions/tools as needed.

        Args:
            tools: List of LangChain tools (BaseTool) or Mellea tools (MelleaTool).
                   LangChain tools will be automatically converted to Mellea format.
            **kwargs: Additional binding parameters (e.g., tool_choice)

        Returns:
            New MelleaChatModel instance with tools bound

        Example:
            ```python
            from langchain_core.tools import tool

            @tool
            def web_search(query: str) -> str:
                '''Search the web.'''
                return f"Results for: {query}"

            # Bind tools to the model
            model_with_tools = chat_model.bind_tools([web_search])

            # Use with LangChain agents
            response = model_with_tools.invoke([
                HumanMessage(content="Search for Python tutorials")
            ])
            ```
        """
        # Create a new instance with the same configuration
        bound_model = self.__class__(
            mellea_session=self.mellea_session,
            model_name=self.model_name,
            streaming=self.streaming,
        )

        # Store tools and tool choice for later use
        bound_model._bound_tools = tools  # type: ignore
        bound_model._tool_choice = kwargs.get("tool_choice")  # type: ignore

        return bound_model

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "mellea"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous generation.

        Args:
            messages: List of messages to generate from
            stop: Stop sequences (not currently supported)
            run_manager: Callback manager for this run
            **kwargs: Additional generation parameters

        Returns:
            ChatResult with generated message
        """
        # Convert LangChain messages to Mellea format
        mellea_messages = langchain_to_mellea_messages(messages)

        # Extract the last message content for generation
        if not mellea_messages:
            raise ValueError("No messages provided for generation")

        last_message_content = mellea_messages[-1].content

        # Extract model options and special parameters from kwargs
        model_options = kwargs.get("model_options", {}).copy()

        # Extract requirements/strategy parameters (can be passed directly or in model_options)
        requirements = kwargs.get("requirements") or model_options.pop("requirements", None)
        strategy = kwargs.get("strategy") or model_options.pop("strategy", None)
        return_sampling_results = kwargs.get("return_sampling_results", False) or model_options.pop(
            "return_sampling_results", False
        )

        # Add tools to model_options if bound via bind_tools()
        tool_calls_enabled = False
        if hasattr(self, "_bound_tools") and self._bound_tools:
            from mellea.backends import ModelOption

            from .tool_conversion import langchain_to_mellea_tools

            mellea_tools = langchain_to_mellea_tools(self._bound_tools)
            model_options[ModelOption.TOOLS] = mellea_tools
            tool_calls_enabled = True

        # Use instruct method when requirements or strategy are provided for validation
        if requirements is not None or strategy is not None:
            # Use synchronous instruct method for requirements and strategy support
            response = self.mellea_session.instruct(
                last_message_content,
                requirements=requirements,
                strategy=strategy,
                model_options=model_options,
                return_sampling_results=return_sampling_results,
            )

            # Handle sampling results if requested
            if return_sampling_results:
                # response is a SamplingResult object
                if response.success:
                    # Use the successful result
                    result = mellea_to_langchain_result(response.result)
                else:
                    # Use the first sample if validation failed
                    if response.sample_generations:
                        result = mellea_to_langchain_result(response.sample_generations[0])
                    else:
                        raise ValueError("No samples generated during validation")
            else:
                # Standard response
                result = mellea_to_langchain_result(response)
        else:
            # Use standard synchronous chat method
            response = self.mellea_session.chat(
                last_message_content,
                model_options=model_options,
                tool_calls=tool_calls_enabled,  # Enable tool calling if tools are bound
            )
            result = mellea_to_langchain_result(response)

        # Invoke callbacks
        if run_manager:
            run_manager.on_llm_end(result)

        return result

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Asynchronous generation using Mellea.

        Supports both standard chat and validated generation with requirements/strategies.
        When requirements or strategy are provided, uses Mellea's ainstruct method for
        validation and retry logic. Otherwise, uses standard achat method.

        Args:
            messages: List of messages to generate from
            stop: Stop sequences (not currently supported)
            run_manager: Async callback manager for this run
            **kwargs: Additional generation parameters including:
                - requirements: List of requirement strings or req()/check() objects
                - strategy: Sampling strategy (e.g., RejectionSamplingStrategy)
                - return_sampling_results: Boolean to get detailed validation info
                - model_options: Additional Mellea model options

        Returns:
            ChatResult with generated message
        """
        # Convert LangChain messages to Mellea format
        mellea_messages = langchain_to_mellea_messages(messages)

        # Extract the last message content for generation
        if not mellea_messages:
            raise ValueError("No messages provided for generation")

        last_message_content = mellea_messages[-1].content

        # Extract model options and special parameters from kwargs
        model_options = kwargs.get("model_options", {}).copy()

        # Extract requirements/strategy parameters (can be passed directly or in model_options)
        requirements = kwargs.get("requirements") or model_options.pop("requirements", None)
        strategy = kwargs.get("strategy") or model_options.pop("strategy", None)
        return_sampling_results = kwargs.get("return_sampling_results", False) or model_options.pop(
            "return_sampling_results", False
        )

        # Add tools to model_options if bound via bind_tools()
        tool_calls_enabled = False
        if hasattr(self, "_bound_tools") and self._bound_tools:
            from mellea.backends import ModelOption

            from .tool_conversion import langchain_to_mellea_tools

            mellea_tools = langchain_to_mellea_tools(self._bound_tools)
            model_options[ModelOption.TOOLS] = mellea_tools
            tool_calls_enabled = True

        # Use ainstruct method when requirements or strategy are provided for validation
        if requirements is not None or strategy is not None:
            # Use instruct method for requirements and strategy support
            response = await self.mellea_session.ainstruct(
                last_message_content,
                requirements=requirements,
                strategy=strategy,
                model_options=model_options,
                return_sampling_results=return_sampling_results,
            )

            # Handle sampling results if requested
            if return_sampling_results:
                # response is a SamplingResult object
                if response.success:
                    # Use the successful result
                    result = mellea_to_langchain_result(response.result)
                else:
                    # Use the first sample if validation failed
                    if response.sample_generations:
                        result = mellea_to_langchain_result(response.sample_generations[0])
                    else:
                        raise ValueError("No samples generated during validation")
            else:
                # Standard response
                result = mellea_to_langchain_result(response)
        else:
            # Use standard chat method
            response = await self.mellea_session.achat(
                last_message_content,
                model_options=model_options,
                tool_calls=tool_calls_enabled,  # Enable tool calling if tools are bound
            )
            result = mellea_to_langchain_result(response)

        # Invoke callbacks
        if run_manager:
            await run_manager.on_llm_end(result)

        return result

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Synchronous streaming.

        Args:
            messages: List of messages to generate from
            stop: Stop sequences
            run_manager: Callback manager
            **kwargs: Additional parameters

        Yields:
            ChatGenerationChunk objects

        Note:
            Mellea's achat doesn't support streaming, so this returns
            the full response as a single chunk.
        """
        # Generate full response
        result = self._generate(messages, stop, run_manager, **kwargs)

        # Return as single chunk
        content = result.generations[0].message.content
        chunk = ChatGenerationChunk(message=AIMessageChunk(content=content))

        if run_manager:
            run_manager.on_llm_new_token(content)

        yield chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Asynchronous streaming.

        Args:
            messages: List of messages to generate from
            stop: Stop sequences
            run_manager: Async callback manager
            **kwargs: Additional parameters

        Yields:
            ChatGenerationChunk objects

        Note:
            Mellea's achat doesn't support streaming, so this returns
            the full response as a single chunk.
        """
        # Generate full response
        result = await self._agenerate(messages, stop, run_manager, **kwargs)

        # Return as single chunk
        content = result.generations[0].message.content
        chunk = ChatGenerationChunk(message=AIMessageChunk(content=content))

        if run_manager:
            await run_manager.on_llm_new_token(content)

        yield chunk
