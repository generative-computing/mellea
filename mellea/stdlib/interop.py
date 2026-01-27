"""High-level API for validating outputs from external LLMs.

This module provides a simplified API for validating outputs from external LLM
frameworks (LangChain, OpenAI SDK, etc.) using Mellea's validation capabilities.

Example:
    >>> from mellea.stdlib.interop import external_validate, ExternalSession
    >>> from mellea.stdlib.requirements import req
    >>>
    >>> # Simple validation
    >>> results = external_validate(
    ...     output="The answer is 42",
    ...     requirements=["Must contain a number"],
    ...     backend=backend,
    ... )
    >>>
    >>> # Session-based validation
    >>> session = ExternalSession.from_output("The answer is 42", backend)
    >>> results = session.validate(["Must contain a number"])
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeAlias, TypeGuard

from ..core import Backend, CBlock, ModelOutputThunk, Requirement, ValidationResult
from ..helpers import _run_async_in_thread
from .components import Message
from .components.chat_converters import (
    langchain_message_to_mellea,
    openai_message_to_mellea,
)
from .context import ChatContext, SimpleContext
from .requirements import req

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


# Type alias for OpenAI message format
OpenAIMessage: TypeAlias = dict[str, Any]

# Type alias for supported context formats
ContextInput: TypeAlias = (
    "list[OpenAIMessage] | list[Message] | list[BaseMessage] | None"
)
OutputInput: TypeAlias = (
    "str | OpenAIMessage | BaseMessage | Message | ModelOutputThunk"
)


# =============================================================================
# Helper Functions
# =============================================================================


def _is_openai_message(msg: Any) -> TypeGuard[dict[str, Any]]:
    """Check if a message is in OpenAI format (dict with role and content)."""
    return isinstance(msg, dict) and "role" in msg


def _is_langchain_message(msg: Any) -> bool:
    """Check if a message is a LangChain BaseMessage."""
    # Check for BaseMessage attributes without importing langchain
    return (
        hasattr(msg, "content")
        and hasattr(msg, "type")
        and not isinstance(msg, dict | Message)
    )


def _convert_message_to_mellea(msg: Any) -> Message:
    """Convert a single message from any supported format to Mellea Message.

    Args:
        msg: A message in OpenAI dict format, LangChain BaseMessage, or Mellea Message.

    Returns:
        A Mellea Message object.

    Raises:
        ValueError: If the message format is not recognized.
    """
    if isinstance(msg, Message):
        return msg
    elif _is_openai_message(msg):
        return openai_message_to_mellea(msg)
    elif _is_langchain_message(msg):
        return langchain_message_to_mellea(msg)
    else:
        raise ValueError(
            f"Unsupported message format: {type(msg)}. "
            "Expected OpenAI dict, LangChain BaseMessage, or Mellea Message."
        )


def _build_context(context: ContextInput = None) -> ChatContext:
    """Build a ChatContext from various input formats.

    Auto-detects the context format (OpenAI, LangChain, or Mellea) and converts
    to a ChatContext.

    Args:
        context: Optional list of messages in any supported format.

    Returns:
        A ChatContext containing the converted messages.
    """
    ctx = ChatContext()

    if context is None or len(context) == 0:
        return ctx

    for msg in context:
        mellea_msg = _convert_message_to_mellea(msg)
        ctx = ctx.add(mellea_msg)

    return ctx


def _extract_output_string(output: OutputInput) -> str:
    """Extract a string from various output formats.

    Args:
        output: Output in string, OpenAI dict, LangChain message, Mellea Message,
            or ModelOutputThunk format.

    Returns:
        The extracted string content.

    Raises:
        ValueError: If the output format is not recognized or content is None.
    """
    if isinstance(output, str):
        return output
    elif isinstance(output, ModelOutputThunk):
        if output.value is None:
            raise ValueError("ModelOutputThunk has no value")
        return output.value
    elif isinstance(output, Message):
        return output.content
    elif _is_openai_message(output):
        content = output.get("content", "")
        if isinstance(content, str):
            return content
        # Handle multimodal content (list of blocks)
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            return "".join(text_parts)
        return str(content)
    elif _is_langchain_message(output):
        content = getattr(output, "content", "")
        if isinstance(content, str):
            return content
        # Handle multimodal content
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            return "".join(text_parts)
        return str(content)
    else:
        raise ValueError(
            f"Unsupported output format: {type(output)}. "
            "Expected str, OpenAI dict, LangChain message, Message, or ModelOutputThunk."
        )


def _normalize_requirements(requirements: list[str | Requirement]) -> list[Requirement]:
    """Convert a mixed list of strings and Requirements to all Requirements.

    Args:
        requirements: List of string descriptions or Requirement objects.

    Returns:
        List of Requirement objects.
    """
    return [req(r) if isinstance(r, str) else r for r in requirements]


def _default_repair_prompt(
    failed_requirements: list[tuple[Requirement, ValidationResult]], output: str
) -> str:
    """Generate a default repair prompt for IVR loops.

    Args:
        failed_requirements: List of (requirement, validation_result) tuples for
            requirements that failed.
        output: The original output that failed validation.

    Returns:
        A repair prompt string.
    """
    failed_desc = "\n".join(
        f"- {req.description}" + (f": {vr.reason}" if vr.reason else "")
        for req, vr in failed_requirements
    )
    return (
        f"Your previous response did not meet the following requirements:\n"
        f"{failed_desc}\n\n"
        f"Your previous response was:\n{output}\n\n"
        f"Please try again, ensuring all requirements are met."
    )


# =============================================================================
# Core Validation Functions
# =============================================================================


def external_validate(
    output: OutputInput,
    requirements: list[str | Requirement],
    backend: Backend,
    *,
    context: ContextInput = None,
    model_options: dict | None = None,
) -> list[ValidationResult]:
    """Validate an external LLM output against a list of requirements.

    This is a convenience function that handles the boilerplate of converting
    external formats to Mellea's internal types and running validation.

    Args:
        output: The LLM output to validate. Can be a string, OpenAI-format dict,
            LangChain message, Mellea Message, or ModelOutputThunk.
        requirements: List of requirements to validate against. Can be strings
            (for LLM-as-judge validation) or Requirement objects.
        backend: The Mellea backend to use for validation.
        context: Optional conversation context. Can be a list of messages in
            OpenAI dict format, LangChain messages, or Mellea Messages.
        model_options: Optional model configuration options.

    Returns:
        A list of ValidationResult objects, one per requirement.

    Example:
        >>> results = external_validate(
        ...     output="The capital of France is Paris.",
        ...     requirements=["Must mention a city", "Must be factually correct"],
        ...     backend=backend,
        ... )
        >>> all(r.as_bool() for r in results)
        True
    """
    return _run_async_in_thread(
        aexternal_validate(
            output=output,
            requirements=requirements,
            backend=backend,
            context=context,
            model_options=model_options,
        )
    )


async def aexternal_validate(
    output: OutputInput,
    requirements: list[str | Requirement],
    backend: Backend,
    *,
    context: ContextInput = None,
    model_options: dict | None = None,
) -> list[ValidationResult]:
    """Async version of external_validate.

    See external_validate for full documentation.
    """
    from . import functional as mfuncs

    # Convert context and output to Mellea types
    ctx = _build_context(context)
    output_str = _extract_output_string(output)
    reqs = _normalize_requirements(requirements)

    # Wrap output in a CBlock for validation
    output_block = CBlock(output_str)

    # Run validation
    return await mfuncs.avalidate(
        reqs=reqs,
        context=ctx,
        backend=backend,
        output=output_block,
        model_options=model_options,
    )


# =============================================================================
# Session-Based API
# =============================================================================


class ExternalSession:
    """A session for validating external LLM outputs.

    ExternalSession provides a convenient way to validate outputs from external
    LLMs with state management. It wraps a ChatContext and provides factory
    methods for common use cases.

    Example:
        >>> session = ExternalSession.from_langchain(lc_messages, backend)
        >>> results = session.validate(["Must be helpful"])
        >>> if session.all_passed(["Must be helpful"]):
        ...     print("Validation passed!")
    """

    def __init__(
        self, context: ChatContext, backend: Backend, output: str | None = None
    ):
        """Initialize an ExternalSession.

        Args:
            context: The ChatContext containing conversation history.
            backend: The Mellea backend to use for validation.
            output: Optional output string to validate (if not in context).
        """
        self.context = context
        self.backend = backend
        self._output = output

    @classmethod
    def from_output(
        cls, output: OutputInput, backend: Backend, *, context: ContextInput = None
    ) -> ExternalSession:
        """Create a session from an output and optional context.

        Args:
            output: The LLM output to validate.
            backend: The Mellea backend to use.
            context: Optional conversation context.

        Returns:
            An ExternalSession configured with the output and context.
        """
        ctx = _build_context(context)
        output_str = _extract_output_string(output)

        # Add output as assistant message to context
        assistant_msg = Message(role="assistant", content=output_str)
        ctx = ctx.add(assistant_msg)

        return cls(context=ctx, backend=backend, output=output_str)

    @classmethod
    def from_openai(
        cls, messages: list[OpenAIMessage], backend: Backend
    ) -> ExternalSession:
        """Create a session from OpenAI-format messages.

        The last assistant message is treated as the output to validate.

        Args:
            messages: List of messages in OpenAI dict format.
            backend: The Mellea backend to use.

        Returns:
            An ExternalSession configured with the messages.
        """
        if not messages:
            return cls(context=ChatContext(), backend=backend)

        # Find the last assistant message as output
        output_str: str | None = None
        context_messages = messages

        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                output_str = _extract_output_string(messages[i])
                break

        ctx = _build_context(context_messages)
        return cls(context=ctx, backend=backend, output=output_str)

    @classmethod
    def from_langchain(
        cls,
        messages: list[Any],  # list[BaseMessage]
        backend: Backend,
    ) -> ExternalSession:
        """Create a session from LangChain messages.

        The last AI message is treated as the output to validate.

        Args:
            messages: List of LangChain BaseMessage objects.
            backend: The Mellea backend to use.

        Returns:
            An ExternalSession configured with the messages.
        """
        if not messages:
            return cls(context=ChatContext(), backend=backend)

        # Find the last assistant/AI message as output
        output_str: str | None = None

        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            msg_type = getattr(msg, "type", "")
            if msg_type in ("ai", "assistant"):
                output_str = _extract_output_string(msg)
                break

        ctx = _build_context(messages)
        return cls(context=ctx, backend=backend, output=output_str)

    @property
    def output(self) -> str | None:
        """Get the current output string."""
        return self._output

    def validate(
        self,
        requirements: list[str | Requirement],
        *,
        model_options: dict | None = None,
    ) -> list[ValidationResult]:
        """Validate the session's output against requirements.

        Args:
            requirements: List of requirements to validate against.
            model_options: Optional model configuration options.

        Returns:
            A list of ValidationResult objects.

        Raises:
            ValueError: If no output has been set.
        """
        if self._output is None:
            raise ValueError(
                "No output to validate. Use from_output() or set an output."
            )

        return external_validate(
            output=self._output,
            requirements=requirements,
            backend=self.backend,
            model_options=model_options,
        )

    async def avalidate(
        self,
        requirements: list[str | Requirement],
        *,
        model_options: dict | None = None,
    ) -> list[ValidationResult]:
        """Async version of validate."""
        if self._output is None:
            raise ValueError(
                "No output to validate. Use from_output() or set an output."
            )

        return await aexternal_validate(
            output=self._output,
            requirements=requirements,
            backend=self.backend,
            model_options=model_options,
        )

    def all_passed(
        self,
        requirements: list[str | Requirement],
        *,
        model_options: dict | None = None,
    ) -> bool:
        """Check if all requirements pass validation.

        Args:
            requirements: List of requirements to validate against.
            model_options: Optional model configuration options.

        Returns:
            True if all requirements pass, False otherwise.
        """
        results = self.validate(requirements, model_options=model_options)
        return all(r.as_bool() for r in results)

    async def aall_passed(
        self,
        requirements: list[str | Requirement],
        *,
        model_options: dict | None = None,
    ) -> bool:
        """Async version of all_passed."""
        results = await self.avalidate(requirements, model_options=model_options)
        return all(r.as_bool() for r in results)


# =============================================================================
# IVR (Instruct-Validate-Repair) Helper
# =============================================================================


@dataclass
class IVRResult:
    """Result from an external IVR loop.

    Attributes:
        success: Whether validation passed.
        output: The final output (whether successful or not).
        attempts: Number of generation attempts made.
        validation_results: Results from the final validation.
        all_outputs: List of all outputs generated during the loop.
    """

    success: bool
    output: str
    attempts: int
    validation_results: list[ValidationResult]
    all_outputs: list[str] = field(default_factory=list)


def external_ivr(
    generate_fn: Callable[[list[Message]], str],
    requirements: list[str | Requirement],
    backend: Backend,
    *,
    initial_context: ContextInput = None,
    loop_budget: int = 3,
    repair_prompt_fn: Callable[[list[tuple[Requirement, ValidationResult]], str], str]
    | None = None,
) -> IVRResult:
    """Run an Instruct-Validate-Repair loop with an external generate function.

    This function implements the IVR pattern for external LLMs, allowing you to
    use Mellea's validation with any LLM provider.

    Args:
        generate_fn: A function that takes a list of Mellea Messages and returns
            a string output. This is called to generate (and regenerate) outputs.
        requirements: List of requirements to validate against.
        backend: The Mellea backend to use for validation.
        initial_context: Optional initial conversation context.
        loop_budget: Maximum number of generation attempts (default: 3).
        repair_prompt_fn: Optional custom function to generate repair prompts.
            If not provided, uses _default_repair_prompt.

    Returns:
        An IVRResult containing the final output and validation status.

    Example:
        >>> def generate(context):
        ...     # Use your LLM here
        ...     return langchain_agent.invoke({"messages": context})
        ...
        >>> result = external_ivr(
        ...     generate_fn=generate,
        ...     requirements=["Must be polite", "Must answer the question"],
        ...     backend=backend,
        ...     loop_budget=5,
        ... )
        >>> if result.success:
        ...     print(f"Success after {result.attempts} attempts")
    """
    return _run_async_in_thread(
        aexternal_ivr(
            generate_fn=generate_fn,
            requirements=requirements,
            backend=backend,
            initial_context=initial_context,
            loop_budget=loop_budget,
            repair_prompt_fn=repair_prompt_fn,
        )
    )


async def aexternal_ivr(
    generate_fn: Callable[[list[Message]], str],
    requirements: list[str | Requirement],
    backend: Backend,
    *,
    initial_context: ContextInput = None,
    loop_budget: int = 3,
    repair_prompt_fn: Callable[[list[tuple[Requirement, ValidationResult]], str], str]
    | None = None,
) -> IVRResult:
    """Async version of external_ivr.

    See external_ivr for full documentation.
    """
    from . import functional as mfuncs

    if repair_prompt_fn is None:
        repair_prompt_fn = _default_repair_prompt

    # Normalize requirements
    reqs = _normalize_requirements(requirements)

    # Build initial context
    ctx = _build_context(initial_context)
    context_messages = ctx.as_list()
    mellea_messages = [m for m in context_messages if isinstance(m, Message)]

    all_outputs: list[str] = []

    for attempt in range(1, loop_budget + 1):
        # Generate output
        output_str = generate_fn(mellea_messages)
        all_outputs.append(output_str)

        # Add output to context for validation
        output_block = CBlock(output_str)

        # Validate
        validation_results = await mfuncs.avalidate(
            reqs=reqs, context=ctx, backend=backend, output=output_block
        )

        # Check if all passed
        if all(r.as_bool() for r in validation_results):
            return IVRResult(
                success=True,
                output=output_str,
                attempts=attempt,
                validation_results=validation_results,
                all_outputs=all_outputs,
            )

        # If not last attempt, add repair message
        if attempt < loop_budget:
            failed_reqs = [
                (req, vr)
                for req, vr in zip(reqs, validation_results)
                if not vr.as_bool()
            ]
            repair_prompt = repair_prompt_fn(failed_reqs, output_str)

            # Add assistant response and repair prompt to context
            assistant_msg = Message(role="assistant", content=output_str)
            repair_msg = Message(role="user", content=repair_prompt)

            ctx = ctx.add(assistant_msg)
            ctx = ctx.add(repair_msg)

            # Update mellea_messages for next iteration
            context_messages = ctx.as_list()
            mellea_messages = [m for m in context_messages if isinstance(m, Message)]

    # Return last attempt results (failed)
    return IVRResult(
        success=False,
        output=all_outputs[-1] if all_outputs else "",
        attempts=loop_budget,
        validation_results=validation_results,
        all_outputs=all_outputs,
    )
