"""Requirements for RAG (Retrieval-Augmented Generation) workflows."""

from ...backends.adapters import AdapterMixin
from ...core import Backend, Context, Requirement, ValidationResult
from ..components import Message
from ..components.intrinsic import rag


class HallucinationRequirement(Requirement):
    """Requirement that validates RAG responses for hallucinated content.

    Uses the hallucination_detection intrinsic to check if sentences in an
    assistant's response are faithful to the retrieved documents. Sentences
    with faithfulness_likelihood below the threshold are flagged as potential
    hallucinations.

    This requirement is designed for RAG workflows where you have:
    - A user question in the context
    - Retrieved documents
    - An assistant response to validate

    Example:
        ```python
        from mellea.stdlib.requirements.rag import HallucinationRequirement

        req = HallucinationRequirement(
            threshold=0.5,  # Flag sentences with faithfulness < 0.5
            max_hallucinated_ratio=0.2  # Fail if >20% of response is hallucinated
        )
        ```
    """

    def __init__(
        self,
        threshold: float = 0.5,
        max_hallucinated_ratio: float = 0.0,
        description: str | None = None,
    ):
        """Initialize hallucination detection requirement.

        Args:
            threshold: Faithfulness likelihood threshold (0.0-1.0). Sentences
                with faithfulness_likelihood below this value are considered
                hallucinated. Default: 0.5
            max_hallucinated_ratio: Maximum allowed ratio of hallucinated
                content (0.0-1.0). If the ratio of hallucinated characters
                to total response length exceeds this, validation fails.
                Default: 0.0 (any hallucination fails validation)
            description: Custom description for the requirement. If None,
                generates a description based on threshold and ratio.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be between 0.0 and 1.0, got {threshold}")
        if not 0.0 <= max_hallucinated_ratio <= 1.0:
            raise ValueError(
                f"max_hallucinated_ratio must be between 0.0 and 1.0, got {max_hallucinated_ratio}"
            )

        self.threshold = threshold
        self.max_hallucinated_ratio = max_hallucinated_ratio

        # Generate description if not provided
        if description is None:
            if max_hallucinated_ratio == 0.0:
                description = (
                    f"Response must be faithful to the provided documents "
                    f"(faithfulness threshold: {threshold})"
                )
            else:
                description = (
                    f"Response must be mostly faithful to the provided documents "
                    f"(faithfulness threshold: {threshold}, "
                    f"max hallucinated ratio: {max_hallucinated_ratio})"
                )

        # Initialize parent without validation function - we override validate() instead
        super().__init__(description=description, validation_fn=None)

    async def validate(
        self,
        backend: Backend,
        ctx: Context,
        *,
        format: type | None = None,
        model_options: dict | None = None,
    ) -> ValidationResult:
        """Validate hallucination in the context using the backend.

        Args:
            backend: Backend to use for hallucination detection
            ctx: Context containing the conversation history
            format: Unused for this requirement
            model_options: Unused for this requirement

        Returns:
            ValidationResult with pass/fail status, reason, and score
        """
        # Extract last message (should be assistant response)
        messages = ctx.as_list()
        if not messages:
            return ValidationResult(
                False, reason="Context is empty, cannot validate hallucination"
            )

        last_message = messages[-1]
        if not isinstance(last_message, Message):
            return ValidationResult(
                False,
                reason="Last context item is not a Message, cannot validate hallucination",
            )

        if last_message.role != "assistant":
            return ValidationResult(
                False,
                reason=f"Last message must be assistant response, got role: {last_message.role}",
            )

        response = last_message.content
        # Access private _docs attribute since documents property returns formatted strings
        documents = last_message._docs or []

        if not documents:
            return ValidationResult(
                False,
                reason="No documents provided in assistant message for hallucination validation",
            )

        # Backend is passed as parameter to validate(), not from context

        if not isinstance(backend, AdapterMixin):
            return ValidationResult(
                False,
                reason=f"Backend {backend.__class__.__name__} does not support adapters required for hallucination detection",
            )

        # Create context before the response by getting all but the last message
        all_messages = ctx.as_list()
        if len(all_messages) > 1:
            # Rebuild context without last message
            from ..context import ChatContext

            context_before_response = ChatContext()
            for msg in all_messages[:-1]:
                context_before_response = context_before_response.add(msg)
        else:
            # If only one message, use empty context
            from ..context import ChatContext

            context_before_response = ChatContext()

        # Call hallucination detection intrinsic
        try:
            results: list[dict] = rag.flag_hallucinated_content(
                response, documents, context_before_response, backend
            )
        except Exception as e:
            return ValidationResult(
                False, reason=f"Hallucination detection intrinsic failed: {e!s}"
            )

        # Apply threshold to identify hallucinated segments
        hallucinated_segments: list[dict] = [
            r for r in results if r["faithfulness_likelihood"] < self.threshold
        ]

        # If no hallucinations detected, pass validation
        if not hallucinated_segments:
            return ValidationResult(
                True, reason="No hallucinated content detected", score=1.0
            )

        # Calculate hallucination ratio
        total_chars = len(response)
        hallucinated_chars = sum(
            r["response_end"] - r["response_begin"] for r in hallucinated_segments
        )
        hallucination_ratio = (
            hallucinated_chars / total_chars if total_chars > 0 else 0.0
        )

        # Check against max_hallucinated_ratio
        passed = hallucination_ratio <= self.max_hallucinated_ratio

        # Build detailed reason
        reason = self._build_reason(hallucinated_segments, hallucination_ratio, passed)

        return ValidationResult(passed, reason=reason, score=1.0 - hallucination_ratio)

    def _build_reason(
        self,
        hallucinated_segments: list[dict],
        hallucination_ratio: float,
        passed: bool,
    ) -> str:
        """Build a detailed reason string for the validation result.

        Args:
            hallucinated_segments: List of hallucinated segment records
            hallucination_ratio: Ratio of hallucinated content
            passed: Whether validation passed

        Returns:
            Detailed reason string
        """
        num_segments = len(hallucinated_segments)
        ratio_pct = hallucination_ratio * 100

        if passed:
            reason = (
                f"Detected {num_segments} potentially hallucinated segment(s) "
                f"({ratio_pct:.1f}% of response), which is within the acceptable "
                f"threshold ({self.max_hallucinated_ratio * 100:.1f}%)."
            )
        else:
            reason = (
                f"Detected {num_segments} hallucinated segment(s) "
                f"({ratio_pct:.1f}% of response), exceeding the maximum allowed "
                f"ratio of {self.max_hallucinated_ratio * 100:.1f}%."
            )

        # Add details about the first few hallucinated segments
        if hallucinated_segments:
            reason += "\n\nHallucinated segments:"
            for i, segment in enumerate(hallucinated_segments[:3]):  # Show first 3
                text = segment["response_text"].strip()
                likelihood = segment["faithfulness_likelihood"]
                explanation = segment.get("explanation", "No explanation provided")
                reason += (
                    f"\n  {i + 1}. '{text}' "
                    f"(faithfulness: {likelihood:.2f})\n"
                    f"     Reason: {explanation}"
                )

            if len(hallucinated_segments) > 3:
                reason += (
                    f"\n  ... and {len(hallucinated_segments) - 3} more segment(s)"
                )

        return reason
