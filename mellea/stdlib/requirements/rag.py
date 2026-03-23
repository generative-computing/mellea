"""Requirements for RAG (Retrieval-Augmented Generation) workflows."""

from collections.abc import Iterable

from ...backends.adapters import AdapterMixin
from ...core import Backend, Context, Requirement, ValidationResult
from ..components import Document, Message


class CitationRequirement(Requirement):
    """Requirement that validates RAG responses have adequate citation coverage.

    Uses the find_citations intrinsic to identify which parts of an assistant's
    response are supported by explicit citations to retrieved documents. Content
    without citations below the minimum coverage threshold fails validation.

    **Important**: This requirement requires a HuggingFace backend (LocalHFBackend)
    as the find_citations intrinsic only works with HuggingFace models. Using other
    backends (Ollama, OpenAI, etc.) will result in a validation error.

    This requirement is designed for RAG workflows where you want to ensure
    responses properly cite their sources. It works with:
    - A user question in the context
    - Retrieved documents
    - An assistant response to validate

    Documents can be provided either:
    1. In the constructor (for reusable requirements with fixed documents)
    2. Attached to the assistant message in the context (for dynamic documents)

    Args:
        min_citation_coverage: Minimum ratio of cited content (0.0-1.0).
            The ratio of characters with citations to total response length
            must meet or exceed this threshold. Default is 0.8 (80% coverage).
        documents: Optional documents to validate against. Can be Document
            objects or strings (will be converted to Documents). If provided,
            these documents will be used instead of documents attached to
            messages in the context. Default is None (use context documents).
        description: Custom description for the requirement. If None,
            generates a description based on coverage threshold.

    Example:
        ```python
        from mellea.backends.huggingface import LocalHFBackend
        from mellea.stdlib.requirements.rag import CitationRequirement

        backend = LocalHFBackend(model_id="meta-llama/Llama-3.2-1B-Instruct")

        # Option 1: Documents in constructor
        req = CitationRequirement(
            documents=doc_objects,
            min_citation_coverage=0.8
        )

        # Option 2: Documents in context (original pattern)
        req = CitationRequirement(min_citation_coverage=0.8)
        ctx = ChatContext().add(
            Message("assistant", response, documents=doc_objects)
        )
        ```
    """

    def __init__(
        self,
        min_citation_coverage: float = 0.8,
        documents: Iterable[Document] | Iterable[str] | None = None,
        description: str | None = None,
    ):
        """Initialize citation coverage requirement."""
        if not 0.0 <= min_citation_coverage <= 1.0:
            raise ValueError(
                f"min_citation_coverage must be between 0.0 and 1.0, got {min_citation_coverage}"
            )

        self.min_citation_coverage = min_citation_coverage

        # Convert documents to Document objects if provided
        if documents is not None:
            self.documents: list[Document] | None = [
                doc
                if isinstance(doc, Document)
                else Document(doc_id=str(i), text=str(doc))
                for i, doc in enumerate(documents)
            ]
        else:
            self.documents = None

        # Generate description if not provided
        if description is None:
            description = (
                f"Response must have adequate citation coverage "
                f"(minimum {min_citation_coverage * 100:.0f}% of content cited)"
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
        """Validate citation coverage in the context using the backend.

        Args:
            backend: Backend to use for citation detection. Must be LocalHFBackend
                as the find_citations intrinsic only works with HuggingFace models.
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
                False, reason="Context is empty, cannot validate citation coverage"
            )

        last_message = messages[-1]
        if not isinstance(last_message, Message):
            return ValidationResult(
                False,
                reason="Last context item is not a Message, cannot validate citation coverage",
            )

        if last_message.role != "assistant":
            return ValidationResult(
                False,
                reason=f"Last message must be assistant response, got role: {last_message.role}",
            )

        response = last_message.content

        # Use constructor documents if provided, otherwise get from message
        if self.documents is not None:
            documents = self.documents
        else:
            # Access private _docs attribute since documents property returns formatted strings
            documents = last_message._docs or []

        if not documents:
            return ValidationResult(
                False,
                reason="No documents provided for citation validation. "
                "Either pass documents to CitationRequirement constructor "
                "or attach them to the assistant message.",
            )

        # Check backend compatibility
        if not isinstance(backend, AdapterMixin):
            return ValidationResult(
                False,
                reason=f"Backend {backend.__class__.__name__} does not support adapters required for citation detection",
            )

        # More specific check for HuggingFace backend
        try:
            from ...backends.huggingface import LocalHFBackend

            if not isinstance(backend, LocalHFBackend):
                return ValidationResult(
                    False,
                    reason=f"Citation detection requires LocalHFBackend (HuggingFace), "
                    f"but got {backend.__class__.__name__}. The find_citations intrinsic "
                    f"only works with HuggingFace models.",
                )
        except ImportError:
            return ValidationResult(
                False,
                reason="HuggingFace backend not available. Please install mellea[hf] to use citation detection.",
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

        # Handle empty response before calling intrinsic
        total_chars = len(response)
        if total_chars == 0:
            return ValidationResult(
                True, reason="Empty response has 100% citation coverage", score=1.0
            )

        # Call find_citations intrinsic
        try:
            # Import here to avoid circular dependency
            from ..components.intrinsic import rag

            citations: list[dict] = rag.find_citations(
                response, documents, context_before_response, backend
            )
        except Exception as e:
            return ValidationResult(
                False, reason=f"Citation detection intrinsic failed: {e!s}"
            )

        # Calculate citation coverage

        cited_chars = sum(
            citation["response_end"] - citation["response_begin"]
            for citation in citations
        )
        coverage_ratio = cited_chars / total_chars

        # Check against min_citation_coverage
        passed = coverage_ratio >= self.min_citation_coverage

        # Build detailed reason
        reason = self._build_reason(citations, coverage_ratio, passed)

        return ValidationResult(passed, reason=reason, score=coverage_ratio)

    def _build_reason(
        self, citations: list[dict], coverage_ratio: float, passed: bool
    ) -> str:
        """Build a detailed reason string for the validation result.

        Args:
            citations: List of citation records from find_citations
            coverage_ratio: Ratio of cited content
            passed: Whether validation passed

        Returns:
            Detailed reason string
        """
        num_citations = len(citations)
        coverage_pct = coverage_ratio * 100
        threshold_pct = self.min_citation_coverage * 100

        if passed:
            reason = (
                f"Response has adequate citation coverage "
                f"({coverage_pct:.1f}% cited, threshold: {threshold_pct:.1f}%)"
            )
        else:
            reason = (
                f"Response has insufficient citation coverage "
                f"({coverage_pct:.1f}% cited, threshold: {threshold_pct:.1f}%)"
            )

        # Add details about citations
        if citations:
            reason += f"\n\nCitations found ({num_citations}):"
            for i, citation in enumerate(citations[:5]):  # Show first 5
                response_text = citation["response_text"].strip()
                doc_id = citation.get("citation_doc_id", "unknown")
                citation_text = citation.get("citation_text", "").strip()
                # Truncate long texts
                if len(response_text) > 60:
                    response_text = response_text[:57] + "..."
                if len(citation_text) > 60:
                    citation_text = citation_text[:57] + "..."
                reason += f"\n  {i + 1}. '{response_text}' → Document '{doc_id}'"
                if citation_text:
                    reason += f"\n     Source: '{citation_text}'"

            if len(citations) > 5:
                reason += f"\n  ... and {len(citations) - 5} more citation(s)"
        else:
            reason += "\n\nNo citations found in the response."

        if not passed:
            uncited_pct = 100.0 - coverage_pct
            reason += (
                f"\n\nUncited content represents {uncited_pct:.1f}% of the response."
            )

        return reason


def citation_check(
    documents: Iterable[Document] | Iterable[str],
    min_citation_coverage: float = 0.8,
    description: str | None = None,
) -> CitationRequirement:
    """Create a citation coverage requirement with pre-attached documents.

    This is a convenience factory function that creates a CitationRequirement
    with documents already attached. This is useful when you have a fixed set of
    documents to validate against and want a cleaner API.

    **Important**: This requirement requires a HuggingFace backend (LocalHFBackend).

    Args:
        documents: Documents to check for citations. Can be Document objects
            or strings (will be converted to Documents).
        min_citation_coverage: Minimum ratio of cited content (0.0-1.0),
            defaults to 0.8 (80% coverage).
        description: Custom description for the requirement. If None,
            generates a description based on coverage threshold.

    Returns:
        A CitationRequirement with documents attached

    Example:
        ```python
        from mellea.backends.huggingface import LocalHFBackend
        from mellea.stdlib.requirements.rag import citation_check
        from mellea.stdlib.components import Document

        backend = LocalHFBackend(model_id="meta-llama/Llama-3.2-1B-Instruct")
        docs = [
            Document(doc_id="1", text="The sky is blue."),
            Document(doc_id="2", text="Grass is green.")
        ]
        req = citation_check(docs, min_citation_coverage=0.8)

        # Use with instruct() - no need to attach documents to messages
        result = m.instruct(
            "Answer: {{query}}",
            grounding_context={"query": "What color is the sky?"},
            requirements=[req],
            backend=backend,
            strategy=RejectionSamplingStrategy()
        )
        ```
    """
    return CitationRequirement(
        min_citation_coverage=min_citation_coverage,
        documents=documents,
        description=description,
    )
