"""Requirements for RAG (Retrieval-Augmented Generation) workflows."""

import json
import re
from collections.abc import Iterable

from ...backends.adapters import AdapterMixin
from ...core import Backend, CBlock, Context, Requirement, ValidationResult
from ...core.utils import FancyLogger
from ..components import Document, Message
from ..context import ChatContext

logger = FancyLogger.get_logger()


class GroundednessRequirement(Requirement):
    """Requirement that validates LLM responses are grounded by citations.

    This requirement implements a sophisticated 4-step validation pipeline to ensure
    that LLM responses are fully grounded by citations to provided documents:

    1. **Citation Generation**: Generate citations for the response using the
       find_citations intrinsic.
    2. **Citation Necessity**: Identify which response spans require citations (vs.
       conversational/inference text that doesn't need citations).
    3. **Citation Support**: For spans that need citations, assess the level of
       citation support: fully, partially, or not supported.
    4. **Groundedness Output**: Declare response grounded iff all spans needing
       citations are fully supported.

    **Important**: This requirement requires a HuggingFace backend (LocalHFBackend)
    as it uses both the find_citations intrinsic and LLM-as-Judge for assessment.

    Args:
        documents: Optional documents to validate against. Can be Document
            objects or strings (will be converted to Documents). If provided,
            these documents will be used instead of documents attached to
            messages in the context. Default is None (use context documents).
        allow_partial_support: Whether to accept partially supported spans as
            grounded. If False (default), response is grounded iff all spans
            needing citations are FULLY supported. If True, response is grounded
            if spans are fully or partially supported.
        description: Custom description for the requirement. If None,
            generates a default description.

    Example:
        ```python
        from mellea.backends.huggingface import LocalHFBackend
        from mellea.stdlib.requirements.rag import GroundednessRequirement

        backend = LocalHFBackend(model_id="ibm-granite/granite-4.0-micro")

        req = GroundednessRequirement(
            documents=doc_objects,
            allow_partial_support=False
        )

        result = await req.validate(backend, ctx)
        ```
    """

    def __init__(
        self,
        documents: Iterable[Document] | Iterable[str] | None = None,
        allow_partial_support: bool = False,
        description: str | None = None,
    ):
        """Initialize grounded requirement."""
        self.allow_partial_support = allow_partial_support

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
            support_text = (
                "fully supported" if not allow_partial_support else "supported"
            )
            description = (
                f"Response must be grounded in citations "
                f"(all spans needing citations must be {support_text})"
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
        """Validate groundedness of the response using the 4-step pipeline.

        Args:
            backend: Backend to use for citation detection and LLM judgment.
                Must be LocalHFBackend as this uses find_citations intrinsic.
            ctx: Context containing the conversation history
            format: Unused for this requirement
            model_options: Unused for this requirement

        Returns:
            ValidationResult with pass/fail status and detailed reason
        """
        # Extract last message (should be assistant response)
        messages = ctx.as_list()
        if not messages:
            return ValidationResult(
                False, reason="Context is empty, cannot validate groundedness"
            )

        last_message = messages[-1]
        if not isinstance(last_message, Message):
            return ValidationResult(
                False,
                reason="Last context item is not a Message, cannot validate groundedness",
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
            documents = last_message._docs or []

        if not documents:
            return ValidationResult(
                False,
                reason="No documents provided for grounding validation. "
                "Either pass documents to GroundednessRequirement constructor "
                "or attach them to the assistant message.",
            )

        # Check backend compatibility
        if not isinstance(backend, AdapterMixin):
            return ValidationResult(
                False,
                reason=f"Backend {backend.__class__.__name__} does not support adapters required for grounding validation",
            )

        # Handle empty response
        if not response.strip():
            return ValidationResult(
                True, reason="Empty response is considered grounded"
            )

        # Create context without the response being validated.
        # This removes the assistant's response from the context passed to intrinsics,
        # ensuring they only see the conversation history (user messages and prior context).
        # This is important because intrinsics need to assess citations/necessity/support
        # for the response independently, without the response itself influencing the judgment.
        # ChatContext is immutable, so we build a new one with all messages except the last.
        all_messages = ctx.as_list()
        if len(all_messages) > 1:
            context_before_response = ChatContext()
            for msg in all_messages[:-1]:
                context_before_response = context_before_response.add(msg)
        else:
            context_before_response = ChatContext()

        try:
            # Step 1: Citation Generation
            # Call intrinsic directly for explicit control over model options
            from ..components.intrinsic._util import call_intrinsic

            citation_context = context_before_response.add(
                Message("assistant", response, documents=list(documents))
            )
            citations: list[dict] = call_intrinsic(
                "citations", citation_context, backend
            )
            logger.debug(
                f"Step 1 - Citations generated: {len(citations)} citations found"
            )
            for i, cit in enumerate(citations):
                logger.debug(
                    f"  Citation {i}: response_begin={cit.get('response_begin')}, response_end={cit.get('response_end')}, text={cit.get('citation_text', '')[:50]}"
                )
        except Exception as e:
            return ValidationResult(
                False, reason=f"Citation generation intrinsic failed: {e!s}"
            )

        # Step 2: Citation Necessity
        try:
            span_necessity = await self._identify_citation_necessity(
                response, citations, backend, context_before_response
            )
            logger.debug(
                f"Step 2 - Citation necessity identified: {len(span_necessity)} spans"
            )
            for span_key, needs_cit in span_necessity.items():
                begin, end = span_key
                logger.debug(
                    f"  Span [{begin}:{end}] needs_citation={needs_cit}: '{response[begin:end][:50]}'"
                )
        except Exception as e:
            return ValidationResult(
                False, reason=f"Citation necessity assessment failed: {e!s}"
            )

        # Step 3: Citation Support
        try:
            span_support = await self._assess_citation_support(
                response, citations, span_necessity, backend, context_before_response
            )
            logger.debug(
                f"Step 3 - Citation support assessed: {len(span_support)} spans"
            )
            for span_key, support in span_support.items():
                begin, end = span_key
                logger.debug(f"  Span [{begin}:{end}] support={support}")
        except Exception as e:
            return ValidationResult(
                False, reason=f"Citation support assessment failed: {e!s}"
            )

        # Step 4: Groundedness Output
        passed, reason = self._build_groundedness_result(
            response, citations, span_necessity, span_support
        )

        return ValidationResult(passed, reason=reason)

    async def _identify_citation_necessity(
        self,
        response: str,
        citations: list[dict],
        backend: Backend,
        context: ChatContext,
    ) -> dict[tuple[int, int], bool]:
        """Identify which response spans need citations.

        Args:
            response: The assistant response text
            citations: Citation records from find_citations
            backend: Backend for LLM judgment
            context: Chat context

        Returns:
            Dictionary mapping span (begin, end) to boolean (needs_citation)
        """
        # Extract response spans: both cited and uncited portions
        spans = self._extract_response_spans(response, citations)

        if not spans:
            return {}

        # Build prompt for LLM to assess citation necessity
        prompt = self._build_necessity_prompt(response, spans)
        logger.debug(f"Necessity judgment prompt:\n{prompt}\n")

        # Get LLM judgment using prompt as the action/instruction
        try:
            action = CBlock(prompt)
            result, _ = await backend.generate_from_context(
                action,
                context,
                model_options={"temperature": 0.0, "max_new_tokens": 500},
            )
            await result.avalue()
            output_text = result.value
            if output_text is None:
                raise ValueError("LLM judgment returned None")
            logger.debug(f"LLM necessity judgment output: {output_text}")
        except Exception as e:
            raise ValueError(f"LLM judgment failed: {e}")

        # Parse LLM output
        span_necessity = self._parse_necessity_output(output_text, spans)

        return span_necessity

    async def _assess_citation_support(
        self,
        response: str,
        citations: list[dict],
        span_necessity: dict[tuple[int, int], bool],
        backend: Backend,
        context: ChatContext,
    ) -> dict[tuple[int, int], str]:
        """Assess level of support for spans that need citations.

        Uses a single batch LLM call to assess all spans requiring support assessment,
        rather than making individual calls per span. This significantly improves latency.

        Args:
            response: The assistant response text
            citations: Citation records from find_citations
            span_necessity: Mapping of span (begin, end) to needs_citation flag
            backend: Backend for LLM judgment
            context: Chat context

        Returns:
            Dictionary mapping span (begin, end) to support level
            ("FULLY_SUPPORTED", "PARTIALLY_SUPPORTED", or "NOT_SUPPORTED")
        """
        span_support: dict[tuple[int, int], str] = {}
        spans_to_assess: list[dict] = []  # Spans that need LLM assessment
        span_key_to_assess_idx: dict[tuple[int, int], int] = {}

        # First pass: identify spans needing LLM assessment vs those without citations
        for span_key, needs_citation in span_necessity.items():
            if not needs_citation:
                # Skip spans that don't need citations
                continue

            begin, end = span_key
            span_text = response[begin:end].strip()

            # Find citations that overlap with this span
            span_citations = []
            for citation in citations:
                # Check if citation overlaps with span
                # Both use half-open intervals [begin, end), so overlap occurs when:
                # citation.begin < span.end AND citation.end > span.begin
                if (
                    citation["response_begin"] < end
                    and citation["response_end"] > begin
                ):
                    span_citations.append(citation)

            if not span_citations:
                # No citations for this span → NOT_SUPPORTED (no need for LLM)
                span_support[span_key] = "NOT_SUPPORTED"
            else:
                # Has citations → collect for batch assessment
                idx = len(spans_to_assess)
                span_key_to_assess_idx[span_key] = idx
                spans_to_assess.append(
                    {
                        "span_key": span_key,
                        "text": span_text,
                        "citations": span_citations,
                    }
                )

        # If no spans need LLM assessment, return early
        if not spans_to_assess:
            logger.debug("No spans requiring LLM assessment for citation support")
            return span_support

        # Single batch LLM call for all spans
        prompt = self._build_batch_support_prompt(response, spans_to_assess)
        logger.debug(
            f"Batch support assessment prompt (spans={len(spans_to_assess)}):\n{prompt}\n"
        )

        try:
            action = CBlock(prompt)
            result, _ = await backend.generate_from_context(
                action,
                context,
                model_options={"temperature": 0.0, "max_new_tokens": 500},
            )
            await result.avalue()
            output_text = result.value
            if output_text is None:
                raise ValueError("LLM judgment returned None")
        except Exception as e:
            raise ValueError(f"Batch LLM judgment for support failed: {e}")

        # Parse batch results
        batch_results = self._parse_batch_support_output(
            output_text, len(spans_to_assess)
        )
        logger.debug(f"Batch support assessment results: {batch_results}")

        # Map results back to span_keys
        for span_key, idx in span_key_to_assess_idx.items():
            support_level = batch_results.get(idx, "NOT_SUPPORTED")
            span_support[span_key] = support_level

        logger.debug(
            f"Citation support assessment complete: {len(span_support)} spans assessed"
        )
        return span_support

    def _extract_response_spans(
        self, response: str, citations: list[dict]
    ) -> list[dict]:
        """Extract all response spans (cited and uncited) from response.

        Uses sentence-level segmentation for clarity. Each span is either:
        - A full response span covered by citations
        - An uncited gap between citations

        Args:
            response: The assistant response text
            citations: Citation records with response_begin/response_end

        Returns:
            List of span dictionaries with begin, end, text
        """
        if not response.strip():
            return []

        # Build a list of covered ranges (intervals) from citations
        covered_ranges: list[tuple[int, int]] = []
        for citation in citations:
            begin = citation["response_begin"]
            end = min(citation["response_end"], len(response))
            if begin < end:
                covered_ranges.append((begin, end))

        # Merge overlapping ranges for efficient coverage lookup
        covered_ranges.sort()
        merged_ranges: list[tuple[int, int]] = []
        for begin, end in covered_ranges:
            if merged_ranges and begin <= merged_ranges[-1][1]:
                merged_ranges[-1] = (
                    merged_ranges[-1][0],
                    max(merged_ranges[-1][1], end),
                )
            else:
                merged_ranges.append((begin, end))

        covered_chars = sum(end - begin for begin, end in merged_ranges)
        logger.debug(
            f"Response span extraction - coverage: {covered_chars}/{len(response)} chars covered by citations"
        )

        # Check if a position is covered by any citation
        def is_covered(pos: int) -> bool:
            for begin, end in merged_ranges:
                if begin <= pos < end:
                    return True
                if begin > pos:
                    break
            return False

        # Extract spans by finding boundaries between covered and uncovered regions
        spans: list[dict] = []
        current_span_start = 0
        current_is_covered = is_covered(0) if response else False

        for i in range(1, len(response) + 1):
            # Check if we're at a boundary (coverage changed or end of response)
            at_end = i == len(response)
            next_is_covered = False if at_end else is_covered(i)
            at_boundary = at_end or next_is_covered != current_is_covered

            if at_boundary:
                span_text = response[current_span_start:i].strip()
                if span_text:  # Only include non-empty spans
                    spans.append(
                        {
                            "begin": current_span_start,
                            "end": i,
                            "text": span_text,
                            "is_cited": current_is_covered,
                        }
                    )

                current_span_start = i
                if not at_end:
                    current_is_covered = next_is_covered

        logger.debug(f"Response span extraction - extracted {len(spans)} spans")
        for span in spans:
            logger.debug(
                f"  Span: is_cited={span['is_cited']}, text='{span['text'][:60]}'"
            )

        return spans

    def _build_necessity_prompt(self, response: str, spans: list[dict]) -> str:
        """Build prompt to determine if spans need citations.

        Args:
            response: The full response text
            spans: List of response spans

        Returns:
            Formatted prompt for LLM
        """
        # Create labeled spans for LLM
        labeled_spans = []
        for i, span in enumerate(spans):
            labeled_spans.append(f'{{"span_id": {i}, "text": "{span["text"]}"}}')

        spans_text = ",\n".join(labeled_spans)

        prompt = (
            "You are given a response and a set of spans extracted from the response. "
            "For each span, determine whether it contains factual claims that require grounding in source material. "
            "A span needs citation if it makes factual or informational claims about the world. "
            "A span does NOT need citation only if it is:\n"
            "  - An explicit I-do-not-know or uncertainty statement (e.g., 'I don't have information about...')\n"
            "  - Purely conversational or transitional text (e.g., 'Let me explain', 'In summary')\n"
            "  - A direct restatement or reformulation of information already stated in another span\n\n"
            "Output a JSON array of the form "
            '[{"span_id": ..., "needs_citation": ...}, ...], with one object for each span. '
            'Set "needs_citation" to "yes" if the span contains factual claims needing grounding, '
            'or "no" if it is exempt per the criteria above. '
            "Output ONLY the JSON array, no other text.\n\n"
            f"Response:\n{response}\n\n"
            f"Spans to Evaluate:\n[{spans_text}]\n\n"
            "JSON Output:\n"
        )
        return prompt

    def _build_support_prompt(self, span_text: str, span_citations: list[dict]) -> str:
        """Build prompt to assess citation support level.

        Args:
            span_text: The response span text
            span_citations: Citation records for this span

        Returns:
            Formatted prompt for LLM
        """
        citations_text = []
        for i, citation in enumerate(span_citations):
            citation_text = citation.get("citation_text", "")
            doc_id = citation.get("citation_doc_id", "unknown")
            citations_text.append(
                f'Citation {i} (from doc {doc_id}): "{citation_text}"'
            )

        citations_formatted = "\n".join(citations_text)

        prompt = (
            "Assess the level of support for a response span based on provided citations.\n\n"
            f"Response span:\n{span_text}\n\n"
            f"Provided citations:\n{citations_formatted}\n\n"
            "Determine if the citations fully support, partially support, or do not support the span.\n"
            "Respond with ONLY one of these three words: FULLY_SUPPORTED, PARTIALLY_SUPPORTED, or NOT_SUPPORTED."
        )
        return prompt

    def _build_batch_support_prompt(
        self, response: str, spans_to_assess: list[dict]
    ) -> str:
        """Build prompt to assess citation support level for multiple spans at once.

        This batch approach reduces latency by combining all span assessments into a
        single LLM call instead of making individual calls per span.

        Args:
            response: The full assistant response text
            spans_to_assess: List of span dicts with keys:
                - text: span text
                - citations: list of citation records for this span

        Returns:
            Formatted prompt for LLM expecting JSON array output
        """
        # Build structured list of spans with their citations
        span_assessments = []
        for i, span_info in enumerate(spans_to_assess):
            span_text = span_info["text"]
            span_citations = span_info["citations"]

            # Format citations for this span
            citations_lines = []
            for j, citation in enumerate(span_citations):
                citation_text = citation.get("citation_text", "")
                doc_id = citation.get("citation_doc_id", "unknown")
                citations_lines.append(
                    f'    Citation {j} (from doc {doc_id}): "{citation_text}"'
                )

            citations_formatted = "\n".join(citations_lines)

            # Build span entry
            span_entry = (
                f'  {{"span_id": {i}, '
                f'"text": "{span_text}", '
                f'"citations": [\n{citations_formatted}\n  ]}}'
            )
            span_assessments.append(span_entry)

        spans_formatted = ",\n".join(span_assessments)

        prompt = (
            "Assess the level of support for each response span based on provided citations.\n\n"
            "For each span, determine if the citations fully support, partially support, "
            "or do not support the span.\n\n"
            "Respond with a JSON array of the form:\n"
            '[{"span_id": ..., "support_level": ...}, ...]\n\n'
            "Support levels must be ONLY one of: FULLY_SUPPORTED, PARTIALLY_SUPPORTED, or NOT_SUPPORTED.\n\n"
            f"Response context:\n{response}\n\n"
            f"Spans to assess:\n[\n{spans_formatted}\n]\n\n"
            "JSON Output:\n"
        )
        return prompt

    def _parse_batch_support_output(
        self, output_text: str, expected_count: int
    ) -> dict[int, str]:
        """Parse LLM output for batch support level assessments.

        Args:
            output_text: LLM output text
            expected_count: Expected number of span assessments

        Returns:
            Dictionary mapping span assessment index to support level
            ("FULLY_SUPPORTED", "PARTIALLY_SUPPORTED", or "NOT_SUPPORTED")
        """
        result: dict[int, str] = {}

        logger.debug(f"Parsing batch support output (expected {expected_count} spans)")

        try:
            # Extract JSON array from output
            json_start = output_text.find("[")
            if json_start == -1:
                raise ValueError("No JSON array start found in LLM output")

            # Find matching closing bracket
            bracket_count = 0
            json_end = -1
            for i in range(json_start, len(output_text)):
                if output_text[i] == "[":
                    bracket_count += 1
                elif output_text[i] == "]":
                    bracket_count -= 1
                    if bracket_count == 0:
                        json_end = i + 1
                        break

            if json_end == -1:
                logger.debug("No matching closing bracket found, attempting recovery")
                json_text = output_text[json_start:] + "]"
            else:
                json_text = output_text[json_start:json_end]

            logger.debug(f"Extracted JSON text (first 200 chars): {json_text[:200]}")

            # Try to parse
            try:
                judgments = json.loads(json_text)
            except json.JSONDecodeError as e:
                logger.debug(f"JSON parse error: {e}, attempting repair")
                # Try to find and close the last object
                last_brace = json_text.rfind("}")
                if last_brace != -1:
                    json_text = json_text[: last_brace + 1] + "]"
                    judgments = json.loads(json_text)
                else:
                    raise

            logger.debug(f"Parsed {len(judgments)} judgments from batch output")

            # Process judgments
            for judgment in judgments:
                if not isinstance(judgment, dict):
                    logger.debug(f"Skipping non-dict judgment: {judgment}")
                    continue

                span_id = judgment.get("span_id")
                support_level_raw = judgment.get("support_level", "").upper().strip()

                logger.debug(
                    f"  Judgment: span_id={span_id}, support_level={support_level_raw}"
                )

                if span_id is None:
                    logger.debug("Skipping judgment with no span_id")
                    continue

                # Normalize support level
                if "FULLY" in support_level_raw and "SUPPORTED" in support_level_raw:
                    support_level = "FULLY_SUPPORTED"
                elif (
                    "PARTIALLY" in support_level_raw
                    and "SUPPORTED" in support_level_raw
                ):
                    support_level = "PARTIALLY_SUPPORTED"
                else:
                    support_level = "NOT_SUPPORTED"

                result[span_id] = support_level

            # Ensure all expected spans have results (default to NOT_SUPPORTED if missing)
            for i in range(expected_count):
                if i not in result:
                    logger.debug(
                        f"Span {i} not in batch output, defaulting to NOT_SUPPORTED"
                    )
                    result[i] = "NOT_SUPPORTED"

        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(
                f"Parse error in batch support output: {e}, assuming all NOT_SUPPORTED"
            )
            # On parse error, conservatively assume all spans are not supported
            for i in range(expected_count):
                result[i] = "NOT_SUPPORTED"

        return result

    def _parse_necessity_output(
        self, output_text: str, spans: list[dict]
    ) -> dict[tuple[int, int], bool]:
        """Parse LLM output for citation necessity judgments.

        Args:
            output_text: LLM output text
            spans: Original span list

        Returns:
            Dictionary mapping span (begin, end) to needs_citation boolean
        """
        span_necessity: dict[tuple[int, int], bool] = {}

        logger.debug(f"Parsing necessity output: {output_text[:300]}")

        try:
            # Try to extract JSON array from output
            # Use a non-greedy match first to find the start, then look for closing bracket
            json_start = output_text.find("[")
            if json_start == -1:
                raise ValueError("No JSON array start found in LLM output")

            # Find matching closing bracket
            bracket_count = 0
            json_end = -1
            for i in range(json_start, len(output_text)):
                if output_text[i] == "[":
                    bracket_count += 1
                elif output_text[i] == "]":
                    bracket_count -= 1
                    if bracket_count == 0:
                        json_end = i + 1
                        break

            if json_end == -1:
                # Didn't find matching bracket, try to recover
                logger.debug("No matching closing bracket found, attempting recovery")
                # Take from start to end and close with ]
                json_text = output_text[json_start:] + "]"
            else:
                json_text = output_text[json_start:json_end]

            # Try to parse
            try:
                judgments = json.loads(json_text)
            except json.JSONDecodeError as e:
                logger.debug(f"JSON parse error: {e}, attempting repair")
                # Try to find and close the last object
                last_brace = json_text.rfind("}")
                if last_brace != -1:
                    json_text = json_text[: last_brace + 1] + "]"
                    judgments = json.loads(json_text)
                else:
                    raise

            logger.debug(f"Parsed JSON judgments: {len(judgments)} judgments")

            # Map judgments to spans
            for judgment in judgments:
                if not isinstance(judgment, dict):
                    continue

                span_id = judgment.get("span_id")
                needs_citation_flag = judgment.get("needs_citation", "").lower().strip()

                logger.debug(
                    f"  Judgment: span_id={span_id}, needs_citation={needs_citation_flag}"
                )

                if span_id is not None and 0 <= span_id < len(spans):
                    span = spans[span_id]
                    span_key = (span["begin"], span["end"])
                    # Handle variations: "yes", "true", "1" -> True
                    span_necessity[span_key] = needs_citation_flag in (
                        "yes",
                        "true",
                        "1",
                    )

            # Ensure all spans are in the result
            for span in spans:
                span_key = (span["begin"], span["end"])
                if span_key not in span_necessity:
                    # Default: assume needs citation if not specified
                    logger.debug(
                        f"  Span {span_key} not in judgments, defaulting to True"
                    )
                    span_necessity[span_key] = True

        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(
                f"Parse error in necessity output: {e}, assuming all spans need citations"
            )
            # On parse error, conservatively assume all spans need citations
            for span in spans:
                span_necessity[(span["begin"], span["end"])] = True

        return span_necessity

    def _parse_support_output(self, output_text: str) -> str:
        """Parse LLM output for support level.

        Args:
            output_text: LLM output text

        Returns:
            Support level: "FULLY_SUPPORTED", "PARTIALLY_SUPPORTED", or "NOT_SUPPORTED"
        """
        output_upper = output_text.upper()

        if "FULLY" in output_upper and "SUPPORTED" in output_upper:
            return "FULLY_SUPPORTED"
        elif "PARTIALLY" in output_upper and "SUPPORTED" in output_upper:
            return "PARTIALLY_SUPPORTED"
        else:
            return "NOT_SUPPORTED"

    def _build_groundedness_result(
        self,
        response: str,
        citations: list[dict],
        span_necessity: dict[tuple[int, int], bool],
        span_support: dict[tuple[int, int], str],
    ) -> tuple[bool, str]:
        """Build final groundedness result.

        Args:
            response: The assistant response
            citations: Citation records
            span_necessity: Mapping of span to needs_citation flag
            span_support: Mapping of span to support level

        Returns:
            Tuple of (passed: bool, reason: str)
        """
        # Find problematic spans (those that need citations but aren't fully supported)
        problematic_spans: list[tuple[int, int]] = []

        for span_key, needs_citation in span_necessity.items():
            if not needs_citation:
                continue

            support = span_support.get(span_key, "NOT_SUPPORTED")

            # Check if support is acceptable
            if support == "NOT_SUPPORTED":
                problematic_spans.append(span_key)
            elif support == "PARTIALLY_SUPPORTED" and not self.allow_partial_support:
                problematic_spans.append(span_key)

        # Determine pass/fail
        passed = len(problematic_spans) == 0

        # Build reason string
        if passed:
            reason = "Response is grounded in citations."
        else:
            reason = "Response is not grounded - the following spans are not properly supported:\n\n"

            for begin, end in problematic_spans:
                span_text = response[begin:end]
                support = span_support.get((begin, end), "NOT_SUPPORTED")
                reason += f'- "{span_text}" [{support}]\n'

        # Add summary statistics
        total_spans_needing_citations = sum(
            1 for needs in span_necessity.values() if needs
        )
        fully_supported = sum(
            1
            for span_key, needs in span_necessity.items()
            if needs and span_support.get(span_key) == "FULLY_SUPPORTED"
        )

        reason += (
            f"\nSummary: {fully_supported}/{total_spans_needing_citations} "
            f"spans needing citations are fully supported."
        )

        return passed, reason
