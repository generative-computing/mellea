# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Requirements for RAG (Retrieval-Augmented Generation) workflows."""

import json
import re
from collections.abc import Iterable

from ...backends.adapters import AdapterMixin
from ...backends.model_options import ModelOption
from ...core import (
    Backend,
    CBlock,
    Context,
    MelleaLogger,
    Requirement,
    ValidationResult,
)
from ..components import Document, Message
from ..context import ChatContext

logger = MelleaLogger.get_logger()

_SUPPORT_LEVEL_RANK: dict[str, int] = {
    "FULLY_SUPPORTED": 0,
    "PARTIALLY_SUPPORTED": 1,
    "NOT_SUPPORTED": 2,
}


def _normalise_span_id(raw: object, expected_count: int) -> int | None:
    """Normalise a model-returned span identifier."""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        span_id = raw
    elif isinstance(raw, float) and raw.is_integer():
        span_id = int(raw)
    elif isinstance(raw, str):
        match = re.fullmatch(r"[0-9]+", raw.strip())
        if match is None:
            return None
        span_id = int(match.group())
    else:
        return None

    return span_id if 0 <= span_id < expected_count else None


def _normalise_label(raw: object) -> str:
    """Normalise a model-returned string label."""
    return raw.strip() if isinstance(raw, str) else ""


def _normalise_needs_citation(raw: object) -> bool | None:
    """Normalise a model-returned citation-necessity label.

    Returns:
        `True` or `False` for a recognised value, `None` when the value is
        unreadable. Only the integers 0 and 1 are recognised.
    """
    # bool is a subclass of int, so it must be checked first.
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int):
        if raw == 1:
            return True
        if raw == 0:
            return False
        return None
    if not isinstance(raw, str):
        return None

    label = raw.strip().lower()
    if label in ("yes", "true", "1"):
        return True
    if label in ("no", "false", "0"):
        return False
    return None


def _merge_necessity(a: bool | None, b: bool | None) -> bool | None:
    """Merge two necessity judgments for one span: `True` beats `None` beats `False`."""
    if a is True or b is True:
        return True
    if a is None or b is None:
        return None
    return False


def _reduce_support_levels(levels: Iterable[str | None]) -> str | None:
    """Reduce several support judgments for one span to one level.

    Any `NOT_SUPPORTED` wins; otherwise any unreadable (`None`) judgment makes
    the result `None`; otherwise the most pessimistic recognised level wins.
    An empty collection gives `None`.
    """
    collected = list(levels)
    if not collected:
        return None
    if "NOT_SUPPORTED" in collected:
        return "NOT_SUPPORTED"
    if any(level is None for level in collected):
        return None
    return max(
        (level for level in collected if level is not None),
        key=_SUPPORT_LEVEL_RANK.__getitem__,
    )


class GroundednessRequirement(Requirement):
    """Requirement that validates LLM responses are grounded by citations.

    This requirement implements a sophisticated 4-step validation pipeline to ensure
    that LLM responses are fully grounded by citations to provided documents:

    1. **Citation Generation**: Generate citations for the response using the
       find_citations adapter function.
    2. **Citation Necessity**: Identify which response spans require citations (vs.
       conversational/inference text that doesn't need citations).
    3. **Citation Support**: For spans that need citations, assess the level of
       citation support: fully, partially, or not supported.
    4. **Groundedness Output**: Declare response grounded iff all spans needing
       citations are fully supported.

    **Important**: The backend must implement `AdapterMixin` and must actually
    provide the citation adapter used by the find_citations adapter function (the
    `AdapterMixin` check alone does not guarantee that; if the adapter is missing,
    `validate()` returns an errored result rather than a verdict). The backend must
    also support the two LLM-as-Judge calls (`generate_from_context` with a chat
    context and `ModelOption.MAX_NEW_TOKENS`).

    If validation cannot run (setup problem, adapter or backend failure, or
    unreadable judgment output), the returned `ValidationResult` has `error` set
    and `reason` set to `None`, instead of reporting an ordinary failure.

    Args:
        documents: Optional documents to validate against. Can be Document
            objects or strings (will be converted to Documents). If provided,
            these documents will be used instead of documents attached to
            messages in the context. Default is None (use context documents).
        allow_partial_support: Whether to accept partially supported spans as
            grounded. If False (default), response is grounded iff all spans
            needing citations are FULLY supported. If True, response is grounded
            if spans are fully or partially supported.
        max_new_tokens: Maximum tokens for LLM judgment outputs. Increase this
            if LLM outputs are being truncated (particularly for complex
            responses with many spans). Default is 500.
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
        max_new_tokens: int = 500,
        description: str | None = None,
    ):
        """Initialize grounded requirement."""
        self.allow_partial_support = allow_partial_support
        self.max_new_tokens = max_new_tokens

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
                Must implement `AdapterMixin` and provide the citation adapter
                used by the find_citations adapter function. The backend must
                also support the two LLM-as-Judge calls (`generate_from_context`
                with a chat context and `ModelOption.MAX_NEW_TOKENS`).
            ctx: Context containing the conversation history
            format: Unused for this requirement
            model_options: Unused for this requirement

        Returns:
            ValidationResult with pass/fail status and detailed reason. When
            validation could not run or could not reach a verdict, an errored
            result (`reason=None`, `error` set) is returned instead.
        """
        # Extract last message (should be assistant response)
        messages = ctx.as_list()
        if not messages:
            return ValidationResult(
                False,
                reason=None,
                error=ValueError("Context is empty, cannot validate groundedness"),
            )

        last_message = messages[-1]
        if not isinstance(last_message, Message):
            return ValidationResult(
                False,
                reason=None,
                error=ValueError(
                    "Last context item is not a Message, cannot validate groundedness"
                ),
            )

        if last_message.role != "assistant":
            return ValidationResult(
                False,
                reason=None,
                error=ValueError(
                    f"Last message must be assistant response, got role: {last_message.role}"
                ),
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
                reason=None,
                error=ValueError(
                    "No documents provided for grounding validation. "
                    "Either pass documents to GroundednessRequirement constructor "
                    "or attach them to the assistant message."
                ),
            )

        # Check backend compatibility
        if not isinstance(backend, AdapterMixin):
            return ValidationResult(
                False,
                reason=None,
                error=ValueError(
                    f"Backend {backend.__class__.__name__} does not support adapters required for grounding validation"
                ),
            )

        # Handle empty response
        if not response.strip():
            return ValidationResult(
                True, reason="Empty response is considered grounded"
            )

        # Create context without the response being validated.
        # This removes the assistant's response from the context passed to adapter functions,
        # ensuring they only see the conversation history (user messages and prior context).
        # This is important because adapter functions need to assess citations/necessity/support
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
            # Import lazily to avoid circular dependency
            from ..components.intrinsic.rag import find_citations

            citations: list[dict] = find_citations(
                response, list(documents), context_before_response, backend
            )
            logger.debug(
                f"Step 1 - Citations generated: {len(citations)} citations found"
            )
            for i, cit in enumerate(citations):
                logger.debug(
                    f"  Citation {i}: response_begin={cit.get('response_begin')}, response_end={cit.get('response_end')}, text={cit.get('citation_text', '')[:50]}"
                )
        except Exception as e:
            logger.exception("Citation generation adapter function failed")
            return ValidationResult(False, reason=None, error=e)

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
            logger.exception("Citation necessity assessment failed")
            return ValidationResult(False, reason=None, error=e)

        # Step 3: Citation Support
        try:
            span_support = await self._assess_citation_support(
                response,
                citations,
                span_necessity,
                backend,
                context_before_response,
                documents,
            )
            logger.debug(
                f"Step 3 - Citation support assessed: {len(span_support)} spans"
            )
            for span_key, support in span_support.items():
                begin, end = span_key
                logger.debug(f"  Span [{begin}:{end}] support={support}")
        except Exception as e:
            logger.exception("Citation support assessment failed")
            return ValidationResult(False, reason=None, error=e)

        # Step 4: Groundedness Output
        try:
            return self._build_groundedness_result(
                response, citations, span_necessity, span_support
            )
        except Exception as e:
            logger.exception("Groundedness result construction failed")
            return ValidationResult(False, reason=None, error=e)

    async def _identify_citation_necessity(
        self,
        response: str,
        citations: list[dict],
        backend: Backend,
        context: ChatContext,
    ) -> dict[tuple[int, int], bool | None]:
        """Identify which response spans need citations.

        Args:
            response: The assistant response text
            citations: Citation records from find_citations
            backend: Backend for LLM judgment
            context: Chat context

        Returns:
            Dictionary mapping span (begin, end) to `True` or `False`
            (needs_citation), or `None` when the judgment was unreadable
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
                model_options={
                    "temperature": 0.0,
                    ModelOption.MAX_NEW_TOKENS: self.max_new_tokens,
                },
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
        span_necessity: dict[tuple[int, int], bool | None],
        backend: Backend,
        context: ChatContext,
        documents: list[Document],
    ) -> dict[tuple[int, int], str | None]:
        """Assess level of support for spans that need citations.

        Uses a single batch LLM call to assess all spans requiring support assessment,
        rather than making individual calls per span. This significantly improves latency.
        Spans whose necessity could not be read (`None`) are assessed like spans that
        need citations; only spans known not to need citations (`False`) are skipped.

        Args:
            response: The assistant response text
            citations: Citation records from find_citations
            span_necessity: Mapping of span (begin, end) to needs_citation flag
                (`None` = unreadable)
            backend: Backend for LLM judgment
            context: Chat context
            documents: List of source documents for context in LLM assessment

        Returns:
            Dictionary mapping span (begin, end) to support level
            ("FULLY_SUPPORTED", "PARTIALLY_SUPPORTED", or "NOT_SUPPORTED"), or
            `None` when the judgment was unreadable. Spans with necessity `False`
            have no entry.
        """
        span_support: dict[tuple[int, int], str | None] = {}
        spans_to_assess: list[dict] = []  # Spans that need LLM assessment
        span_key_to_assess_idx: dict[tuple[int, int], int] = {}

        # First pass: identify spans needing LLM assessment vs those without citations
        for span_key, needs_citation in span_necessity.items():
            if needs_citation is False:
                # Skip spans that are known not to need citations
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
        prompt = self._build_batch_support_prompt(response, spans_to_assess, documents)
        logger.debug(
            f"Batch support assessment prompt (spans={len(spans_to_assess)}):\n{prompt}\n"
        )

        try:
            action = CBlock(prompt)
            result, _ = await backend.generate_from_context(
                action,
                context,
                model_options={
                    "temperature": 0.0,
                    ModelOption.MAX_NEW_TOKENS: self.max_new_tokens,
                },
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
            support_level = batch_results.get(idx)
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
            if merged_ranges and begin < merged_ranges[-1][1]:
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

        # Extract spans by finding boundaries between covered and uncovered regions
        # Iterate over merged_ranges boundaries rather than every character for efficiency
        spans: list[dict] = []

        # Build boundary points from merged ranges
        boundaries = [0]  # Start of response
        for begin, end in merged_ranges:
            boundaries.append(begin)
            boundaries.append(end)
        boundaries.append(len(response))  # End of response
        boundaries = sorted(set(boundaries))  # Remove duplicates and sort

        # Process each span between boundaries
        for i in range(len(boundaries) - 1):
            span_start = boundaries[i]
            span_end = boundaries[i + 1]

            # Determine if this span is covered by any merged range
            is_cited = any(
                begin <= span_start and span_end <= end for begin, end in merged_ranges
            )

            span_text = response[span_start:span_end].strip()
            if span_text:  # Only include non-empty spans
                spans.append(
                    {
                        "begin": span_start,
                        "end": span_end,
                        "text": span_text,
                        "is_cited": is_cited,
                    }
                )

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
            labeled_spans.append(json.dumps({"span_id": i, "text": span["text"]}))

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
            '[{"span_id": 0, "needs_citation": "yes"}], with one object for each span. '
            'Set "needs_citation" to "yes" if the span contains factual claims needing grounding, '
            'or "no" if it is exempt per the criteria above. '
            "Output ONLY the JSON array, no other text.\n\n"
            f"Response:\n{response}\n\n"
            f"Spans to Evaluate:\n[{spans_text}]\n\n"
            "JSON Output:\n"
        )
        return prompt

    def _build_batch_support_prompt(
        self, response: str, spans_to_assess: list[dict], documents: list[Document]
    ) -> str:
        """Build prompt to assess citation support level for multiple spans at once.

        This batch approach reduces latency by combining all span assessments into a
        single LLM call instead of making individual calls per span.

        Args:
            response: The full assistant response text
            spans_to_assess: List of span dicts with keys:
                - text: span text
                - citations: list of citation records for this span
            documents: List of source documents for context

        Returns:
            Formatted prompt for LLM expecting JSON array output
        """
        # json.dumps keeps this valid JSON even if text contains quotes/newlines.
        spans_payload = []
        for i, span_info in enumerate(spans_to_assess):
            span_text = span_info["text"]
            span_citations = span_info["citations"]

            evidence_entries = []
            for j, citation in enumerate(span_citations):
                citation_text = citation.get("citation_text", "")
                doc_id = citation.get("citation_doc_id", "unknown")
                evidence_entries.append(
                    f'Evidence {j} (from doc {doc_id}): "{citation_text}"'
                )

            # "evidence" (not "citations") avoids echoing the output key name (#1316).
            spans_payload.append(
                {"span_id": i, "text": span_text, "evidence": evidence_entries}
            )

        spans_formatted = json.dumps(spans_payload, indent=2)

        # Build source documents section for context
        documents_section = ""
        if documents:
            doc_lines = []
            for doc in documents:
                doc_id = doc.doc_id if hasattr(doc, "doc_id") else "unknown"
                doc_text = doc.text if hasattr(doc, "text") else str(doc)
                doc_lines.append(f"Document {doc_id}:\n{doc_text}")
            documents_section = "Source Documents:\n" + "\n\n".join(doc_lines) + "\n\n"

        prompt = (
            "Assess the level of support for each response span based on its provided evidence "
            "and the source documents.\n\n"
            "For each span, determine if the evidence fully supports, partially supports, "
            "or does not support the span. Consider the full context from the source documents "
            "where the evidence appears.\n\n"
            "Respond with a flat JSON array (no nested arrays), with one object for each span. Example output:\n"
            '[{"span_id": 0, "support_level": "FULLY_SUPPORTED"}]\n\n'
            "Support levels must be ONLY one of: FULLY_SUPPORTED, PARTIALLY_SUPPORTED, or NOT_SUPPORTED.\n\n"
            f"{documents_section}"
            f"Response context:\n{response}\n\n"
            f"Spans to assess:\n{spans_formatted}\n\n"
            "JSON Output:\n"
        )
        return prompt

    def _extract_and_repair_json_array(self, output_text: str) -> list:
        """Extract and repair a JSON array from LLM output.

        Handles common issues like missing closing brackets and incomplete JSON.

        Args:
            output_text: Raw LLM output text

        Returns:
            Parsed JSON array

        Raises:
            ValueError: If no JSON array start found
            json.JSONDecodeError: If JSON cannot be parsed even after repair attempts
        """
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
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.debug(f"JSON parse error: {e}, attempting repair")
            # Try to find and close the last object
            last_brace = json_text.rfind("}")
            if last_brace != -1:
                json_text = json_text[: last_brace + 1] + "]"
                return json.loads(json_text)
            else:
                raise

    def _parse_batch_support_output(
        self, output_text: str, expected_count: int
    ) -> dict[int, str | None]:
        """Parse LLM output for batch support level assessments.

        Args:
            output_text: LLM output text
            expected_count: Expected number of span assessments

        Returns:
            Dictionary with exactly the keys `range(expected_count)`, mapping each
            span assessment index to its support level ("FULLY_SUPPORTED",
            "PARTIALLY_SUPPORTED", or "NOT_SUPPORTED"), or `None` when that
            span's judgment was unreadable, missing or unrecognised. Multiple
            judgments for one span reduce via `_reduce_support_levels`.
        """
        logger.debug(f"Parsing batch support output (expected {expected_count} spans)")

        try:
            judgments = self._extract_and_repair_json_array(output_text)
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(
                f"Parse error in batch support output: {e}, treating all spans as unknown"
            )
            return dict.fromkeys(range(expected_count))

        logger.debug(f"Parsed {len(judgments)} judgments from batch output")

        # Normalize each nested citation through the same substring
        # logic the flat path uses below, so near-miss labels like
        # "FULLY SUPPORTED" (space) aren't silently downgraded.
        def _norm(raw: object) -> str | None:
            # Split camelCase boundaries and collapse separators to spaces
            # so \b sees word edges around "NOT"/"UNSUPPORTED" regardless
            # of delimiter style (underscore, hyphen, or camelCase).
            label = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", _normalise_label(raw))
            raw = re.sub(r"[^A-Za-z0-9]+", " ", label).upper().strip()
            if re.search(r"\b(?:NOT|UNSUPPORTED)\b", raw):
                return "NOT_SUPPORTED"
            if "FULLY" in raw and "SUPPORTED" in raw:
                return "FULLY_SUPPORTED"
            if "PARTIALLY" in raw and "SUPPORTED" in raw:
                return "PARTIALLY_SUPPORTED"
            return None

        per_span: dict[int, list[str | None]] = {}
        for judgment in judgments:
            if not isinstance(judgment, dict):
                logger.debug(f"Skipping non-dict judgment: {judgment}")
                continue

            span_id = _normalise_span_id(judgment.get("span_id"), expected_count)
            # Case is preserved here (not upper-cased) so _norm can still
            # see camelCase word boundaries like "notFullySupported".
            level: str | None
            if "support_level" in judgment:
                support_level_raw = _normalise_label(judgment.get("support_level"))
                level = _norm(support_level_raw) if support_level_raw else None
            else:
                # Handle nested format: {"span_id": 0, "evidence": [{"support_level": "..."}]}
                # Checks both "evidence" and "citations" (prior key, kept defensively).
                nested_source: list[object] = []
                malformed_nested = False
                for nested_key in ("evidence", "citations"):
                    if nested_key in judgment:
                        nested_value = judgment.get(nested_key)
                        if isinstance(nested_value, list):
                            nested_source.extend(nested_value)
                        else:
                            malformed_nested = True
                nested_levels: list[str | None] = [
                    _norm(cit.get("support_level")) if isinstance(cit, dict) else None
                    for cit in nested_source
                ]
                if malformed_nested:
                    nested_levels.append(None)
                level = _reduce_support_levels(nested_levels)

            logger.debug(f"  Judgment: span_id={span_id}, support_level={level}")

            if span_id is None:
                logger.debug("Skipping judgment with no span_id")
                continue

            per_span.setdefault(span_id, []).append(level)

        result: dict[int, str | None] = {}
        for i in range(expected_count):
            if i not in per_span:
                logger.debug(f"Span {i} not in batch output, treating as unknown")
            # Duplicate judgments follow the same reduction as nested labels,
            # regardless of whether the model used flat or nested output.
            result[i] = _reduce_support_levels(per_span.get(i, []))

        return result

    def _parse_necessity_output(
        self, output_text: str, spans: list[dict]
    ) -> dict[tuple[int, int], bool | None]:
        """Parse LLM output for citation necessity judgments.

        Args:
            output_text: LLM output text
            spans: Original span list

        Returns:
            Dictionary mapping every span (begin, end) to `True` (needs a
            citation), `False` (does not), or `None` (the judgment was
            unreadable or missing). Duplicate judgments merge as
            `True` over `None` over `False`.
        """
        logger.debug(f"Parsing necessity output: {output_text[:300]}")

        span_necessity: dict[tuple[int, int], bool | None] = {
            (span["begin"], span["end"]): None for span in spans
        }

        try:
            judgments = self._extract_and_repair_json_array(output_text)
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(
                f"Parse error in necessity output: {e}, treating all spans as unknown"
            )
            return span_necessity

        logger.debug(f"Parsed JSON judgments: {len(judgments)} judgments")

        judged: set[tuple[int, int]] = set()
        for judgment in judgments:
            if not isinstance(judgment, dict):
                continue

            needs_citation_raw = judgment.get("needs_citation")
            span_id = _normalise_span_id(judgment.get("span_id"), len(spans))
            needs_citation = _normalise_needs_citation(needs_citation_raw)

            logger.debug(
                f"  Judgment: span_id={span_id}, needs_citation_raw={needs_citation_raw!r}, "
                f"needs_citation={needs_citation}"
            )

            if span_id is None:
                continue

            span = spans[span_id]
            span_key = (span["begin"], span["end"])
            if span_key not in judged:
                judged.add(span_key)
                span_necessity[span_key] = needs_citation
            else:
                span_necessity[span_key] = _merge_necessity(
                    span_necessity[span_key], needs_citation
                )

        for span_key, value in span_necessity.items():
            if value is None:
                logger.debug(f"  Span {span_key} necessity unreadable or missing")

        return span_necessity

    def _build_groundedness_result(
        self,
        response: str,
        citations: list[dict],
        span_necessity: dict[tuple[int, int], bool | None],
        span_support: dict[tuple[int, int], str | None],
    ) -> ValidationResult:
        """Build final groundedness result.

        A span whose necessity is `True` fails when it is not supported (or only
        partially supported when partial support is not allowed), and is unknown
        when its support could not be read. A span whose necessity could not be
        read is ok when its support meets the acceptance policy (fully supported,
        or partially supported if partial support is allowed); otherwise it is unknown.
        Spans with necessity `False` are ignored. Any failing span returns an
        ordinary failure naming only the failing spans; otherwise any unknown
        span returns an errored result; otherwise the result passes.

        Args:
            response: The assistant response
            citations: Citation records
            span_necessity: Mapping of span to needs_citation flag (`None` = unread)
            span_support: Mapping of span to support level (`None` = unread)

        Returns:
            A passing result, an ordinary failure naming the failing spans, or an
            errored result (`reason=None`, `error` set) when a span could not be
            assessed.

        Raises:
            ValueError: If `span_support` has a key that is not in
                `span_necessity`, if a span with necessity `True` or `None` has no
                `span_support` entry, or if a `span_support` value for such a span
                is neither `None` nor a known support level.
        """

        def _describe(span_key: tuple[int, int]) -> str:
            begin, end = span_key
            return f"{span_key} {response[begin:end][:50]!r}"

        # Invariants: inconsistent inputs point at a bug, never at a default.
        for span_key in span_support:
            if span_key not in span_necessity:
                raise ValueError(
                    f"Support recorded for span {span_key} which has no necessity entry"
                )
        for span_key, needs_citation in span_necessity.items():
            if needs_citation is False:
                continue
            if span_key not in span_support:
                raise ValueError(
                    f"Missing support entry for span {_describe(span_key)}"
                )
            value = span_support[span_key]
            if value is not None and value not in _SUPPORT_LEVEL_RANK:
                raise ValueError(
                    f"Unrecognised support value {value!r} for span {_describe(span_key)}"
                )

        failing: list[tuple[int, int]] = []
        unknown: list[tuple[int, int]] = []
        for span_key, needs_citation in span_necessity.items():
            if needs_citation is False:
                continue
            support = span_support[span_key]
            if needs_citation is True:
                if support is None:
                    unknown.append(span_key)
                elif support == "NOT_SUPPORTED" or (
                    support == "PARTIALLY_SUPPORTED" and not self.allow_partial_support
                ):
                    failing.append(span_key)
            elif support == "FULLY_SUPPORTED" or (
                support == "PARTIALLY_SUPPORTED" and self.allow_partial_support
            ):
                pass
            else:
                unknown.append(span_key)

        if not failing and unknown:
            shown = "; ".join(_describe(k) for k in unknown[:5])
            more = f" (and {len(unknown) - 5} more)" if len(unknown) > 5 else ""
            return ValidationResult(
                False,
                reason=None,
                error=ValueError(
                    f"Could not determine groundedness for {len(unknown)} span(s): "
                    f"{shown}{more}"
                ),
            )

        # Summary counts fully-supported spans, not spans that passed the
        # configured policy: a partially supported span passes when
        # allow_partial_support=True but is not in the numerator.
        total_spans_needing_citations = sum(
            1 for needs in span_necessity.values() if needs is True
        )
        fully_supported = sum(
            1
            for span_key, needs in span_necessity.items()
            if needs is True and span_support.get(span_key) == "FULLY_SUPPORTED"
        )
        summary = (
            f"\nSummary: {fully_supported}/{total_spans_needing_citations} "
            f"spans needing citations are fully supported."
        )

        if failing:
            reason = "Response is not grounded - the following spans are not properly supported:\n\n"
            for begin, end in failing:
                span_text = response[begin:end]
                reason += f'- "{span_text}" [{span_support[(begin, end)]}]\n'
            return ValidationResult(False, reason=reason + summary)

        return ValidationResult(
            True, reason="Response is grounded in citations." + summary
        )
