"""Pre-built guardrail profiles for common use cases.

This module provides ready-to-use RequirementSet configurations for
common scenarios, making it easy to apply consistent guardrail policies
across an application.
"""

from __future__ import annotations

from .guardrails import (
    contains_keywords,
    excludes_keywords,
    factual_grounding,
    is_code,
    json_valid,
    max_length,
    min_length,
    no_harmful_content,
    no_pii,
)
from .requirement_set import RequirementSet


class GuardrailProfiles:
    """Pre-built requirement sets for common use cases.

    This class provides static methods that return RequirementSet instances
    configured for common scenarios. These profiles can be used as-is or
    customized by adding/removing requirements.

    Examples:
        Use a pre-built profile:
        >>> from mellea.stdlib.requirements import GuardrailProfiles
        >>> from mellea.stdlib.session import start_session
        >>>
        >>> m = start_session()
        >>> result = m.instruct(
        ...     "Generate Python code",
        ...     requirements=GuardrailProfiles.code_generation("python")
        ... )

        Customize a profile:
        >>> profile = GuardrailProfiles.basic_safety()
        >>> profile = profile.add(json_valid())
        >>> result = m.instruct("Generate data", requirements=profile)
    """

    @staticmethod
    def basic_safety() -> RequirementSet:
        """Basic safety guardrails: no PII, no harmful content.

        This is the minimum recommended set of guardrails for any
        user-facing content generation.

        Returns:
            RequirementSet with basic safety guardrails

        Examples:
            >>> profile = GuardrailProfiles.basic_safety()
            >>> result = m.instruct("Generate text", requirements=profile)
        """
        return RequirementSet([no_pii(), no_harmful_content()])

    @staticmethod
    def json_output(max_size: int = 1000) -> RequirementSet:
        """JSON output with validation and safety.

        Ensures output is valid JSON, contains no PII, and respects
        size constraints.

        Args:
            max_size: Maximum output size in characters (default: 1000)

        Returns:
            RequirementSet for JSON output

        Examples:
            >>> profile = GuardrailProfiles.json_output(max_size=500)
            >>> result = m.instruct("Generate JSON", requirements=profile)
        """
        return RequirementSet([json_valid(), max_length(max_size), no_pii()])

    @staticmethod
    def code_generation(language: str = "python") -> RequirementSet:
        """Code generation guardrails.

        Validates code syntax, ensures no harmful content, and excludes
        common placeholder markers.

        Args:
            language: Programming language for validation (default: "python")

        Returns:
            RequirementSet for code generation

        Examples:
            >>> profile = GuardrailProfiles.code_generation("javascript")
            >>> result = m.instruct("Generate code", requirements=profile)
        """
        return RequirementSet(
            [
                is_code(language),
                no_harmful_content(),
                excludes_keywords(["TODO", "FIXME", "XXX", "HACK"]),
            ]
        )

    @staticmethod
    def professional_content() -> RequirementSet:
        """Professional, safe content generation.

        Ensures content is appropriate for professional contexts:
        no PII, no profanity/violence, no placeholder text.

        Returns:
            RequirementSet for professional content

        Examples:
            >>> profile = GuardrailProfiles.professional_content()
            >>> result = m.instruct("Write article", requirements=profile)
        """
        return RequirementSet(
            [
                no_pii(),
                no_harmful_content(risk_types=["profanity", "violence"]),
                excludes_keywords(["TODO", "FIXME", "hack", "workaround", "temporary"]),
            ]
        )

    @staticmethod
    def api_documentation() -> RequirementSet:
        """API documentation guardrails.

        Ensures documentation is valid JSON, contains required keywords,
        excludes placeholders, and respects size limits.

        Returns:
            RequirementSet for API documentation

        Examples:
            >>> profile = GuardrailProfiles.api_documentation()
            >>> result = m.instruct("Document API", requirements=profile)
        """
        return RequirementSet(
            [
                json_valid(),
                no_pii(),
                contains_keywords(["endpoint", "method"], require_all=True),
                excludes_keywords(["TODO", "placeholder", "FIXME"]),
                max_length(2000),
            ]
        )

    @staticmethod
    def grounded_summary(context: str, threshold: float = 0.5) -> RequirementSet:
        """Factually grounded summary generation.

        Ensures summaries are grounded in provided context, contain no PII,
        and respect length constraints.

        Args:
            context: Reference context for grounding validation
            threshold: Minimum overlap ratio (0.0-1.0, default: 0.5)

        Returns:
            RequirementSet for grounded summaries

        Examples:
            >>> context = "Python is a programming language..."
            >>> profile = GuardrailProfiles.grounded_summary(context)
            >>> result = m.instruct("Summarize", requirements=profile)
        """
        return RequirementSet(
            [factual_grounding(context, threshold=threshold), no_pii(), max_length(500)]
        )

    @staticmethod
    def safe_chat() -> RequirementSet:
        """Safe chat/conversation guardrails.

        Appropriate for chatbot or conversational AI applications.
        Ensures safety without being overly restrictive.

        Returns:
            RequirementSet for safe chat

        Examples:
            >>> profile = GuardrailProfiles.safe_chat()
            >>> result = m.instruct("Respond to user", requirements=profile)
        """
        return RequirementSet([no_pii(), no_harmful_content(), max_length(1000)])

    @staticmethod
    def structured_data(
        schema: dict | None = None, max_size: int = 2000
    ) -> RequirementSet:
        """Structured data generation with optional schema validation.

        For generating structured data outputs. If a schema is provided,
        validates against it; otherwise just ensures valid JSON.

        Args:
            schema: Optional JSON schema for validation
            max_size: Maximum output size in characters (default: 2000)

        Returns:
            RequirementSet for structured data

        Examples:
            >>> schema = {"type": "object", "properties": {...}}
            >>> profile = GuardrailProfiles.structured_data(schema)
            >>> result = m.instruct("Generate data", requirements=profile)
        """
        reqs = RequirementSet([json_valid(), no_pii(), max_length(max_size)])

        if schema is not None:
            from .guardrails import matches_schema

            reqs = reqs.add(matches_schema(schema))

        return reqs

    @staticmethod
    def content_moderation() -> RequirementSet:
        """Content moderation guardrails.

        Strict guardrails for user-generated content or public-facing
        applications. Checks multiple risk types and excludes problematic
        keywords.

        Returns:
            RequirementSet for content moderation

        Examples:
            >>> profile = GuardrailProfiles.content_moderation()
            >>> result = m.instruct("Generate content", requirements=profile)
        """
        return RequirementSet(
            [
                no_pii(),
                no_harmful_content(
                    risk_types=[
                        "violence",
                        "profanity",
                        "social_bias",
                        "sexual_content",
                        "unethical_behavior",
                    ]
                ),
                excludes_keywords(
                    ["hate", "discrimination", "offensive", "inappropriate"]
                ),
            ]
        )

    @staticmethod
    def minimal() -> RequirementSet:
        """Minimal guardrails: just PII protection.

        Use when you need minimal constraints but still want basic
        privacy protection.

        Returns:
            RequirementSet with minimal guardrails

        Examples:
            >>> profile = GuardrailProfiles.minimal()
            >>> result = m.instruct("Generate text", requirements=profile)
        """
        return RequirementSet([no_pii()])

    @staticmethod
    def strict() -> RequirementSet:
        """Strict guardrails for high-risk applications.

        Comprehensive guardrails for applications where safety and
        compliance are critical.

        Returns:
            RequirementSet with strict guardrails

        Examples:
            >>> profile = GuardrailProfiles.strict()
            >>> result = m.instruct("Generate content", requirements=profile)
        """
        return RequirementSet(
            [
                no_pii(strict=True),
                no_harmful_content(
                    risk_types=[
                        "harm",
                        "violence",
                        "profanity",
                        "social_bias",
                        "sexual_content",
                        "unethical_behavior",
                        "jailbreak",
                    ]
                ),
                max_length(1000),
                excludes_keywords(
                    [
                        "TODO",
                        "FIXME",
                        "XXX",
                        "HACK",
                        "hate",
                        "discrimination",
                        "offensive",
                    ]
                ),
            ]
        )
