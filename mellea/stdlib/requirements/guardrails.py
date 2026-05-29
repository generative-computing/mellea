"""Pre-built, reusable requirements for common guardrail patterns.

This module provides a library of ready-to-use guardrails that integrations can use
out of the box for common validation patterns like PII detection, format compliance,
length constraints, and more.

Examples:
    >>> from mellea.stdlib.requirements.guardrails import no_pii, json_valid, max_length
    >>> from mellea.stdlib.session import start_session
    >>>
    >>> m = start_session()
    >>> result = m.instruct(
    ...     "Generate a customer profile",
    ...     requirements=[no_pii(), json_valid(), max_length(500)]
    ... )
"""

from __future__ import annotations

import re
from typing import Literal

from ...core import Context, FancyLogger, Requirement, ValidationResult
from .requirement import simple_validate

logger = FancyLogger.get_logger()

# region PII Detection

# Common PII regex patterns
_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
_PHONE_PATTERN = re.compile(
    r"\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b"
)
_SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CREDIT_CARD_PATTERN = re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b")

# Lazy-loaded spaCy model
_SPACY_NLP = None

# Harmful content keywords for risk detection
_RISK_KEYWORDS = {
    "violence": [
        "kill",
        "murder",
        "attack",
        "assault",
        "weapon",
        "blood",
        "death",
        "violent",
    ],
    "profanity": ["fuck", "shit", "damn", "hell", "ass", "bitch", "bastard"],
    "social_bias": ["racist", "sexist", "discrimination", "prejudice", "stereotype"],
    "sexual_content": ["sex", "sexual", "porn", "nude", "explicit"],
    "unethical_behavior": ["cheat", "steal", "fraud", "illegal", "unethical"],
    "jailbreak": ["ignore previous", "disregard", "bypass", "override"],
    "harm": ["harm", "hurt", "damage", "injure", "dangerous"],
}


def _get_spacy_nlp():
    """Lazy load spaCy model."""
    global _SPACY_NLP
    if _SPACY_NLP is None:
        try:
            import spacy  # type: ignore[import-not-found]

            try:
                _SPACY_NLP = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning(
                    "spaCy model 'en_core_web_sm' not found. "
                    "Install with: python -m spacy download en_core_web_sm"
                )
                _SPACY_NLP = False  # Mark as unavailable
        except ImportError:
            logger.warning(
                "spaCy not installed. Install with: pip install spacy\n"
                "For better PII detection, also run: python -m spacy download en_core_web_sm"
            )
            _SPACY_NLP = False
    return _SPACY_NLP if _SPACY_NLP is not False else None


def _detect_pii_regex(text: str) -> tuple[bool, list[str]]:
    """Detect PII using regex patterns.

    Args:
        text: Text to check for PII

    Returns:
        Tuple of (has_pii, list of detected PII types)
    """
    detected = []

    if _EMAIL_PATTERN.search(text):
        detected.append("email")
    if _PHONE_PATTERN.search(text):
        detected.append("phone")
    if _SSN_PATTERN.search(text):
        detected.append("SSN")
    if _CREDIT_CARD_PATTERN.search(text):
        detected.append("credit_card")

    return len(detected) > 0, detected


def _detect_pii_spacy(text: str) -> tuple[bool, list[str]]:
    """Detect PII using spaCy NER.

    Args:
        text: Text to check for PII

    Returns:
        Tuple of (has_pii, list of detected entity types)
    """
    nlp = _get_spacy_nlp()
    if nlp is None:
        return False, []

    doc = nlp(text)
    pii_entities = ["PERSON", "GPE", "LOC", "ORG"]
    detected = []

    for ent in doc.ents:
        if ent.label_ in pii_entities and ent.label_ not in detected:
            detected.append(ent.label_)

    return len(detected) > 0, detected


def _validate_no_pii(
    ctx: Context,
    method: Literal["regex", "spacy", "auto"],
    strict: bool,
    check_only: bool,
) -> ValidationResult:
    """Validation function for no_pii guardrail.

    Args:
        ctx: Context to validate
        method: Detection method to use
        strict: If True, use LLM for final verification
        check_only: If True, provide brief reason; if False, provide actionable repair guidance

    Returns:
        ValidationResult indicating if PII was detected
    """
    last_output = ctx.last_output()
    if last_output is None or last_output.value is None:
        return ValidationResult(result=False, reason="No output found in context")

    text = str(last_output.value)
    has_pii = False
    detected_types: list[str] = []
    pii_examples = []

    # Apply detection method and collect examples
    if method == "regex":
        has_pii, detected_types = _detect_pii_regex(text)
        # Collect examples for repair guidance
        if has_pii and not check_only:
            if "email" in detected_types:
                match = _EMAIL_PATTERN.search(text)
                if match:
                    pii_examples.append(f"email address '{match.group()}'")
            if "phone" in detected_types:
                match = _PHONE_PATTERN.search(text)
                if match:
                    pii_examples.append(f"phone number '{match.group()}'")
            if "SSN" in detected_types:
                match = _SSN_PATTERN.search(text)
                if match:
                    pii_examples.append(f"SSN '{match.group()}'")
            if "credit_card" in detected_types:
                match = _CREDIT_CARD_PATTERN.search(text)
                if match:
                    pii_examples.append(f"credit card number '{match.group()}'")
    elif method == "spacy":
        has_pii, detected_types = _detect_pii_spacy(text)
        if not has_pii:  # Fallback to regex if spaCy finds nothing
            has_pii, regex_types = _detect_pii_regex(text)
            detected_types.extend(regex_types)
    else:  # auto
        # Try spaCy first, fallback to regex
        has_pii_spacy, spacy_types = _detect_pii_spacy(text)
        has_pii_regex, regex_types = _detect_pii_regex(text)

        has_pii = has_pii_spacy or has_pii_regex
        detected_types = list(set(spacy_types + regex_types))

    # Build reason message
    if has_pii:
        if check_only:
            reason = f"Detected potential PII: {', '.join(detected_types)}"
        else:
            # Provide actionable repair guidance
            if pii_examples:
                reason = (
                    f"Output contains personally identifiable information (PII): {', '.join(pii_examples)}. "
                    f"Please remove or redact this sensitive information to protect privacy."
                )
            else:
                reason = (
                    f"Output contains PII of type(s): {', '.join(detected_types)}. "
                    f"Please remove or redact all personally identifiable information including "
                    f"names, email addresses, phone numbers, and other sensitive data."
                )
    else:
        reason = "No PII detected"

    # Note: strict mode with LLM verification can be added in future iteration
    if strict and has_pii:
        reason += " (strict mode: consider LLM verification)"

    return ValidationResult(result=not has_pii, reason=reason)


def no_pii(
    *,
    method: Literal["regex", "spacy", "auto"] = "auto",
    strict: bool = False,
    check_only: bool = True,
) -> Requirement:
    """Reject outputs containing personally identifiable information (PII).

    This guardrail detects common PII patterns including:
    - Names (via spaCy NER)
    - Email addresses
    - Phone numbers
    - Social Security Numbers
    - Credit card numbers
    - Organizations and locations (via spaCy NER)

    The detection uses a hybrid approach:
    - **regex**: Fast pattern matching for emails, phones, SSNs, credit cards
    - **spacy**: NER-based detection for names, organizations, locations (requires spacy)
    - **auto** (default): Try spaCy first, fallback to regex

    Args:
        method: Detection method - "regex", "spacy", or "auto" (default)
        strict: If True, be more conservative in PII detection (future: LLM verification)
        check_only: If True, only validate without attempting repair (default: True)

    Returns:
        Requirement that validates output contains no PII

    Examples:
        Basic usage:
        >>> from mellea.stdlib.requirements.guardrails import no_pii
        >>> req = no_pii()

        Regex-only (no dependencies):
        >>> req = no_pii(method="regex")

        Strict mode:
        >>> req = no_pii(strict=True)

        In a session:
        >>> from mellea.stdlib.session import start_session
        >>> m = start_session()
        >>> result = m.instruct(
        ...     "Describe a customer without revealing personal details",
        ...     requirements=[no_pii()]
        ... )

    Note:
        - For best results, install spaCy: `pip install spacy`
        - Download model: `python -m spacy download en_core_web_sm`
        - Regex-only mode works without additional dependencies
        - False positives may occur (e.g., fictional names in creative writing)
    """
    return Requirement(
        description="Output must not contain personally identifiable information (PII) "
        "such as names, email addresses, phone numbers, or other sensitive data.",
        validation_fn=lambda ctx: _validate_no_pii(ctx, method, strict, check_only),
        check_only=check_only,
    )


# endregion

# region JSON Validation


def json_valid(*, check_only: bool = True) -> Requirement:
    """Validate that output is valid JSON format.

    This guardrail ensures the generated output can be parsed as valid JSON.
    Useful for ensuring structured data outputs.

    Args:
        check_only: If True, only validate without attempting repair (default: True)

    Returns:
        Requirement that validates output is valid JSON

    Examples:
        Basic usage:
        >>> from mellea.stdlib.requirements.guardrails import json_valid
        >>> req = json_valid()

        In a session:
        >>> from mellea.stdlib.session import start_session
        >>> m = start_session()
        >>> result = m.instruct(
        ...     "Generate a JSON object with name and age fields",
        ...     requirements=[json_valid()]
        ... )
    """
    import json

    def validate_json(ctx: Context) -> ValidationResult:
        last_output = ctx.last_output()
        if last_output is None or last_output.value is None:
            return ValidationResult(result=False, reason="No output found in context")

        text = str(last_output.value).strip()

        try:
            json.loads(text)
            return ValidationResult(result=True, reason="Valid JSON")
        except json.JSONDecodeError as e:
            if check_only:
                reason = f"Invalid JSON: {e.msg} at line {e.lineno}, column {e.colno}"
            else:
                reason = (
                    f"Output is not valid JSON. Error: {e.msg} at line {e.lineno}, column {e.colno}. "
                    f"Please ensure the output is properly formatted JSON with correct syntax, "
                    f"including proper quotes, commas, and bracket matching."
                )
            return ValidationResult(result=False, reason=reason)

    return Requirement(
        description="Output must be valid JSON format.",
        validation_fn=validate_json,
        check_only=check_only,
    )


# endregion

# region Length Constraints


def max_length(
    n: int, *, unit: str = "characters", check_only: bool = True
) -> Requirement:
    """Enforce maximum length constraint on output.

    Args:
        n: Maximum allowed length
        unit: Unit of measurement - "characters", "words", or "tokens" (default: "characters")
        check_only: If True, only validate without attempting repair (default: True)

    Returns:
        Requirement that validates output length

    Examples:
        Character limit:
        >>> from mellea.stdlib.requirements.guardrails import max_length
        >>> req = max_length(500)

        Word limit:
        >>> req = max_length(100, unit="words")

        In a session:
        >>> from mellea.stdlib.session import start_session
        >>> m = start_session()
        >>> result = m.instruct(
        ...     "Write a brief summary",
        ...     requirements=[max_length(200)]
        ... )
    """

    def validate_max_length(ctx: Context) -> ValidationResult:
        last_output = ctx.last_output()
        if last_output is None or last_output.value is None:
            return ValidationResult(result=False, reason="No output found in context")

        text = str(last_output.value)

        if unit == "characters":
            length = len(text)
        elif unit == "words":
            length = len(text.split())
        elif unit == "tokens":
            # Simple approximation: ~4 characters per token
            length = len(text) // 4
        else:
            return ValidationResult(
                result=False,
                reason=f"Invalid unit '{unit}'. Use 'characters', 'words', or 'tokens'.",
            )

        if length <= n:
            return ValidationResult(
                result=True, reason=f"Length {length} {unit} is within limit of {n}"
            )
        else:
            if check_only:
                reason = f"Length {length} {unit} exceeds maximum of {n}"
            else:
                excess = length - n
                reason = (
                    f"Output exceeds maximum length of {n} {unit}. "
                    f"Current length: {length} {unit} (exceeds by {excess} {unit}). "
                    f"Please shorten the output by removing unnecessary content or being more concise."
                )
            return ValidationResult(result=False, reason=reason)

    return Requirement(
        description=f"Output must not exceed {n} {unit}.",
        validation_fn=validate_max_length,
        check_only=check_only,
    )


def min_length(
    n: int, *, unit: str = "characters", check_only: bool = True
) -> Requirement:
    """Enforce minimum length constraint on output.

    Args:
        n: Minimum required length
        unit: Unit of measurement - "characters", "words", or "tokens" (default: "characters")
        check_only: If True, only validate without attempting repair (default: True)

    Returns:
        Requirement that validates output length

    Examples:
        Character minimum:
        >>> from mellea.stdlib.requirements.guardrails import min_length
        >>> req = min_length(100)

        Word minimum:
        >>> req = min_length(50, unit="words")

        In a session:
        >>> from mellea.stdlib.session import start_session
        >>> m = start_session()
        >>> result = m.instruct(
        ...     "Write a detailed explanation",
        ...     requirements=[min_length(500)]
        ... )
    """

    def validate_min_length(ctx: Context) -> ValidationResult:
        last_output = ctx.last_output()
        if last_output is None or last_output.value is None:
            return ValidationResult(result=False, reason="No output found in context")

        text = str(last_output.value)

        if unit == "characters":
            length = len(text)
        elif unit == "words":
            length = len(text.split())
        elif unit == "tokens":
            # Simple approximation: ~4 characters per token
            length = len(text) // 4
        else:
            return ValidationResult(
                result=False,
                reason=f"Invalid unit '{unit}'. Use 'characters', 'words', or 'tokens'.",
            )

        if length >= n:
            return ValidationResult(
                result=True, reason=f"Length {length} {unit} meets minimum of {n}"
            )
        else:
            if check_only:
                reason = f"Length {length} {unit} is below minimum of {n}"
            else:
                shortage = n - length
                reason = (
                    f"Output is below minimum length of {n} {unit}. "
                    f"Current length: {length} {unit} (short by {shortage} {unit}). "
                    f"Please expand the output with more detail, examples, or explanation."
                )
            return ValidationResult(result=False, reason=reason)

    return Requirement(
        description=f"Output must be at least {n} {unit}.",
        validation_fn=validate_min_length,
        check_only=check_only,
    )


# endregion

# region Keyword Matching


def contains_keywords(
    keywords: list[str],
    *,
    case_sensitive: bool = False,
    require_all: bool = False,
    check_only: bool = True,
) -> Requirement:
    """Require output to contain specific keywords.

    Args:
        keywords: List of keywords that should appear in output
        case_sensitive: If True, perform case-sensitive matching (default: False)
        require_all: If True, all keywords must be present; if False, at least one (default: False)
        check_only: If True, only validate without attempting repair (default: True)

    Returns:
        Requirement that validates keyword presence

    Examples:
        Require any keyword:
        >>> from mellea.stdlib.requirements.guardrails import contains_keywords
        >>> req = contains_keywords(["Python", "Java", "JavaScript"])

        Require all keywords:
        >>> req = contains_keywords(["API", "REST", "JSON"], require_all=True)

        Case-sensitive:
        >>> req = contains_keywords(["NASA", "SpaceX"], case_sensitive=True)

        In a session:
        >>> from mellea.stdlib.session import start_session
        >>> m = start_session()
        >>> result = m.instruct(
        ...     "Explain web development",
        ...     requirements=[contains_keywords(["HTML", "CSS", "JavaScript"])]
        ... )
    """

    def validate_keywords(ctx: Context) -> ValidationResult:
        last_output = ctx.last_output()
        if last_output is None or last_output.value is None:
            return ValidationResult(result=False, reason="No output found in context")

        text = str(last_output.value)
        if not case_sensitive:
            text = text.lower()
            keywords_to_check = [k.lower() for k in keywords]
        else:
            keywords_to_check = keywords

        found_keywords = [kw for kw in keywords_to_check if kw in text]

        if require_all:
            if len(found_keywords) == len(keywords):
                return ValidationResult(
                    result=True,
                    reason=f"All required keywords found: {', '.join(keywords)}",
                )
            else:
                missing = [
                    kw
                    for kw in keywords
                    if (kw.lower() if not case_sensitive else kw)
                    not in list(found_keywords)
                ]
                return ValidationResult(
                    result=False,
                    reason=f"Missing required keywords: {', '.join(missing)}",
                )
        else:
            if len(found_keywords) > 0:
                return ValidationResult(
                    result=True,
                    reason=f"Found keywords: {', '.join([keywords[keywords_to_check.index(fk)] for fk in found_keywords])}",
                )
            else:
                return ValidationResult(
                    result=False,
                    reason=f"None of the required keywords found: {', '.join(keywords)}",
                )

    mode = "all" if require_all else "any"
    return Requirement(
        description=f"Output must contain {mode} of these keywords: {', '.join(keywords)}.",
        validation_fn=validate_keywords,
        check_only=check_only,
    )


def excludes_keywords(
    keywords: list[str], *, case_sensitive: bool = False, check_only: bool = True
) -> Requirement:
    """Require output to NOT contain specific keywords.

    Args:
        keywords: List of keywords that should NOT appear in output
        case_sensitive: If True, perform case-sensitive matching (default: False)
        check_only: If True, only validate without attempting repair (default: True)

    Returns:
        Requirement that validates keyword absence

    Examples:
        Exclude specific terms:
        >>> from mellea.stdlib.requirements.guardrails import excludes_keywords
        >>> req = excludes_keywords(["TODO", "FIXME", "XXX"])

        Case-sensitive exclusion:
        >>> req = excludes_keywords(["CONFIDENTIAL"], case_sensitive=True)

        In a session:
        >>> from mellea.stdlib.session import start_session
        >>> m = start_session()
        >>> result = m.instruct(
        ...     "Write professional documentation",
        ...     requirements=[excludes_keywords(["slang", "informal"])]
        ... )
    """

    def validate_exclusions(ctx: Context) -> ValidationResult:
        last_output = ctx.last_output()
        if last_output is None or last_output.value is None:
            return ValidationResult(result=False, reason="No output found in context")

        text = str(last_output.value)
        if not case_sensitive:
            text = text.lower()
            keywords_to_check = [k.lower() for k in keywords]
        else:
            keywords_to_check = keywords

        found_keywords = [kw for kw in keywords_to_check if kw in text]

        if len(found_keywords) == 0:
            return ValidationResult(result=True, reason="No forbidden keywords found")
        else:
            # Map back to original case for reporting
            original_found = [
                keywords[keywords_to_check.index(fk)] for fk in found_keywords
            ]
            if check_only:
                reason = f"Found forbidden keywords: {', '.join(original_found)}"
            else:
                reason = (
                    f"Output contains forbidden keywords: {', '.join(original_found)}. "
                    f"Please remove or rephrase to avoid these terms."
                )
            return ValidationResult(result=False, reason=reason)

    return Requirement(
        description=f"Output must not contain these keywords: {', '.join(keywords)}.",
        validation_fn=validate_exclusions,
        check_only=check_only,
    )


# endregion

# region Harmful Content Detection


def no_harmful_content(
    *, risk_types: list[str] | None = None, check_only: bool = True
) -> Requirement:
    """Detect harmful content using Guardian risk detection.

    This guardrail uses Guardian models to detect various types of harmful content
    including violence, profanity, social bias, sexual content, and unethical behavior.

    Available risk types:
    - "harm": General harmful content
    - "violence": Violent content
    - "profanity": Profane language
    - "social_bias": Social bias and discrimination
    - "sexual_content": Sexual or adult content
    - "unethical_behavior": Unethical behavior
    - "jailbreak": Jailbreak attempts

    Args:
        risk_types: List of specific risk types to check. If None, checks for general harm.
        check_only: If True, only validate without attempting repair (default: True)

    Returns:
        Requirement that validates output contains no harmful content

    Examples:
        Check for general harm:
        >>> from mellea.stdlib.requirements.guardrails import no_harmful_content
        >>> req = no_harmful_content()

        Check specific risk types:
        >>> req = no_harmful_content(risk_types=["violence", "profanity"])

        In a session:
        >>> from mellea.stdlib.session import start_session
        >>> m = start_session()
        >>> result = m.instruct(
        ...     "Write a story about conflict resolution",
        ...     requirements=[no_harmful_content()]
        ... )

    Note:
        This is a lightweight implementation that uses keyword-based detection.
        For production use with Guardian models, use the Guardian intrinsics directly
        or the deprecated GuardianCheck class with appropriate backends.
    """

    def validate_no_harmful_content(ctx: Context) -> ValidationResult:
        last_output = ctx.last_output()
        if last_output is None or last_output.value is None:
            return ValidationResult(result=False, reason="No output found in context")

        text = str(last_output.value).lower()

        # Determine which risk types to check
        risks_to_check = risk_types if risk_types else ["harm"]

        # Check for harmful keywords
        detected_risks = []
        for risk in risks_to_check:
            if risk not in _RISK_KEYWORDS:
                logger.warning(f"Unknown risk type: {risk}. Skipping.")
                continue

            keywords = _RISK_KEYWORDS[risk]
            for keyword in keywords:
                if keyword in text:
                    detected_risks.append(risk)
                    break

        if detected_risks:
            if check_only:
                reason = f"Detected potentially harmful content: {', '.join(set(detected_risks))}"
            else:
                reason = (
                    f"Output contains potentially harmful content related to: {', '.join(set(detected_risks))}. "
                    f"Please revise to remove harmful, offensive, or inappropriate content."
                )
            return ValidationResult(result=False, reason=reason)
        else:
            return ValidationResult(result=True, reason="No harmful content detected")

    risk_desc = ", ".join(risk_types) if risk_types else "harmful content"
    return Requirement(
        description=f"Output must not contain {risk_desc}.",
        validation_fn=validate_no_harmful_content,
        check_only=check_only,
    )


# endregion

# region JSON Schema Validation


def matches_schema(schema: dict, *, check_only: bool = True) -> Requirement:
    """Validate JSON output against a JSON schema.

    This guardrail validates that the output conforms to a JSON Schema (Draft 7).
    Requires the jsonschema library to be installed.

    Args:
        schema: JSON schema dictionary (JSON Schema Draft 7 format)
        check_only: If True, only validate without attempting repair (default: True)

    Returns:
        Requirement that validates output matches the schema

    Examples:
        Basic schema validation:
        >>> from mellea.stdlib.requirements.guardrails import matches_schema
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {
        ...         "name": {"type": "string"},
        ...         "age": {"type": "number", "minimum": 0}
        ...     },
        ...     "required": ["name", "age"]
        ... }
        >>> req = matches_schema(schema)

        Array validation:
        >>> schema = {
        ...     "type": "array",
        ...     "items": {"type": "string"},
        ...     "minItems": 1
        ... }
        >>> req = matches_schema(schema)

        In a session:
        >>> from mellea.stdlib.session import start_session
        >>> m = start_session()
        >>> result = m.instruct(
        ...     "Generate a person object with name and age",
        ...     requirements=[matches_schema(schema)]
        ... )

    Note:
        Requires jsonschema library. Install with: pip install jsonschema
        or: pip install mellea[schema]
    """
    import json

    def validate_schema(ctx: Context) -> ValidationResult:
        last_output = ctx.last_output()
        if last_output is None or last_output.value is None:
            return ValidationResult(result=False, reason="No output found in context")

        text = str(last_output.value).strip()

        # First, validate it's valid JSON
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            return ValidationResult(
                result=False,
                reason=f"Invalid JSON: {e.msg} at line {e.lineno}, column {e.colno}",
            )

        # Try to import jsonschema
        try:
            import jsonschema
        except ImportError:
            return ValidationResult(
                result=False,
                reason="jsonschema library not installed. Install with: pip install jsonschema",
            )

        # Validate against schema
        try:
            jsonschema.validate(instance=data, schema=schema)
            return ValidationResult(result=True, reason="Output matches schema")
        except jsonschema.ValidationError as e:
            if check_only:
                reason = f"Schema validation failed: {e.message}"
            else:
                reason = (
                    f"Output does not match the required JSON schema. "
                    f"Validation error: {e.message}. "
                    f"Please ensure the JSON structure matches the specified schema requirements."
                )
            return ValidationResult(result=False, reason=reason)
        except jsonschema.SchemaError as e:
            return ValidationResult(result=False, reason=f"Invalid schema: {e.message}")

    return Requirement(
        description="Output must match the provided JSON schema.",
        validation_fn=validate_schema,
        check_only=check_only,
    )


# endregion

# region Code Validation


def is_code(language: str | None = None, *, check_only: bool = True) -> Requirement:
    """Validate that output is valid code in the specified language.

    This guardrail validates code syntax using language-specific parsers or heuristics.

    Supported languages:
    - "python": Uses ast.parse() for syntax validation
    - "javascript", "typescript": Heuristic detection (function, const, let, var, =>)
    - "java", "c", "cpp": Heuristic detection (class, public, void, int)
    - None: Generic code detection using multiple heuristics

    Args:
        language: Programming language to validate (python, javascript, java, etc.)
                 If None, performs generic code detection
        check_only: If True, only validate without attempting repair (default: True)

    Returns:
        Requirement that validates output is valid code

    Examples:
        Python syntax validation:
        >>> from mellea.stdlib.requirements.guardrails import is_code
        >>> req = is_code("python")

        Generic code detection:
        >>> req = is_code()

        JavaScript validation:
        >>> req = is_code("javascript")

        In a session:
        >>> from mellea.stdlib.session import start_session
        >>> m = start_session()
        >>> result = m.instruct(
        ...     "Write a Python function to calculate factorial",
        ...     requirements=[is_code("python")]
        ... )

    Note:
        - Python validation uses ast.parse() for accurate syntax checking
        - Other languages use heuristic detection (may have false positives/negatives)
        - Generic detection checks for common code patterns
    """

    def validate_code(ctx: Context) -> ValidationResult:
        last_output = ctx.last_output()
        if last_output is None or last_output.value is None:
            return ValidationResult(result=False, reason="No output found in context")

        text = str(last_output.value).strip()

        if not text:
            return ValidationResult(result=False, reason="Empty output")

        # Python: Use ast.parse for accurate syntax checking
        if language and language.lower() == "python":
            import ast

            try:
                ast.parse(text)
                return ValidationResult(result=True, reason="Valid Python syntax")
            except SyntaxError as e:
                if check_only:
                    reason = f"Invalid Python syntax: {e.msg} at line {e.lineno}"
                else:
                    reason = (
                        f"Output is not valid Python code. "
                        f"Syntax error: {e.msg} at line {e.lineno}. "
                        f"Please provide syntactically correct Python code."
                    )
                return ValidationResult(result=False, reason=reason)

        # For other languages, use heuristic detection
        lang_lower = language.lower() if language else None

        # Check balanced braces/brackets/parentheses
        def check_balanced(text: str) -> bool:
            stack = []
            pairs = {"(": ")", "[": "]", "{": "}"}
            for char in text:
                if char in pairs:
                    stack.append(char)
                elif char in pairs.values():
                    if not stack:
                        return False
                    if pairs[stack.pop()] != char:
                        return False
            return len(stack) == 0

        if not check_balanced(text):
            if check_only:
                reason = "Unbalanced braces, brackets, or parentheses"
            else:
                reason = (
                    "Code has unbalanced braces, brackets, or parentheses. "
                    "Please ensure all opening symbols have matching closing symbols."
                )
            return ValidationResult(result=False, reason=reason)

        # Language-specific heuristics
        if lang_lower in ["javascript", "typescript", "js", "ts"]:
            # JavaScript/TypeScript patterns
            js_patterns = [
                r"\bfunction\s+\w+\s*\(",
                r"\bconst\s+\w+",
                r"\blet\s+\w+",
                r"\bvar\s+\w+",
                r"=>",
                r"\bclass\s+\w+",
            ]
            matches = sum(1 for pattern in js_patterns if re.search(pattern, text))
            if matches >= 2:
                return ValidationResult(
                    result=True, reason=f"Valid {language} code detected"
                )
            else:
                if check_only:
                    reason = f"Does not appear to be valid {language} code"
                else:
                    reason = (
                        f"Output does not appear to be valid {language} code. "
                        f"Please provide proper {language} code with appropriate syntax and structure."
                    )
                return ValidationResult(result=False, reason=reason)

        elif lang_lower in ["java", "c", "cpp", "c++"]:
            # Java/C/C++ patterns
            c_patterns = [
                r"\b(public|private|protected)\s+",
                r"\bclass\s+\w+",
                r"\b(void|int|float|double|char|bool)\s+\w+\s*\(",
                r"\breturn\s+",
                r";",
            ]
            matches = sum(1 for pattern in c_patterns if re.search(pattern, text))
            if matches >= 2:
                return ValidationResult(
                    result=True, reason=f"Valid {language} code detected"
                )
            else:
                if check_only:
                    reason = f"Does not appear to be valid {language} code"
                else:
                    reason = (
                        f"Output does not appear to be valid {language} code. "
                        f"Please provide proper {language} code with appropriate syntax and structure."
                    )
                return ValidationResult(result=False, reason=reason)

        # Generic code detection (no specific language)
        else:
            code_indicators = 0

            # Check for function definitions
            if re.search(r"\b(function|def|func|fn)\s+\w+\s*\(", text):
                code_indicators += 1

            # Check for control flow
            if re.search(r"\b(if|else|for|while|switch|case)\b", text):
                code_indicators += 1

            # Check for variable declarations
            if re.search(r"\b(var|let|const|int|string|float|double)\s+\w+", text):
                code_indicators += 1

            # Check for operators
            if re.search(r"[=+\-*/]{1,2}", text):
                code_indicators += 1

            # Check for function calls
            if re.search(r"\w+\s*\([^)]*\)", text):
                code_indicators += 1

            # Check for semicolons or significant indentation
            if ";" in text or re.search(r"\n\s{4,}", text):
                code_indicators += 1

            # Threshold: at least 3 indicators for generic code
            if code_indicators >= 3:
                return ValidationResult(
                    result=True, reason=f"Code detected ({code_indicators} indicators)"
                )
            else:
                return ValidationResult(
                    result=False,
                    reason=f"Does not appear to be code ({code_indicators} indicators, need 3+)",
                )

    lang_desc = f"{language} code" if language else "code"
    return Requirement(
        description=f"Output must be valid {lang_desc}.",
        validation_fn=validate_code,
        check_only=check_only,
    )


# endregion

# region Factual Grounding


def factual_grounding(
    context: str, *, threshold: float = 0.5, check_only: bool = True
) -> Requirement:
    """Validate that output is grounded in the provided context.

    This guardrail checks that the generated output is factually grounded in the
    provided reference context. The basic implementation uses keyword overlap;
    for production use, consider using NLI models or Guardian intrinsics.

    Args:
        context: Reference context text for grounding validation
        threshold: Minimum overlap ratio (0.0-1.0) for validation (default: 0.5)
        check_only: If True, only validate without attempting repair (default: True)

    Returns:
        Requirement that validates output is grounded in context

    Examples:
        Basic grounding check:
        >>> from mellea.stdlib.requirements.guardrails import factual_grounding
        >>> context = "Python is a high-level programming language created by Guido van Rossum."
        >>> req = factual_grounding(context)

        Stricter threshold:
        >>> req = factual_grounding(context, threshold=0.7)

        In a session:
        >>> from mellea.stdlib.session import start_session
        >>> m = start_session()
        >>> context = "The company was founded in 2020 and has 50 employees."
        >>> result = m.instruct(
        ...     "Summarize the company information",
        ...     requirements=[factual_grounding(context)]
        ... )

    Note:
        This is a basic implementation using keyword overlap. For production use:
        - Use NLI (Natural Language Inference) models for semantic validation
        - Use Guardian intrinsics for hallucination detection
        - Consider using embedding-based similarity measures
    """
    # Simple stopwords list
    STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "that",
        "the",
        "to",
        "was",
        "will",
        "with",
        "the",
        "this",
        "but",
        "they",
        "have",
        "had",
        "what",
        "when",
        "where",
        "who",
        "which",
        "why",
        "how",
    }

    def extract_keywords(text: str) -> set[str]:
        """Extract keywords from text (remove stopwords and punctuation)."""
        # Convert to lowercase and split
        words = re.findall(r"\b\w+\b", text.lower())
        # Remove stopwords and short words
        keywords = {w for w in words if w not in STOPWORDS and len(w) > 2}
        return keywords

    def validate_grounding(ctx: Context) -> ValidationResult:
        last_output = ctx.last_output()
        if last_output is None or last_output.value is None:
            return ValidationResult(result=False, reason="No output found in context")

        output_text = str(last_output.value)

        # Extract keywords from both context and output
        context_keywords = extract_keywords(context)
        output_keywords = extract_keywords(output_text)

        if not output_keywords:
            return ValidationResult(
                result=False, reason="Output contains no meaningful keywords"
            )

        if not context_keywords:
            return ValidationResult(
                result=False, reason="Context contains no meaningful keywords"
            )

        # Calculate overlap ratio
        overlap = context_keywords.intersection(output_keywords)
        overlap_ratio = len(overlap) / len(output_keywords)

        if overlap_ratio >= threshold:
            return ValidationResult(
                result=True,
                reason=f"Output is grounded in context (overlap: {overlap_ratio:.2%})",
            )
        else:
            if check_only:
                reason = f"Output not sufficiently grounded (overlap: {overlap_ratio:.2%}, threshold: {threshold:.2%})"
            else:
                reason = (
                    f"Output contains claims not sufficiently supported by the provided context. "
                    f"Keyword overlap: {overlap_ratio:.2%} (threshold: {threshold:.2%}). "
                    f"Please ensure all claims are grounded in the given information and avoid adding unsupported facts."
                )
            return ValidationResult(result=False, reason=reason)

    return Requirement(
        description=f"Output must be grounded in the provided context (threshold: {threshold:.2%}).",
        validation_fn=validate_grounding,
        check_only=check_only,
    )


# endregion


# endregion
