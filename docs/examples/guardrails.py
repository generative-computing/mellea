# pytest: ollama, llm
"""Comprehensive example demonstrating all guardrails in the Mellea library.

This example showcases all 10 pre-built guardrails:

Basic Guardrails:
- no_pii: PII detection (hybrid: spaCy + regex)
- json_valid: JSON format validation
- max_length/min_length: Length constraints
- contains_keywords/excludes_keywords: Keyword matching

Advanced Guardrails:
- no_harmful_content: Harmful content detection
- matches_schema: JSON schema validation
- is_code: Code syntax validation
- factual_grounding: Context grounding validation
"""

from mellea.stdlib.requirements.guardrails import (
    contains_keywords,
    excludes_keywords,
    factual_grounding,
    is_code,
    json_valid,
    matches_schema,
    max_length,
    min_length,
    no_harmful_content,
    no_pii,
)
from mellea.stdlib.session import start_session


# ============================================================================
# BASIC GUARDRAILS EXAMPLES
# ============================================================================


def example_no_pii_basic():
    """Basic example of PII detection with default settings."""
    print("\n=== PII Detection (Basic) ===")
    
    m = start_session()
    
    # This should pass - no PII in the output
    result = m.instruct(
        "Describe a typical software engineer's daily routine without mentioning specific people or companies.",
        requirements=[no_pii()]
    )
    print(f"Clean output: {result.value[:100] if result.value else 'None'}...")


def example_no_pii_modes():
    """Example showing different PII detection modes."""
    print("\n=== PII Detection (Different Modes) ===")
    
    m = start_session()
    
    # Regex-only (no dependencies)
    result = m.instruct(
        "Write a professional email template without any contact details.",
        requirements=[no_pii(method="regex")]
    )
    print(f"Regex-only: {result.value[:100] if result.value else 'None'}...")
    
    # Strict mode
    result = m.instruct(
        "Write a short story about a programmer, using only generic descriptions.",
        requirements=[no_pii(strict=True)]
    )
    print(f"Strict mode: {result.value[:100] if result.value else 'None'}...")


def example_json_validation():
    """Example of JSON format validation."""
    print("\n=== JSON Validation ===")
    
    m = start_session()
    
    result = m.instruct(
        "Generate a JSON object with fields: name (string), age (number), hobbies (array)",
        requirements=[json_valid()]
    )
    print(f"Valid JSON output: {result.value}")


def example_length_constraints():
    """Example of length constraints."""
    print("\n=== Length Constraints ===")
    
    m = start_session()
    
    # Maximum length
    result = m.instruct(
        "Write a one-sentence summary of Python",
        requirements=[max_length(100)]
    )
    print(f"Short summary ({len(result.value) if result.value else 0} chars): {result.value}")
    
    # Minimum length
    result = m.instruct(
        "Write a detailed explanation of REST APIs",
        requirements=[min_length(200)]
    )
    print(f"Detailed explanation ({len(result.value) if result.value else 0} chars): {result.value[:100] if result.value else 'None'}...")
    
    # Word-based constraints
    result = m.instruct(
        "List 5 programming languages",
        requirements=[max_length(50, unit="words")]
    )
    word_count = len(result.value.split()) if result.value else 0
    print(f"Word-limited output ({word_count} words): {result.value}")


def example_keyword_matching():
    """Example of keyword matching."""
    print("\n=== Keyword Matching ===")
    
    m = start_session()
    
    # Require specific keywords (any)
    result = m.instruct(
        "Explain web development technologies",
        requirements=[contains_keywords(["HTML", "CSS", "JavaScript"])]
    )
    print(f"Contains keywords: {result.value[:150] if result.value else 'None'}...")
    
    # Require ALL keywords
    result = m.instruct(
        "Describe a RESTful API",
        requirements=[contains_keywords(["HTTP", "JSON", "endpoint"], require_all=True)]
    )
    print(f"Contains all keywords: {result.value[:150] if result.value else 'None'}...")
    
    # Exclude keywords
    result = m.instruct(
        "Write professional documentation about software testing",
        requirements=[excludes_keywords(["TODO", "FIXME", "hack"])]
    )
    print(f"Professional output: {result.value[:150] if result.value else 'None'}...")


def example_case_sensitivity():
    """Example showing case sensitivity options."""
    print("\n=== Case Sensitivity ===")
    
    m = start_session()
    
    # Case-insensitive (default)
    result = m.instruct(
        "Explain python programming",
        requirements=[contains_keywords(["Python"], case_sensitive=False)]
    )
    print(f"Case-insensitive match: Success")
    
    # Case-sensitive
    result = m.instruct(
        "Explain the Python programming language",
        requirements=[contains_keywords(["Python"], case_sensitive=True)]
    )
    print(f"Case-sensitive match: Success")


# ============================================================================
# ADVANCED GUARDRAILS EXAMPLES
# ============================================================================


def example_harmful_content_detection():
    """Example of harmful content detection."""
    print("\n=== Harmful Content Detection ===")
    
    m = start_session()
    
    # Check for general harm
    result = m.instruct(
        "Write a helpful guide about online safety",
        requirements=[no_harmful_content()]
    )
    print(f"Safe content: {result.value[:100] if result.value else 'None'}...")
    
    # Check specific risk types
    result = m.instruct(
        "Write a professional article about conflict resolution",
        requirements=[no_harmful_content(risk_types=["violence", "profanity"])]
    )
    print(f"Professional content: {result.value[:100] if result.value else 'None'}...")


def example_schema_validation():
    """Example of JSON schema validation."""
    print("\n=== JSON Schema Validation ===")
    
    m = start_session()
    
    # Define a schema for a person object
    person_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number", "minimum": 0, "maximum": 150},
            "email": {"type": "string", "format": "email"},
            "skills": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1
            }
        },
        "required": ["name", "age"]
    }
    
    result = m.instruct(
        "Generate a JSON object for a software developer with name, age, email, and skills",
        requirements=[matches_schema(person_schema)]
    )
    print(f"Valid schema output: {result.value}")
    
    # Array schema
    array_schema = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 3,
        "maxItems": 10
    }
    
    result = m.instruct(
        "Generate a JSON array of 5 programming languages",
        requirements=[matches_schema(array_schema)]
    )
    print(f"Valid array: {result.value}")


def example_code_validation():
    """Example of code syntax validation."""
    print("\n=== Code Validation ===")
    
    m = start_session()
    
    # Python code validation
    result = m.instruct(
        "Write a Python function to calculate the factorial of a number",
        requirements=[is_code("python")]
    )
    print(f"Valid Python code:\n{result.value}\n")
    
    # JavaScript code validation
    result = m.instruct(
        "Write a JavaScript function to reverse a string",
        requirements=[is_code("javascript")]
    )
    print(f"Valid JavaScript code:\n{result.value}\n")
    
    # Generic code detection
    result = m.instruct(
        "Write a simple function in any language to add two numbers",
        requirements=[is_code()]
    )
    print(f"Generic code detected:\n{result.value}\n")


def example_factual_grounding():
    """Example of factual grounding validation."""
    print("\n=== Factual Grounding ===")
    
    m = start_session()
    
    # Provide context
    context = """
    Python is a high-level, interpreted programming language created by Guido van Rossum.
    It was first released in 1991. Python emphasizes code readability and uses significant
    indentation. It supports multiple programming paradigms including procedural, object-oriented,
    and functional programming.
    """
    
    # Generate grounded summary
    result = m.instruct(
        "Summarize the key facts about Python programming language",
        requirements=[factual_grounding(context, threshold=0.5)]
    )
    print(f"Grounded summary: {result.value}")
    
    # Stricter grounding
    result = m.instruct(
        "List the main characteristics of Python",
        requirements=[factual_grounding(context, threshold=0.3)]
    )
    print(f"Grounded characteristics: {result.value}")


# ============================================================================
# COMBINED EXAMPLES: Multiple Guardrails
# ============================================================================


def example_combined_basic():
    """Example combining multiple basic guardrails."""
    print("\n=== Combined Basic Guardrails ===")
    
    m = start_session()
    
    result = m.instruct(
        "Generate a JSON profile for a software developer role",
        requirements=[
            json_valid(),
            no_pii(),
            max_length(500),
            contains_keywords(["skills", "experience"]),
            excludes_keywords(["TODO", "placeholder"])
        ]
    )
    print(f"Combined validation result: {result.value}")


def example_combined_advanced():
    """Example combining multiple advanced guardrails."""
    print("\n=== Combined Advanced Guardrails ===")
    
    m = start_session()
    
    # Define schema for code snippet
    code_schema = {
        "type": "object",
        "properties": {
            "language": {"type": "string"},
            "code": {"type": "string"},
            "description": {"type": "string"}
        },
        "required": ["language", "code", "description"]
    }
    
    result = m.instruct(
        "Generate a JSON object with a Python code snippet that sorts a list",
        requirements=[
            matches_schema(code_schema),
            no_harmful_content()
        ]
    )
    print(f"Combined validation result: {result.value}")


def example_all_guardrails():
    """Example using all available guardrails."""
    print("\n=== All Guardrails Combined ===")
    
    m = start_session()
    
    # Schema for a code review
    review_schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "issues": {"type": "array", "items": {"type": "string"}},
            "rating": {"type": "number", "minimum": 1, "maximum": 10}
        },
        "required": ["summary", "issues", "rating"]
    }
    
    context = """
    Our codebase uses Python 3.11 with FastAPI.
    We follow PEP 8 style guidelines and use type hints.
    All functions must have docstrings.
    """
    
    result = m.instruct(
        "Provide a code review in JSON format",
        requirements=[
            json_valid(),
            matches_schema(review_schema),
            no_pii(),
            no_harmful_content(),
            max_length(1000),
            contains_keywords(["Python", "code"]),
            excludes_keywords(["TODO", "FIXME"]),
            factual_grounding(context, threshold=0.2)
        ]
    )
    print(f"Comprehensive validation result: {result.value}")


# ============================================================================
# REAL-WORLD USE CASES
# ============================================================================


def example_use_case_api_documentation():
    """Real-world use case: API documentation generator."""
    print("\n=== Use Case: API Documentation Generator ===")
    
    m = start_session()
    
    doc_schema = {
        "type": "object",
        "properties": {
            "endpoint": {"type": "string"},
            "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE"]},
            "description": {"type": "string"},
            "parameters": {"type": "array", "items": {"type": "object"}}
        },
        "required": ["endpoint", "method", "description"]
    }
    
    result = m.instruct(
        "Generate API documentation for a user registration endpoint",
        requirements=[
            json_valid(),
            matches_schema(doc_schema),
            no_pii(),
            contains_keywords(["endpoint", "method"], require_all=True),
            excludes_keywords(["TODO", "placeholder"]),
            max_length(800)
        ]
    )
    print(f"API Documentation: {result.value}")


def example_use_case_code_review():
    """Real-world use case: Automated code review assistant."""
    print("\n=== Use Case: Code Review Assistant ===")
    
    m = start_session()
    
    codebase_context = """
    Our application uses Python 3.11 with FastAPI for the backend.
    We follow PEP 8 style guidelines and use type hints.
    All functions must have docstrings.
    Security is a top priority.
    """
    
    review_schema = {
        "type": "object",
        "properties": {
            "issues": {"type": "array", "items": {"type": "string"}},
            "suggestions": {"type": "array", "items": {"type": "string"}},
            "rating": {"type": "number", "minimum": 1, "maximum": 10}
        },
        "required": ["issues", "suggestions", "rating"]
    }
    
    result = m.instruct(
        "Review this code and provide feedback in JSON format",
        requirements=[
            json_valid(),
            matches_schema(review_schema),
            factual_grounding(codebase_context, threshold=0.3),
            no_harmful_content(),
            no_pii(),
            contains_keywords(["Python", "code"]),
            min_length(100)
        ]
    )
    print(f"Code Review: {result.value}")


def example_use_case_content_moderation():
    """Real-world use case: Content moderation system."""
    print("\n=== Use Case: Content Moderation ===")
    
    m = start_session()
    
    result = m.instruct(
        "Generate a community guidelines summary for a professional forum",
        requirements=[
            no_harmful_content(risk_types=["violence", "profanity", "social_bias"]),
            no_pii(),
            max_length(500),
            contains_keywords(["respectful", "professional"]),
            excludes_keywords(["hate", "discrimination"])
        ]
    )
    print(f"Community Guidelines: {result.value}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================


if __name__ == "__main__":
    print("=" * 80)
    print("MELLEA GUARDRAILS COMPREHENSIVE EXAMPLES")
    print("=" * 80)
    
    # Basic Guardrails
    print("\n" + "=" * 80)
    print("BASIC GUARDRAILS")
    print("=" * 80)
    example_no_pii_basic()
    example_no_pii_modes()
    example_json_validation()
    example_length_constraints()
    example_keyword_matching()
    example_case_sensitivity()
    
    # Advanced Guardrails
    print("\n" + "=" * 80)
    print("ADVANCED GUARDRAILS")
    print("=" * 80)
    example_harmful_content_detection()
    example_schema_validation()
    example_code_validation()
    example_factual_grounding()
    
    # Combined Examples
    print("\n" + "=" * 80)
    print("COMBINED GUARDRAILS")
    print("=" * 80)
    example_combined_basic()
    example_combined_advanced()
    example_all_guardrails()
    
    # Real-World Use Cases
    print("\n" + "=" * 80)
    print("REAL-WORLD USE CASES")
    print("=" * 80)
    example_use_case_api_documentation()
    example_use_case_code_review()
    example_use_case_content_moderation()
    
    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETE")
    print("=" * 80)

