# pytest: skip_always
"""PII Purifier using Mellea Framework."""

import spacy

import mellea
from cli.serve.models import ChatMessage
from mellea.backends.model_ids import IBM_GRANITE_4_MICRO_3B
from mellea.core import ModelOutputThunk, SamplingResult
from mellea.stdlib.requirements import req, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy


def has_potential_pii(text: str) -> bool:
    """Quick heuristic check for potential PII patterns using spaCy NER."""
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    # Check for person names and locations
    pii_entities = ["PERSON", "GPE", "LOC", "ORG"]
    for ent in doc.ents:
        if ent.label_ in pii_entities:
            return True

    # Additional simple checks for email/phone patterns
    tokens = [token.text for token in doc]
    for token in tokens:
        # Email-like structure
        if "@" in token and "." in token:
            return True
        # Phone-like (contains multiple digits)
        if sum(c.isdigit() for c in token) >= 7:
            return True

    return False


def pii_remove_validate(
    m: mellea.MelleaSession,
    text: str,
    requirements: list[str] | None = None,
    loop_budget: int = 3,
    model_options: None | dict = None,
) -> ModelOutputThunk | SamplingResult | str:
    """PII scrubbing in mellea with validation."""
    # Extra requirements if any.
    requirements = requirements if requirements else []
    result = m.instruct(
        f"Remove all personally identifiable information from the following text "
        f"and replace it with XXX:\n\n{text}",
        requirements=[
            req(
                "Replace all names,email addresses, phone numbers, and addresses with XXX"
            ),
            req("Preserve non-PII content unchanged"),
            req(
                "Output must not contain PII",
                validation_fn=simple_validate(
                    lambda output: not has_potential_pii(output)
                ),
            ),
            *requirements,
        ],
        strategy=RejectionSamplingStrategy(loop_budget=loop_budget),
        return_sampling_results=True,
        model_options=model_options,
    )
    if result.success:
        return result
    else:
        return "The Validation Failed"


session = mellea.start_session(model_id=IBM_GRANITE_4_MICRO_3B)


def serve(
    input: list[ChatMessage],
    requirements: list[str] | None = None,
    model_options: None | dict = None,
) -> ModelOutputThunk | SamplingResult | str:
    """Simple serve example to do PII stuff."""
    message = input[-1].content or ""
    result = pii_remove_validate(
        session, message, requirements=requirements, model_options=model_options
    )
    return result
