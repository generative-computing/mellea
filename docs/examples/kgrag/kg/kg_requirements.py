"""Mellea Requirements for KG-RAG validation."""
import json
from mellea.stdlib.requirement import Requirement, ValidationResult
from mellea.stdlib.base import Context


def is_valid_json(ctx: Context) -> ValidationResult:
    """Check if output is valid JSON."""
    try:
        output = ctx.last_assistant_message.as_str()
        json.loads(output)
        return True
    except (json.JSONDecodeError, TypeError, AttributeError):
        return False


def has_required_json_field(field: str):
    """Check if JSON output has a required field."""
    def validator(ctx: Context) -> ValidationResult:
        try:
            output = ctx.last_assistant_message.as_str()
            data = json.loads(output)
            return field in data and data[field] is not None
        except (json.JSONDecodeError, TypeError, KeyError, AttributeError):
            return False
    return validator


def has_nonempty_list(field: str):
    """Check if JSON field contains a non-empty list."""
    def validator(ctx: Context) -> ValidationResult:
        try:
            output = ctx.last_assistant_message.as_str()
            data = json.loads(output)
            return field in data and isinstance(data[field], list) and len(data[field]) > 0
        except (json.JSONDecodeError, TypeError, KeyError, AttributeError):
            return False
    return validator


def scores_sum_to_one(field: str, tolerance: float = 0.1):
    """Check if scores in a dict approximately sum to 1."""
    def validator(ctx: Context) -> ValidationResult:
        try:
            output = ctx.last_assistant_message.as_str()
            data = json.loads(output)
            if field not in data or not isinstance(data[field], dict):
                return False
            scores = data[field].values()
            total = sum(float(s) for s in scores)
            return abs(total - 1.0) <= tolerance
        except (json.JSONDecodeError, TypeError, ValueError, KeyError, AttributeError):
            return False
    return validator


# Define reusable requirements
VALID_JSON_REQ = Requirement(
    description="Output must be valid JSON format",
    validation_fn=is_valid_json
)

ROUTES_PRESENT_REQ = Requirement(
    description="Output must contain 'routes' field with at least one route",
    validation_fn=has_nonempty_list("routes")
)

ENTITIES_PRESENT_REQ = Requirement(
    description="Output must contain 'entities' field with at least one entity",
    validation_fn=has_nonempty_list("entities")
)

REASON_PRESENT_REQ = Requirement(
    description="Output must contain 'reason' field",
    validation_fn=has_required_json_field("reason")
)

RELEVANT_ENTITIES_REQ = Requirement(
    description="Output must contain 'relevant_entities' dict",
    validation_fn=has_required_json_field("relevant_entities")
)

RELEVANT_RELATIONS_REQ = Requirement(
    description="Output must contain 'relevant_relations' dict",
    validation_fn=has_required_json_field("relevant_relations")
)

SCORES_SUM_REQ = Requirement(
    description="Relevance scores should approximately sum to 1.0",
    validation_fn=scores_sum_to_one("relevant_entities")
)

EVALUATION_FIELDS_REQ = Requirement(
    description="Output must contain 'sufficient', 'reason', and 'answer' fields",
    validation_fn=lambda ctx: all(
        has_required_json_field(f)(ctx) for f in ["sufficient", "reason", "answer"]
    )
)

VALIDATION_FIELDS_REQ = Requirement(
    description="Output must contain 'judgement' and 'final_answer' fields",
    validation_fn=lambda ctx: all(
        has_required_json_field(f)(ctx) for f in ["judgement", "final_answer"]
    )
)


def get_requirements_for_task(task: str) -> list[Requirement]:
    """Get appropriate requirements for a specific KG-RAG task.

    Args:
        task: One of 'break_down', 'extract_entity', 'align_topic', 'prune_relations',
              'prune_triplets', 'evaluate', 'validate', 'direct_answer'

    Returns:
        List of requirements for the task
    """
    requirements_map = {
        "break_down": [VALID_JSON_REQ, ROUTES_PRESENT_REQ, REASON_PRESENT_REQ],
        "extract_entity": [VALID_JSON_REQ, ENTITIES_PRESENT_REQ],
        "align_topic": [VALID_JSON_REQ, RELEVANT_ENTITIES_REQ, REASON_PRESENT_REQ],
        "prune_relations": [VALID_JSON_REQ, RELEVANT_RELATIONS_REQ, REASON_PRESENT_REQ],
        "prune_triplets": [VALID_JSON_REQ, RELEVANT_RELATIONS_REQ, REASON_PRESENT_REQ],
        "evaluate": [VALID_JSON_REQ, EVALUATION_FIELDS_REQ],
        "validate": [VALID_JSON_REQ, VALIDATION_FIELDS_REQ],
        "direct_answer": [VALID_JSON_REQ, EVALUATION_FIELDS_REQ],
    }

    return requirements_map.get(task, [VALID_JSON_REQ])
