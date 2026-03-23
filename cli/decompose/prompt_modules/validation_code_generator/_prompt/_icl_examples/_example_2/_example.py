# ruff: noqa: W293
from .._types import ICLExample

constraint_requirement = """The answer must be a JSON with the following keys:
1. "subject"
2. "content\""""

validation_function = """import json

def validate_input(input: str) -> bool:
    \"""
    Validates that the input is a JSON with required keys: subject and content.
    
    Args:
        input (str): The input to validate
        
    Returns:
        bool: True if JSON has required keys, False otherwise
    \"""
    try:
        data = json.loads(response)
        return isinstance(data, dict) and "subject" in data and "content" in data
    except json.JSONDecodeError:
        return False
    except Exception:
        return False"""

example: ICLExample = {
    "constraint_requirement": constraint_requirement,
    "validation_function": validation_function,
}
