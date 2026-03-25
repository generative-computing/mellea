# ruff: noqa: W293
from .._types import ICLExample

constraint_requirement = """You must not use any uppercase letters"""

validation_function = """def validate_input(input: str) -> bool:
    \"""
    Validates that the input contains only lowercase letters.
    
    Args:
        input (str): The input to validate
        
    Returns:
        bool: True if all characters are lowercase, False otherwise
    \"""
    try:
        return answer.islower()
    except Exception:
        return False"""

example: ICLExample = {
    "constraint_requirement": constraint_requirement,
    "validation_function": validation_function,
}
