# ruff: noqa: W293
from .._types import ICLExample

constraint_requirement = 'Avoid the words "daughter-in-law" and "grandson"'

validation_function = """def validate_input(input: str) -> bool:
    \"""
    Validates that the input does not contain the words "daughter-in-law" and "grandson".
    
    Args:
        input (str): The input to validate
        
    Returns:
        bool: True if neither word is found, False otherwise
    \"""
    try:
        if not input:
            return False
            
        # Convert to lowercase for case-insensitive comparison
        input_lower = input.lower()
        
        # Check if either forbidden word is present
        return "daughter-in-law" not in input_lower and "grandson" not in input_lower
    except Exception:
        return False"""

example: ICLExample = {
    "constraint_requirement": constraint_requirement,
    "validation_function": validation_function,
}
