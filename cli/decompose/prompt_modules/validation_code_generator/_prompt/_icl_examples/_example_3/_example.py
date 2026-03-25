# ruff: noqa: W293
from .._types import ICLExample

constraint_requirement = "Return a list of requirements, using dash bullets (-), where each item begins with the relevant entity"

validation_function = """def validate_input(input: str) -> bool:
    \"""
    Validates that the input is a list of requirements using dash bullets,
    where each item begins with the relevant entity.
    
    Args:
        input (str): The input to validate
        
    Returns:
        bool: True if input follows the required format, False otherwise
    \"""
    try:
        if not input or not isinstance(input, str):
            return False
            
        lines = input.strip().split('\n')
        
        # Check if all lines are empty
        if not any(line.strip() for line in lines):
            return False
            
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Check if line starts with a dash bullet
            if not line.startswith('- '):
                return False
                
            # Check if there's content after the dash bullet
            content = line[2:].strip()  # Remove '- ' prefix
            if not content:
                return False
                
            # Check if content has an entity (word) at the beginning
            words = content.split()
            if not words:
                return False
                
            # Entity should be the first word - just check it exists
            # We're not validating what constitutes a valid entity here
            
        return True
    except Exception:
        return False"""

example: ICLExample = {
    "constraint_requirement": constraint_requirement,
    "validation_function": validation_function,
}
