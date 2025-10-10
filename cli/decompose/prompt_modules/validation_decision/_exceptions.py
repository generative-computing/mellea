from typing import Any


class ValidationDecisionError(Exception):
    def __init__(self, error_message: str, **kwargs: dict[str, Any]):
        self.error_message = error_message
        self.__dict__.update(kwargs)
        super().__init__(f'Module Error "validation_decision"; {self.error_message}')


class BackendGenerationError(ValidationDecisionError):
    """Raised when LLM generation fails in the "validation_decision" prompt module."""

    def __init__(self, error_message: str, **kwargs: dict[str, Any]):
        super().__init__(error_message, **kwargs)


class TagExtractionError(ValidationDecisionError):
    """Raised when tag extraction fails in the "validation_decision" prompt module."""

    def __init__(self, error_message: str, **kwargs: dict[str, Any]):
        super().__init__(error_message, **kwargs)
