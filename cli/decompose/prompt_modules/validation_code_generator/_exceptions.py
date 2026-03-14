from typing import Any


class ValidationCodeGeneratorError(Exception):
    def __init__(self, error_message: str, **kwargs: dict[str, Any]):
        self.error_message = error_message
        self.__dict__.update(kwargs)
        super().__init__(
            f'Module Error "validation_code_generator"; {self.error_message}'
        )


class BackendGenerationError(ValidationCodeGeneratorError):
    """Raised when LLM generation fails in the "validation_code_generator" prompt module."""

    def __init__(self, error_message: str, **kwargs: dict[str, Any]):
        super().__init__(error_message, **kwargs)


class TagExtractionError(ValidationCodeGeneratorError):
    """Raised when tag extraction fails in the "validation_code_generator" prompt module."""

    def __init__(self, error_message: str, **kwargs: dict[str, Any]):
        super().__init__(error_message, **kwargs)
