from typing import Any


class ConstraintExtractorError(Exception):
    def __init__(self, error_message: str, **kwargs: dict[str, Any]):
        self.error_message = error_message
        self.__dict__.update(kwargs)
        super().__init__(f'Module Error "constraint_extractor"; {self.error_message}')


class BackendGenerationError(ConstraintExtractorError):
    def __init__(self, error_message: str, **kwargs: dict[str, Any]):
        super().__init__(error_message, **kwargs)


class TagExtractionError(ConstraintExtractorError):
    def __init__(self, error_message: str, **kwargs: dict[str, Any]):
        super().__init__(error_message, **kwargs)
