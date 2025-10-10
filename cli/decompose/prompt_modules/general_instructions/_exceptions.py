from typing import Any


class GeneralInstructionsError(Exception):
    def __init__(self, error_message: str, **kwargs: dict[str, Any]):
        self.error_message = error_message
        self.__dict__.update(kwargs)
        super().__init__(f'Module Error "general_instructions"; {self.error_message}')


class BackendGenerationError(GeneralInstructionsError):
    def __init__(self, error_message: str, **kwargs: dict[str, Any]):
        super().__init__(error_message, **kwargs)


class TagExtractionError(GeneralInstructionsError):
    def __init__(self, error_message: str, **kwargs: dict[str, Any]):
        super().__init__(error_message, **kwargs)
