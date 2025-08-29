from typing import Any


class SubtaskListModuleError(Exception):
    def __init__(self, error_message: str, **kwargs: dict[str, Any]):
        self.error_message = error_message
        self.__dict__.update(kwargs)
        super().__init__(f'Module Error "subtask_list"; {self.error_message}')


class BackendGenerationError(SubtaskListModuleError):
    def __init__(self, error_message: str, **kwargs: dict[str, Any]):
        super().__init__(error_message, **kwargs)


class TagExtractionError(SubtaskListModuleError):
    def __init__(self, error_message: str, **kwargs: dict[str, Any]):
        super().__init__(error_message, **kwargs)


class SubtaskLineParseError(SubtaskListModuleError):
    def __init__(self, error_message: str, **kwargs: dict[str, Any]):
        super().__init__(error_message, **kwargs)
