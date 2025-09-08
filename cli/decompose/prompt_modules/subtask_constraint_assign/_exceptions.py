from typing import Any


class SubtaskConstraintAssignError(Exception):
    def __init__(self, error_message: str, **kwargs: dict[str, Any]):
        self.error_message = error_message
        self.__dict__.update(kwargs)
        super().__init__(
            f'Module Error "subtask_constraint_assign"; {self.error_message}'
        )


class BackendGenerationError(SubtaskConstraintAssignError):
    def __init__(self, error_message: str, **kwargs: dict[str, Any]):
        super().__init__(error_message, **kwargs)


class TagExtractionError(SubtaskConstraintAssignError):
    def __init__(self, error_message: str, **kwargs: dict[str, Any]):
        super().__init__(error_message, **kwargs)
