"""Abstract interfaces for Backends that implement Process Reward Models (can be adapted to include other scorers)"""

import abc


class PRM(abc.ABC):
    def __init__(self, model_name_or_path):
        # Leave implementation of model to inheriting class
        self.model_name_or_path = model_name_or_path

    @abc.abstractmethod
    def score(self, query: str, response: str) -> tuple[list[float], list[list[float]]]:
        """Returns a final score and per-step score to the input of the model"""
        ...

    @abc.abstractmethod
    def stepify(self, response: str, step_separator: str) -> list[str]:
        """Splits the assistant response into steps to score

        Args:
            response: assistant response to score
            step_separator: string on which to separate the response into steps
        """
        ...
