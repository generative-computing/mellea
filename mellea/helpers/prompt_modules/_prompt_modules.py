from abc import ABC, abstractmethod
from collections import UserString
from collections.abc import Callable
from typing import Any, Generic, TypeVar

from mellea import MelleaSession

T = TypeVar("T")


class PromptModuleString(UserString, Generic[T]):
    """A custom string class with a parse method provided at initialization."""

    def __init__(self, string: str, parser: Callable[[str], T]):
        """Initialize the `PromptModuleString` with a string and
        a parser function.

        Args:
            string (`str`): The string content.
            parser (`Callable[[str], T]`): A function to parse the string content
                and return a value of type T.
        """
        self._parser = parser
        super().__init__(string)

    def parse(self) -> T:
        """Parses the string content using the parser function
        provided at initialization.

        Returns:
            T: The result of applying the parser function to
                the string content.
        """
        return self._parser(self.__str__())


class PromptModule(ABC):
    """Abstract base class for prompt modules."""

    @staticmethod
    @abstractmethod
    def _default_parser(generated_str: str) -> Any:
        """Abstract static method to serve as the default parser
        for the `PromptModuleString` produced by the prompt module.

        Args:
            generated_str (`str`): The generated string to be parsed.

        Returns:
            Any: The parsing result.
        """
        ...

    @abstractmethod
    def generate(
        self,
        mellea_session: MelleaSession,
        input_str: str | None,
        max_new_tokens: int,
        parser: Callable[[str], T] = _default_parser,
        **kwargs: dict[str, Any],
    ) -> PromptModuleString[T]:
        """Abstract method to generate any result based on LLM prompting.

        Args:
            input_str (`str`): The target string of the prompt module.
                The corresponding implementation must document what
                this string input should be.
            mellea_session (`MelleaSession`): A `MelleaSession` with a
                `Backend` to run LLM queries.
            max_new_tokens (`int`): The maximum number of new tokens
                to generate.
            parser (`Callable[[str], Any]`, optional): The parser function
                to use for the generated `PromptModuleString`.
                Defaults to `PromptModule._default_parser`.
            **kwargs (`Dict[str, Any]`): Additional keyword arguments that the
                implementation might need.

        Returns:
            PromptModuleString[Any]: The string result of the prompt module.
        """
        ...
