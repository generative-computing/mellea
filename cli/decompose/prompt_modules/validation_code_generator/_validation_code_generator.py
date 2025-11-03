import re
from collections.abc import Callable
from typing import Any, TypeVar, final

from mellea import MelleaSession
from mellea.backends.types import ModelOption
from mellea.stdlib.chat import Message

from .._prompt_modules import PromptModule, PromptModuleString
from ._exceptions import BackendGenerationError, TagExtractionError
from ._prompt import get_system_prompt, get_user_prompt

T = TypeVar("T")

RE_VALIDATION_FUNCTION = re.compile(
    r"<validation_function>(.+?)</validation_function>", flags=re.IGNORECASE | re.DOTALL
)


@final
class _ValidationCodeGenerator(PromptModule):
    @staticmethod
    def _default_parser(generated_str: str) -> str:
        r"""Default parser of the `validation_code_generator` module.

        _**Disclaimer**: This is a LLM-prompting module, so the results will vary depending
        on the size and capabilities of the LLM used. The results are also not guaranteed, so
        take a look at this module's Exceptions and plan for unreliable results._

        Args:
            generated_str (`str`): The LLM's answer to be parsed.

        Returns:
            str: The extracted Python validation function code.

        Raises:
            TagExtractionError: An error occurred trying to extract content from the
                generated output. The LLM probably failed to open and close
                the \<validation_function\> tags.
        """
        validation_function_match = re.search(RE_VALIDATION_FUNCTION, generated_str)

        validation_function_str: str | None = (
            validation_function_match.group(1).strip()
            if validation_function_match
            else None
        )

        if validation_function_str is None:
            raise TagExtractionError(
                'LLM failed to generate correct tags for extraction: "<validation_function>"'
            )

        return validation_function_str

    def generate(
        self,
        mellea_session: MelleaSession,
        input_str: str | None,
        max_new_tokens: int = 4096,
        parser: Callable[[str], T] = _default_parser,  # type: ignore[assignment]
        # About the mypy ignore above: https://github.com/python/mypy/issues/3737
        **kwargs: dict[str, Any],
    ) -> PromptModuleString[T]:
        """Generates a Python validation function based on a provided constraint/requirement.

        Args:
            mellea_session (`MelleaSession`): A mellea session with a backend.
            input_str (`str`): Natural language constraint/requirement to generate validation code for.
            prompt (`str`, optional): The original task prompt for context. Defaults to None.
            max_new_tokens (`int`, optional): Maximum tokens to generate.
                Defaults to `4096`.
            parser (`Callable[[str], Any]`, optional): A string parsing function.
                Defaults to `_ValidationCodeGenerator._default_parser`.

        Returns:
            PromptModuleString: A `PromptModuleString` class containing the generated output.

            The `PromptModuleString` class behaves like a `str`, but with an additional `parse()` method
            to execute the parsing function passed in the `parser` argument of
            this method (the `parser` argument defaults to `_ValidationCodeGenerator._default_parser`).

        Raises:
            BackendGenerationError: Some error occurred during the LLM generation call.
        """
        assert input_str is not None, 'This module requires the "input_str" argument'

        system_prompt = get_system_prompt()
        user_prompt = get_user_prompt(constraint_requirement=input_str)

        action = Message("user", user_prompt)

        try:
            gen_result = mellea_session.act(
                action=action,
                model_options={
                    ModelOption.SYSTEM_PROMPT: system_prompt,
                    ModelOption.TEMPERATURE: 0,
                    ModelOption.MAX_NEW_TOKENS: max_new_tokens,
                },
            ).value
        except Exception as e:
            raise BackendGenerationError(f"LLM generation failed: {e}")

        if gen_result is None:
            raise BackendGenerationError(
                "LLM generation failed: value attribute is None"
            )

        return PromptModuleString(gen_result, parser)


validation_code_generator = _ValidationCodeGenerator()
