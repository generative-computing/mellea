import re
from collections.abc import Callable
from typing import Any, TypeVar, final

from mellea import MelleaSession
from mellea.backends.types import ModelOption
from mellea.prompt_modules._prompt_modules import PromptModule, PromptModuleString
from mellea.stdlib.instruction import Instruction

from ._exceptions import (
    BackendGenerationError,
    SubtaskLineParseError,
    TagExtractionError,
)
from ._prompt import get_system_prompt, get_user_prompt
from ._types import SubtaskItem

# from mellea.stdlib.requirement import Requirement

T = TypeVar("T")

RE_SUBTASK_AND_TAG = re.compile(
    r"(.*\S)\s*-\s*Variable\s*:\s*(\w+)", flags=re.IGNORECASE
)
RE_FINAL_SUBTASK_LIST = re.compile(
    r"<subtask_list>(.+?)</subtask_list>", flags=re.IGNORECASE | re.DOTALL
)


def _parse_subtask_list_line(line: str) -> tuple[str, str]:
    matches = re.match(RE_SUBTASK_AND_TAG, line)
    try:
        subtask: str | None = matches.group(1).strip() if matches is not None else None
        tag: str | None = matches.group(2).strip() if matches is not None else None
        assert type(subtask) is str and len(subtask) > 0
        assert type(tag) is str and len(tag) > 0
    except (IndexError, AssertionError):
        raise SubtaskLineParseError(f'Wrong subtask line format: "{line}"')

    return (subtask, tag)


@final
class _SubtaskList(PromptModule):
    @staticmethod
    def _default_parser(generated_str: str) -> list[SubtaskItem]:
        r"""Default parser of the `subtask_list` module.

        _**Disclaimer**: This is a LLM-prompting module, so the results will vary depending
        on the size and capabilities of the LLM used. The results are also not guaranteed, so
        take a look at this module's Exceptions and plan for unreliable results._

        Args:
            generated_str (`str`): The LLM's answer to be parsed.

        Returns:
            list[SubtaskItem]: A `list` of `NamedTuple` (`SubtaskItem`) where each
            `tuple` contains the generated "subtask" (`str`) and its generated "tag" (`str`).

            For example
            ```
            [ SubtaskItem(subtask=<str>, tag=<str>),
              SubtaskItem(subtask=<str>, tag=<str>) ]
            ```

            You can use dot notation to access the values. For example
            ```
            result: PromptModuleString = subtask_list.generate(task_prompt, mellea_session)
            parsed_result: list[SubtaskItem] = result.parse()
            subtask_0: str = result[0].subtask
            tag_0: str = result[0].tag
            ```

        Raises:
            TagExtractionError: An error occurred trying to extract content from the
                generated output. The LLM probably failed to open and close
                the \<final_subtask_list\> tags.
            SubtaskLineParseError: An error occurred trying to parse the subtask line.
                The LLM probably failed to generate the expected format inside
                the \<final_subtask_list\> tags.
        """
        subtask_list_match = re.search(RE_FINAL_SUBTASK_LIST, generated_str)

        subtask_list_str: str | None = (
            subtask_list_match.group(1).strip() if subtask_list_match else None
        )

        if subtask_list_str is None:
            raise TagExtractionError(
                'LLM failed to generate correct tags for extraction: "<final_subtask_list>"'
            )

        subtask_list_lines = [line.strip() for line in subtask_list_str.splitlines()]

        try:
            subtask_tag_list = [
                _parse_subtask_list_line(line) for line in subtask_list_lines
            ]
        except AssertionError:
            raise SubtaskLineParseError(
                "Failed parsing a subtask line from the <final_subtask_list> tags"
            )

        return [SubtaskItem(subtask=item[0], tag=item[1]) for item in subtask_tag_list]

    def generate(
        self,
        mellea_session: MelleaSession,
        input_str: str | None,
        max_new_tokens: int = 8192,
        parser: Callable[[str], T] = _default_parser,  # type: ignore[assignment]
        # About the mypy ignore statement above: https://github.com/python/mypy/issues/3737
        **kwargs: dict[str, Any],
    ) -> PromptModuleString[T]:
        """Generates a numbered list of subtasks (titles only) based on a provided task prompt.

        _**Disclaimer**: This is a LLM-prompting module, so the results will vary depending
        on the size and capabilities of the LLM used. The results are also not guaranteed, so
        take a look at this module's Exceptions and use them accordingly._

        Args:
            mellea_session (`MelleaSession`): A mellea session with a backend.
            input_str (`str`): Natural language (non-templated) prompt describing a task to be executed.
            max_new_tokens (`int`, optional): Maximum tokens to generate.
                Try increasing the value if you are getting `TagExtractionError`.
                Defaults to `8192`.
            parser (`Callable[[str], Any]`, optional): A string parsing function. Defaults to `_SubtaskList._default_parser`.

        Returns:
            PromptModuleString: A `PromptModuleString` class containing the generated output.

            The `PromptModuleString` class behaves like a `str`, but with an additional `parse()` method
            to execute the parsing function passed in the `parser` argument of
            this method (the `parser` argument defaults to `_SubtaskList._default_parser`).

        Raises:
            BackendGenerationError: Some error occurred during the LLM generation call.
        """
        assert input_str is not None, 'This module requires the "input_str" argument'

        system_prompt = get_system_prompt()
        user_prompt = get_user_prompt(task_prompt=input_str)

        instruction = Instruction(description=user_prompt, prefix=system_prompt)

        try:
            gen_result = mellea_session.backend.generate_from_context(
                action=instruction,
                ctx=mellea_session.ctx,
                model_options={
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


subtask_list = _SubtaskList()
