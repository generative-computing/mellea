import re
from collections.abc import Callable, Sequence
from typing import TypedDict, TypeVar, cast, final

from typing_extensions import Unpack

from mellea import MelleaSession
from mellea.backends.types import ModelOption
from mellea.stdlib.instruction import Instruction

from .._prompt_modules import PromptModule, PromptModuleString
from ._exceptions import BackendGenerationError, TagExtractionError
from ._prompt import get_system_prompt, get_user_prompt
from ._types import SubtaskPromptItem

T = TypeVar("T")

RE_GEN_DATA_FORMAT = re.compile(
    r"@@@\|(.+?)\|@@@###\|(.+?)\|###(.+?)@@@\|DELIMITER\|@@@",
    flags=re.IGNORECASE | re.DOTALL,
)

RE_SUBTASK_PROMPT = re.compile(
    r"<subtask_prompt_template>(.+?)</subtask_prompt_template>",
    flags=re.IGNORECASE | re.DOTALL,
)


class SubtaskPromptGeneratorArgs(TypedDict):
    subtasks_and_tags: Sequence[tuple[str, str]]


@final
class _SubtaskPromptGenerator(PromptModule):
    @staticmethod
    def _default_parser(generated_str: str) -> list[SubtaskPromptItem]:
        r"""Default parser of the `subtask_prompt_generator` module.

        _**Disclaimer**: This is a LLM-prompting module, so the results will vary depending
        on the size and capabilities of the LLM used. The results are also not guaranteed, so
        take a look at this module's Exceptions and plan for unreliable results._

        Args:
            generated_str (`str`): The LLM's answer to be parsed
                (this `str` contains the result of the LLM calls
                for each subtask, separated by a character combination
                to enable parsing).

        Returns:
            list[SubtaskPromptItem]: A `list` of `NamedTuple` (`SubtaskPromptItem`) where each
            `tuple` contains the "subtask" (`str`), its "tag" (`str`) and
            its generated "prompt_template" (`str`).

            For example
            ```
            [ SubtaskPromptItem(subtask=<str>, tag=<str>, prompt_template=<str>),
              SubtaskPromptItem(subtask=<str>, tag=<str>, prompt_template=<str>) ]
            ```

            You can use dot notation to access the values. For example
            ```
            task_prompt = "..." # Original task prompt to be the reference when generating subtask prompts
            mellea_session = MelleaSession(...) # A mellea session with a backend
            subtasks = [ ("1. Read the document and write a summary", "DOCUMENT_SUMMARY"),
                         ("2. Write the 3 most important phrases in bullet points", "IMPORTANT_PHRASES") ]

            result: PromptModuleString = subtask_prompt_generator.generate(
                mellea_session,
                task_prompt,
                user_input_var_names=["INPUT_DOCUMENT_CONTENT"]
                subtasks_and_tags=subtasks,
            )

            parsed_result: list[SubtaskPromptItem] = result.parse()

            subtask_0: str = result[0].subtask
            tag_0: str = result[0].tag
            prompt_template_0: str = result[0].prompt_template
            ```

        Raises:
            TagExtractionError: An error occurred trying to extract content from the
                generated output. The LLM probably failed to open and close
                the \<subtask_prompt_template\> tags for one of the subtasks.
        """
        gen_data = re.findall(RE_GEN_DATA_FORMAT, generated_str)

        result: list[SubtaskPromptItem] = []

        for data in gen_data:
            data = cast(tuple[str, str, str], data)

            subtask_prompt_generator_match = re.search(RE_SUBTASK_PROMPT, data[2])

            generated_prompt_template: str | None = (
                subtask_prompt_generator_match.group(1).strip()
                if subtask_prompt_generator_match
                else None
            )

            if generated_prompt_template is None:
                raise TagExtractionError(
                    f'Error while processing the subtask: "{data[0]}"\n'
                    + 'LLM failed to generate correct tags for extraction: "<subtask_prompt_template>"'
                )

            result.append(
                SubtaskPromptItem(
                    subtask=data[0].strip(),
                    tag=data[1].strip(),
                    prompt_template=generated_prompt_template.strip(),
                )
            )

        return result

    def generate(  # type: ignore[override]
        # About the mypy ignore above:
        # Contrary to the "_ConstraintExtractor" implementation, this one does actually
        # break the Liskov Substitution Principle because of the required extra
        # arguments (with no default values) inside the "**kwargs". We can
        # later refactor the abstract class or even remove it completely.
        # TODO: Discussion and refactoring necessary (this works for now though).
        self,
        mellea_session: MelleaSession,
        input_str: str | None,
        max_new_tokens: int = 8192,
        parser: Callable[[str], T] = _default_parser,  # type: ignore[assignment]
        # About the mypy ignore statement above: https://github.com/python/mypy/issues/3737
        user_input_var_names: list[str] = [],
        **kwargs: Unpack[SubtaskPromptGeneratorArgs],
    ) -> PromptModuleString[T]:
        """Generates prompt templates for a list of subtasks based on the task that originated
        the list of subtasks.

        _**Disclaimer**: This is a LLM-prompting module, so the results will vary depending
        on the size and capabilities of the LLM used. The results are also not guaranteed, so
        take a look at this module's Exceptions and plan for unreliable results._

        Args:
            mellea_session (`MelleaSession`): A mellea session with a backend.
            input_str (`str`): Natural language (non-templated) prompt of the task that originated
                the list of subtasks passed on the `subtasks_and_tags` argument.
            max_new_tokens (`int`, optional): Maximum tokens to generate.
                Try increasing the value if you are getting `TagExtractionError`.
                Defaults to `8192`.
            parser (`Callable[[str], Any]`, optional): A string parsing function.
                Defaults to `_SubtaskPromptGenerator._default_parser`.
            user_input_var_names (`list[str]`, optional): A list of variable names
                (alphanumeric, uppercase, words separated by underscores) representing the
                user input data that your task needs to ingest.

                Let's say your task is for writing emails addressed to a prospect of a given company, then this task
                needs to ingest some variables, e.g.
                ```
                user_input_var_names = ["YOUR_NAME", "PROSPECT_NAME", "PROSPECT_COMPANY", "PRODUCT_DESCRIPTION"]
                ```
            subtasks_and_tags (`Sequence[tuple[str, str]]`): A list of subtasks and their respective tags.

                This was designed to receive the parsed result of the `subtask_list`
                module, but it's not required, you are able to provide arguments in the correct format.

                The list is composed of `tuple[str, str]` objects where the first position is
                the subtask title/description in natural language and the second position is a tag/variable
                with a descriptive name related to its subtask. e.g.
                ```
                subtasks_and_tags = [
                    ("1. Read the document and write a summary", "DOCUMENT_SUMMARY"),
                    ("2. Write the 3 most important phrases as bullets", "IMPORTANT_PHRASES"),
                ]
                ```

        Returns:
            PromptModuleString: A `PromptModuleString` class containing the generated output.

            The `PromptModuleString` class behaves like a `str`, but with an additional `parse()` method
            to execute the parsing function passed in the `parser` argument of
            this method (the `parser` argument defaults to `_SubtaskPromptGenerator._default_parser`).

        Raises:
            BackendGenerationError: Some error occurred during the LLM generation call.
        """
        assert input_str is not None, 'This module requires the "input_str" argument'

        user_input_variables_exists = True if user_input_var_names else False
        system_prompt = get_system_prompt(
            user_input_variables_exists=user_input_variables_exists
        )

        execution_plan = [
            f"{subtask_tag[0]} - Variable: {subtask_tag[1]}"
            for subtask_tag in kwargs["subtasks_and_tags"]
        ]

        all_results_string = ""

        # TODO: Make this whole segment execute concurrently using regular threading
        for i, subtask_tag in enumerate(kwargs["subtasks_and_tags"]):
            previous_tags = [kwargs["subtasks_and_tags"][j][1] for j in range(i)]

            # TODO: Validate the values of both "user_input_var_names" and "previous_tags"
            # Either use RegEx to validate or try to transform the string into the expected format
            # Requirements:
            # - The strings must be composed of uppercase alphanumeric characters
            # - Words can only be separated by underline character (no spaces)
            # - No consecutive underline characters
            available_content_variables = [
                r"{{" + item.upper() + r"}}" for item in user_input_var_names
            ] + [r"{{" + item + r"}}" for item in previous_tags]

            user_prompt = get_user_prompt(
                task_prompt=input_str,
                execution_plan=execution_plan,
                available_content_variables=available_content_variables,
                target_subtask=subtask_tag[0],
            )

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

            all_results_string = (
                all_results_string
                + f"@@@|{subtask_tag[0]}|@@@###|{subtask_tag[1]}|###\n"
                + gen_result
                + "@@@|DELIMITER|@@@\n"
            )

        return PromptModuleString(all_results_string, parser)


subtask_prompt_generator = _SubtaskPromptGenerator()
