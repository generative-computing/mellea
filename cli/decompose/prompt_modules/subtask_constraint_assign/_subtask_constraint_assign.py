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
from ._types import SubtaskPromptConstraintsItem

T = TypeVar("T")

RE_GEN_DATA_FORMAT = re.compile(
    r"@@@\|(.+?)\|@@@###\|(.+?)\|###[\r\n|\r|\n](.+?)@@@\|GENERATION\|@@@(.+?)@@@\|DELIMITER\|@@@",
    flags=re.IGNORECASE | re.DOTALL,
)

RE_ASSIGNED_CONS = re.compile(
    r"<assigned_constraints>(.+?)</assigned_constraints>",
    flags=re.IGNORECASE | re.DOTALL,
)


class SubtaskConstraintAssignArgs(TypedDict):
    subtasks_tags_and_prompts: Sequence[tuple[str, str, str]]
    constraint_list: Sequence[str]


@final
class _SubtaskConstraintAssign(PromptModule):
    @staticmethod
    def _default_parser(generated_str: str) -> list[SubtaskPromptConstraintsItem]:
        r"""Default parser of the `subtask_constraint_assign` module.

        _**Disclaimer**: This is a LLM-prompting module, so the results will vary depending
        on the size and capabilities of the LLM used. The results are also not guaranteed, so
        take a look at this module's Exceptions and plan for unreliable results._

        Args:
            generated_str (`str`): The LLM's answer to be parsed
                (this `str` contains the result of the LLM calls
                for each subtask, separated by a character combination
                to enable parsing).

        Returns:
            list[SubtaskPromptConstraintsItem]: A `list` of
            `NamedTuple` (`SubtaskPromptConstraintsItem`) where each
            `tuple` contains the "subtask" (`str`), its "tag" (`str`), its
            generated "prompt_template" (`str`), and
            its assigned "constraints" (`list[str]`).

            Note that the result "constraints" list can be empty.

            For example
            ```
            [ SubtaskPromptConstraintsItem(
                  subtask=<str>,
                  tag=<str>,
                  prompt_template=<str>
                  constraints=<list[str]>
              ),
              ...
            ]
            ```

            You can use dot notation to access the values. For example
            ```
            result: PromptModuleString = # Result of the subtask_constraint_assign.generate() method

            parsed_result: list[SubtaskPromptConstraintsItem] = result.parse()

            subtask_0: str = result[0].subtask
            tag_0: str = result[0].tag
            prompt_template_0: str = result[0].prompt_template
            constraints_0: list[str] = result[0].constraints
            ```

        Raises:
            TagExtractionError: An error occurred trying to extract content from the
                generated output. The LLM probably failed to open and close
                the \<subtask_prompt_template\> tags for one of the subtasks.
        """
        gen_data = re.findall(RE_GEN_DATA_FORMAT, generated_str)

        result: list[SubtaskPromptConstraintsItem] = []

        for data in gen_data:
            data = cast(tuple[str, str, str, str], data)

            subtask_constraint_assign_match = re.search(RE_ASSIGNED_CONS, data[3])

            subtask_constraint_assign_str: str | None = (
                subtask_constraint_assign_match.group(1).strip()
                if subtask_constraint_assign_match
                else None
            )

            if subtask_constraint_assign_str is None:
                raise TagExtractionError(
                    'LLM failed to generate correct tags for extraction: "<assigned_constraints>"'
                )

            subtask_constraint_assign_str_upper = subtask_constraint_assign_str.upper()
            if (
                "N/A" in subtask_constraint_assign_str_upper
                or "N / A" in subtask_constraint_assign_str_upper
                or "N/ A" in subtask_constraint_assign_str_upper
                or "N /A" in subtask_constraint_assign_str_upper
            ):
                subtask_constraint_assign = []
            else:
                subtask_constraint_assign = [
                    line.strip()[2:] if line.strip()[:2] == "- " else line.strip()
                    for line in subtask_constraint_assign_str.splitlines()
                ]

            result.append(
                SubtaskPromptConstraintsItem(
                    subtask=data[0].strip(),
                    tag=data[1].strip(),
                    prompt_template=data[2].strip(),
                    constraints=subtask_constraint_assign,
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
        input_str: str | None = None,
        max_new_tokens: int = 8192,
        parser: Callable[[str], T] = _default_parser,  # type: ignore[assignment]
        # About the mypy ignore statement above: https://github.com/python/mypy/issues/3737
        **kwargs: Unpack[SubtaskConstraintAssignArgs],
    ) -> PromptModuleString[T]:
        """Receives a list of subtasks (with their tags and template prompts) and a list of
        constraints written in natural language.

        Selects and assign, to each subtask, the constraints that the LLM judges
        to be appropriate (amongst the provided constraint list) to each subtask.

        _**Disclaimer**: This is a LLM-prompting module, so the results will vary depending
        on the size and capabilities of the LLM used. The results are also not guaranteed, so
        take a look at this module's Exceptions and plan for unreliable results._

        Args:
            mellea_session (`MelleaSession`): A mellea session with a backend.
            input_str (`None`, optional): This module doesn't use the "input_str" argument.
            max_new_tokens (`int`, optional): Maximum tokens to generate.
                Try increasing the value if you are getting `TagExtractionError`.
                Defaults to `8192`.
            parser (`Callable[[str], Any]`, optional): A string parsing function.
                Defaults to `_SubtaskConstraintAssign._default_parser`.
            subtasks_tags_and_prompts (`Sequence[tuple[str, str, str]]`): A list of subtasks,
                their respective tags and prompts.

                This was designed to receive the parsed result of the `subtask_prompt_generator`
                module, but it's not required, you are able to provide arguments in the correct format.

                The list must be composed of `tuple[str, str, str]` objects where the first position is
                the subtask title/description in natural language, the second position is a tag/variable
                with a descriptive name related to its subtask, and the third position is the template
                prompt for an LLM to execute the subtask. e.g.
                ```
                subtasks_tags_and_prompts = [
                    ("1. Read the document and write a summary", "DOCUMENT_SUMMARY", "<template_prompt>"),
                    ("2. Write the 3 most important phrases as bullets", "IMPORTANT_PHRASES", "<template_prompt>")
                ]
                ```
            constraint_list (`Sequence[str]`): A list of constraints written in natural language.

                This was designed to take in a list of constraints identified from the prompt
                that originated the subtasks provided, so they can be correctly
                distributed and assigned to the subtasks.

        Returns:
            PromptModuleString: A `PromptModuleString` class containing the generated output.

            The `PromptModuleString` class behaves like a `str`, but with an additional `parse()` method
            to execute the parsing function passed in the `parser` argument of
            this method (the `parser` argument defaults to `_SubtaskConstraintAssign._default_parser`).

        Raises:
            BackendGenerationError: Some error occurred during the LLM generation call.
        """
        system_prompt = get_system_prompt()

        execution_plan = [
            f"{subtask_tag_prompt[0]} - Variable: {subtask_tag_prompt[1]}"
            for subtask_tag_prompt in kwargs["subtasks_tags_and_prompts"]
        ]

        all_results_string = ""

        # TODO: Make this whole segment execute concurrently using regular threading
        for i, subtask_tag_prompt in enumerate(kwargs["subtasks_tags_and_prompts"]):
            user_prompt = get_user_prompt(
                execution_plan=execution_plan,
                constraint_list=kwargs["constraint_list"],
                subtask_title=subtask_tag_prompt[0],
                subtask_prompt=subtask_tag_prompt[2],
            )

            instruction = Instruction(description=user_prompt)

            try:
                gen_result = mellea_session.backend.generate_from_context(
                    action=instruction,
                    ctx=mellea_session.ctx,
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

            all_results_string = (
                all_results_string
                + f"@@@|{subtask_tag_prompt[0]}|@@@###|{subtask_tag_prompt[1]}|###\n"
                + subtask_tag_prompt[2].strip()
                + "@@@|GENERATION|@@@"
                + gen_result.strip()
                + "@@@|DELIMITER|@@@\n"
            )

        return PromptModuleString(all_results_string, parser)


subtask_constraint_assign = _SubtaskConstraintAssign()
