import re
from collections.abc import Callable
from typing import Any, TypeVar, final

from mellea import MelleaSession
from mellea.backends import ModelOption
from mellea.stdlib.components import Message

from .._prompt_modules import PromptModule, PromptModuleString
from ._exceptions import BackendGenerationError, TagExtractionError
from ._prompt import get_system_prompt, get_user_prompt

T = TypeVar("T")

RE_VERIFIED_CONS_COND = re.compile(
    r"<constraints_and_requirements>(.+?)</constraints_and_requirements>",
    flags=re.IGNORECASE | re.DOTALL,
)


@final
class _ConstraintExtractor(PromptModule):
    @staticmethod
    def _default_parser(generated_str: str) -> list[str]:
        r"""Default parser of the `constraint_extractor` module.

        _**Disclaimer**: This is a LLM-prompting module, so the results will vary depending
        on the size and capabilities of the LLM used. The results are also not guaranteed, so
        take a look at this module's Exceptions and plan for unreliable results._

        Args:
            generated_str (`str`): The LLM's answer to be parsed.

        Returns:
            list[str]: A list of identified constraints and requirements in natural language. The list
            will be empty if no constraints were identified by the LLM.

        Raises:
            TagExtractionError: An error occurred trying to extract content from the
                generated output. The LLM probably failed to open and close
                the \<constraints_and_requirements\> tags.
        """
        constraint_extractor_match = re.search(RE_VERIFIED_CONS_COND, generated_str)

        constraint_extractor_str: str | None = (
            constraint_extractor_match.group(1).strip()
            if constraint_extractor_match
            else None
        )

        if constraint_extractor_str is None:
            raise TagExtractionError(
                'LLM failed to generate correct tags for extraction: "<constraints_and_requirements>"'
            )

        s = constraint_extractor_str.strip()
        s_norm = s.strip().upper().replace(" ", "")
        if s_norm == "N/A":
            return []

        results: list[str] = []

        for line in s.splitlines():
            line = line.strip()
            if not line:
                continue

            # remove bullet / numbering
            line = re.sub(r"^\s*(?:[-*•]|\d+[\.\)])\s+", "", line)

            # split inline multi-constraints
            parts = re.split(r"\s*(?:;|\s-\s|\s—\s|\s–\s)\s*", line)

            results.extend(p.strip() for p in parts if p.strip())

        return results

    def generate(  # type: ignore[override]
        # About the mypy ignore above:
        # Since the extra argument has a default value, it should be safe to override.
        # It doesn't violate the Liskov Substitution Principle, but mypy doesn't like it.
        self,
        mellea_session: MelleaSession,
        input_str: str | None,
        max_new_tokens: int = 4096,
        parser: Callable[[str], T] = _default_parser,  # type: ignore[assignment]
        # About the mypy ignore above: https://github.com/python/mypy/issues/3737
        enforce_same_words: bool = False,
        **kwargs: dict[str, Any],
    ) -> PromptModuleString[T]:
        """Generates an unordered list of identified constraints and requirements based on a provided task prompt.

        _**Disclaimer**: This is a LLM-prompting module, so the results will vary depending
        on the size and capabilities of the LLM used. The results are also not guaranteed, so
        take a look at this module's Exceptions and use them accordingly._

        Args:
            mellea_session (`MelleaSession`): A mellea session with a backend.
            input_str (`str`): Natural language (non-templated) prompt describing a task to be executed.
            max_new_tokens (`int`, optional): Maximum tokens to generate.
                Try increasing the value if you are getting `TagExtractionError`.
                Defaults to `8192`.
            parser (`Callable[[str], Any]`, optional): A string parsing function.
                Defaults to `_ConstraintExtractor._default_parser`.

        Returns:
            PromptModuleString: A `PromptModuleString` class containing the generated output.

            The `PromptModuleString` class behaves like a `str`, but with an additional `parse()` method
            to execute the parsing function passed in the `parser` argument of
            this method (the `parser` argument defaults to `_ConstraintExtractor._default_parser`).

        Raises:
            BackendGenerationError: Some error occurred during the LLM generation call.
        """
        assert input_str is not None, 'This module requires the "input_str" argument'

        system_prompt = get_system_prompt(enforce_same_words=enforce_same_words)
        user_prompt = get_user_prompt(task_prompt=input_str)

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


constraint_extractor = _ConstraintExtractor()
