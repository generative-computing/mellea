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

RE_GENERAL_INSTRUCTIONS = re.compile(
    r"<general_instructions>(.+?)</general_instructions>",
    flags=re.IGNORECASE | re.DOTALL,
)


@final
class _GeneralInstructions(PromptModule):
    @staticmethod
    def _default_parser(generated_str: str) -> str:
        general_instructions_match = re.search(RE_GENERAL_INSTRUCTIONS, generated_str)

        general_instructions_str: str | None = (
            general_instructions_match.group(1).strip()
            if general_instructions_match
            else None
        )

        if general_instructions_str is None:
            raise TagExtractionError(
                'LLM failed to generate correct tags for extraction: "<general_instructions>"'
            )

        return general_instructions_str

    def generate(
        self,
        mellea_session: MelleaSession,
        input_str: str | None,
        max_new_tokens: int = 4096,
        parser: Callable[[str], T] = _default_parser,  # type: ignore[assignment]
        # About the mypy ignore above: https://github.com/python/mypy/issues/3737
        **kwargs: dict[str, Any],
    ) -> PromptModuleString[T]:
        assert input_str is not None, 'This module requires the "input_str" argument'

        system_prompt = get_system_prompt()
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


general_instructions = _GeneralInstructions()
