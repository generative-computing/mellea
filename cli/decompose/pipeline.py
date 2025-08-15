from typing import NotRequired, TypedDict

from mellea import MelleaSession
from mellea.backends.ollama import OllamaModelBackend
from mellea.prompt_modules import (
    constraint_extractor,
    subtask_constraint_assign,
    subtask_list,
    subtask_prompt_generator,
)
from mellea.prompt_modules.subtask_constraint_assign import SubtaskPromptConstraintsItem
from mellea.prompt_modules.subtask_list import SubtaskItem
from mellea.prompt_modules.subtask_prompt_generator import SubtaskPromptItem


class DecompSubtasksResult(TypedDict):
    subtask: str
    tag: str
    constraints: list[str]
    prompt_template: str
    generated_response: NotRequired[str]


class DecompPipelineResult(TypedDict):
    original_task_prompt: str
    subtask_list: list[str]
    identified_constraints: list[str]
    subtasks: list[DecompSubtasksResult]
    final_response: NotRequired[str]


def decompose(
    task_prompt: str, user_input_variable: list[str] | None = None
) -> DecompPipelineResult:
    if user_input_variable is None:
        user_input_variable = []

    m_ollama_session = MelleaSession(
        OllamaModelBackend(model_id="mistral-small3.2:24b")
    )

    subtasks: list[SubtaskItem] = subtask_list.generate(
        m_ollama_session, task_prompt
    ).parse()

    task_prompt_constraints: list[str] = constraint_extractor.generate(
        m_ollama_session, task_prompt
    ).parse()

    subtask_prompts: list[SubtaskPromptItem] = subtask_prompt_generator.generate(
        m_ollama_session,
        task_prompt,
        user_input_var_names=user_input_variable,
        subtasks_and_tags=subtasks,
    ).parse()

    subtask_prompts_with_constraints: list[SubtaskPromptConstraintsItem] = (
        subtask_constraint_assign.generate(
            m_ollama_session,
            subtasks_tags_and_prompts=subtask_prompts,
            constraint_list=task_prompt_constraints,
        ).parse()
    )

    decomp_subtask_result: list[DecompSubtasksResult] = [
        DecompSubtasksResult(
            subtask=subtask_data.subtask,
            tag=subtask_data.tag,
            constraints=subtask_data.constraints,
            prompt_template=subtask_data.prompt_template,
        )
        for subtask_data in subtask_prompts_with_constraints
    ]

    return DecompPipelineResult(
        original_task_prompt=task_prompt,
        subtask_list=[item.subtask for item in subtasks],
        identified_constraints=task_prompt_constraints,
        subtasks=decomp_subtask_result,
    )
