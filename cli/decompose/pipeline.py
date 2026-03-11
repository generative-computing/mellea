import re
from enum import StrEnum
from typing import Literal, NotRequired, TypedDict

from mellea import MelleaSession
from mellea.backends import ModelOption
from mellea.backends.ollama import OllamaModelBackend
from mellea.backends.openai import OpenAIBackend

from .prompt_modules import (
    constraint_extractor,
    general_instructions,
    subtask_constraint_assign,
    subtask_list,
    subtask_prompt_generator,
    validation_code_generator,
    validation_decision,
)
from .prompt_modules.subtask_constraint_assign import SubtaskPromptConstraintsItem
from .prompt_modules.subtask_list import SubtaskItem
from .prompt_modules.subtask_prompt_generator import SubtaskPromptItem


class ConstraintValData(TypedDict):
    val_strategy: Literal["code", "llm"]
    val_fn: str | None


class ConstraintResult(TypedDict):
    constraint: str
    val_strategy: Literal["code", "llm"]
    val_fn: str | None
    val_fn_name: str


class DecompSubtasksResult(TypedDict):
    subtask: str
    tag: str
    constraints: list[ConstraintResult]
    prompt_template: str
    general_instructions: str
    input_vars_required: list[str]
    depends_on: list[str]
    generated_response: NotRequired[str]


class DecompPipelineResult(TypedDict):
    original_task_prompt: str
    subtask_list: list[str]
    identified_constraints: list[ConstraintResult]
    subtasks: list[DecompSubtasksResult]
    final_response: NotRequired[str]


class DecompBackend(StrEnum):
    ollama = "ollama"
    openai = "openai"
    rits = "rits"


RE_JINJA_VAR = re.compile(r"\{\{\s*(.*?)\s*\}\}")


def decompose(
    task_prompt: str,
    user_input_variable: list[str] | None = None,
    model_id: str = "mistral-small3.2:latest",
    backend: DecompBackend = DecompBackend.ollama,
    backend_req_timeout: int = 300,
    backend_endpoint: str | None = None,
    backend_api_key: str | None = None,
) -> DecompPipelineResult:
    if user_input_variable is None:
        user_input_variable = []

    # region Backend Assignment
    match backend:
        case DecompBackend.ollama:
            m_session = MelleaSession(
                OllamaModelBackend(
                    model_id=model_id,
                    base_url=backend_endpoint,
                    model_options={ModelOption.CONTEXT_WINDOW: 16384},
                )
            )
        case DecompBackend.openai:
            assert backend_endpoint is not None, (
                'Required to provide "backend_endpoint" for this configuration'
            )
            assert backend_api_key is not None, (
                'Required to provide "backend_api_key" for this configuration'
            )
            m_session = MelleaSession(
                OpenAIBackend(
                    model_id=model_id,
                    base_url=backend_endpoint,
                    api_key=backend_api_key,
                    model_options={"timeout": backend_req_timeout},
                )
            )
        case DecompBackend.rits:
            assert backend_endpoint is not None, (
                'Required to provide "backend_endpoint" for this configuration'
            )
            assert backend_api_key is not None, (
                'Required to provide "backend_api_key" for this configuration'
            )

            from mellea_ibm.rits import RITSBackend, RITSModelIdentifier  # type: ignore

            m_session = MelleaSession(
                RITSBackend(
                    RITSModelIdentifier(endpoint=backend_endpoint, model_name=model_id),
                    api_key=backend_api_key,
                    model_options={"timeout": backend_req_timeout},
                )
            )
    # endregion

    subtasks: list[SubtaskItem] = subtask_list.generate(m_session, task_prompt).parse()

    task_prompt_constraints: list[str] = constraint_extractor.generate(
        m_session, task_prompt, enforce_same_words=False
    ).parse()

    constraint_validation_strategies: dict[str, Literal["code", "llm"]] = {
        cons_key: validation_decision.generate(m_session, cons_key).parse() or "llm"
        for cons_key in task_prompt_constraints
    }

    constraint_val_data: dict[str, ConstraintValData] = {}

    for cons_key in constraint_val_strategy:
        constraint_val_data[cons_key] = {
            "val_strategy": constraint_val_strategy[cons_key]["val_strategy"],
            "val_fn": None,
        }
        if constraint_val_data[cons_key]["val_strategy"] == "code":
            constraint_val_data[cons_key]["val_fn"] = (
                validation_code_generator.generate(m_session, cons_key).parse()
            )

    subtask_prompts: list[SubtaskPromptItem] = subtask_prompt_generator.generate(
        m_session,
        task_prompt,
        user_input_var_names=user_input_variable,
        subtasks_and_tags=subtasks,
    ).parse()

    subtask_prompts_with_constraints: list[SubtaskPromptConstraintsItem] = (
        subtask_constraint_assign.generate(
            m_session,
            subtasks_tags_and_prompts=subtask_prompts,
            constraint_list=task_prompt_constraints,
        ).parse()
    )

    decomp_subtask_result: list[DecompSubtasksResult] = [
        DecompSubtasksResult(
            subtask=subtask_data.subtask,
            tag=subtask_data.tag,
            constraints=[
                {
                    "constraint": cons_str,
                    "val_strategy": constraint_val_data[cons_str]["val_strategy"],
                    "val_fn_name": f"val_fn_{task_prompt_constraints.index(cons_str) + 1}",
                    # Always include generated "val_fn" code (experimental)
                    "val_fn": constraint_val_data[cons_str]["val_fn"],
                    # Include generated "val_fn" code only for the last subtask (experimental)
                    # "val_fn": constraint_val_data[cons_str]["val_fn"]
                    # if subtask_i + 1 == len(subtask_prompts_with_constraints)
                    # else None,
                }
                for cons_str in subtask_data.constraints
            ],
            prompt_template=subtask_data.prompt_template,
            general_instructions=general_instructions.generate(
                m_session, input_str=subtask_data.prompt_template
            ).parse(),
            input_vars_required=list(
                dict.fromkeys(  # Remove duplicates while preserving the original order.
                    [
                        item
                        for item in re.findall(
                            RE_JINJA_VAR, subtask_data.prompt_template
                        )
                        if item in user_input_variable
                    ]
                )
            ),
            depends_on=list(
                dict.fromkeys(  # Remove duplicates while preserving the original order.
                    [
                        item
                        for item in re.findall(
                            RE_JINJA_VAR, subtask_data.prompt_template
                        )
                        if item not in user_input_variable
                    ]
                )
            ),
        )
        for subtask_i, subtask_data in enumerate(subtask_prompts_with_constraints)
    ]

    return DecompPipelineResult(
        original_task_prompt=task_prompt,
        subtask_list=[item.subtask for item in subtasks],
        identified_constraints=[
            {
                "constraint": cons_str,
                "val_strategy": constraint_val_data[cons_str]["val_strategy"],
                "val_fn": constraint_val_data[cons_str]["val_fn"],
                "val_fn_name": f"val_fn_{cons_i + 1}",
            }
            for cons_i, cons_str in enumerate(task_prompt_constraints)
        ],
        subtasks=decomp_subtask_result,
    )
