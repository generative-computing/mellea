"""Core decomposition pipeline that breaks a task prompt into structured subtasks.

Provides the `decompose()` function, which orchestrates a series of LLM calls
(subtask listing, constraint extraction, validation strategy selection, prompt
generation, and constraint assignment) to produce a `DecompPipelineResult`
containing subtasks, per-subtask prompts, constraints, and dependency information.
Supports Ollama, OpenAI-compatible, and RITS inference backends.
"""

import re
from enum import StrEnum
from typing import Literal, NotRequired, TypedDict

from mellea import MelleaSession
from mellea.backends import ModelOption
from mellea.backends.ollama import OllamaModelBackend
from mellea.backends.openai import OpenAIBackend

from .prompt_modules import (
    constraint_extractor,
    # general_instructions,
    subtask_constraint_assign,
    subtask_list,
    subtask_prompt_generator,
    validation_decision,
)
from .prompt_modules.subtask_constraint_assign import SubtaskPromptConstraintsItem
from .prompt_modules.subtask_list import SubtaskItem
from .prompt_modules.subtask_prompt_generator import SubtaskPromptItem


class ConstraintResult(TypedDict):
    """A single constraint paired with its assigned validation strategy.

    Attributes:
        constraint (str): Natural-language description of the constraint.
        validation_strategy (str): Strategy assigned to validate the constraint;
            either `"code"` or `"llm"`.
    """

    constraint: str
    validation_strategy: str


class DecompSubtasksResult(TypedDict):
    """The full structured result for one decomposed subtask.

    Attributes:
        subtask (str): Natural-language description of the subtask.
        tag (str): Short identifier for the subtask, used as a variable name
            in Jinja2 templates and dependency references.
        constraints (list[ConstraintResult]): List of constraints assigned to
            this subtask, each with a validation strategy.
        prompt_template (str): Jinja2 prompt template string for this subtask,
            with `{{ variable }}` placeholders for inputs and prior subtask results.
        input_vars_required (list[str]): Ordered list of user-provided input
            variable names referenced in `prompt_template`.
        depends_on (list[str]): Ordered list of subtask tags whose results are
            referenced in `prompt_template`.
        generated_response (str): Optional field holding the model response
            produced during execution; not present until the subtask runs.
    """

    subtask: str
    tag: str
    constraints: list[ConstraintResult]
    prompt_template: str
    # general_instructions: str
    input_vars_required: list[str]
    depends_on: list[str]
    generated_response: NotRequired[str]


class DecompPipelineResult(TypedDict):
    """The complete output of a decomposition pipeline run.

    Attributes:
        original_task_prompt (str): The raw task prompt provided by the user.
        subtask_list (list[str]): Ordered list of subtask descriptions produced
            by the subtask-listing stage.
        identified_constraints (list[ConstraintResult]): Constraints extracted
            from the original task prompt, each with a validation strategy.
        subtasks (list[DecompSubtasksResult]): Fully annotated subtask objects
            with prompt templates, constraint assignments, and dependency
            information.
        final_response (str): Optional field holding the aggregated final
            response produced during execution; not present until the pipeline runs.
    """

    original_task_prompt: str
    subtask_list: list[str]
    identified_constraints: list[ConstraintResult]
    subtasks: list[DecompSubtasksResult]
    final_response: NotRequired[str]


class DecompBackend(StrEnum):
    """Inference backends supported by the decomposition pipeline.

    Attributes:
        ollama (str): Local Ollama inference server backend.
        openai (str): Any OpenAI-compatible HTTP endpoint backend.
        rits (str): IBM RITS (Remote Inference and Training Service) backend.
    """

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
    """Break a task prompt into structured subtasks using a multi-step LLM pipeline.

    Orchestrates a series of sequential LLM calls to produce a fully structured
    decomposition: subtask listing, constraint extraction, validation strategy
    selection, prompt template generation, and per-subtask constraint assignment.
    The number of calls depends on the number of constraints extracted.

    Args:
        task_prompt: Natural-language description of the task to decompose.
        user_input_variable: Optional list of variable names that will be
            templated into generated prompts as user-provided input data. Pass
            `None` or an empty list if the task requires no input variables.
        model_id: Model name or ID used for all pipeline steps.
        backend: Inference backend -- `"ollama"`, `"openai"`, or `"rits"`.
        backend_req_timeout: Request timeout in seconds for model inference calls.
        backend_endpoint: Base URL of the OpenAI-compatible endpoint. Required
            when `backend` is `"openai"` or `"rits"`.
        backend_api_key: API key for the configured endpoint. Required when
            `backend` is `"openai"` or `"rits"`.

    Returns:
        A `DecompPipelineResult` containing the original prompt, subtask list,
        identified constraints, and fully annotated subtask objects with prompt
        templates, constraint assignments, and dependency information.
    """
    if user_input_variable is None:
        user_input_variable = []

    # region Backend Assignment
    match backend:
        case DecompBackend.ollama:
            m_session = MelleaSession(
                OllamaModelBackend(
                    model_id=model_id, model_options={ModelOption.CONTEXT_WINDOW: 16384}
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
                    "validation_strategy": constraint_validation_strategies.get(
                        cons_str, "llm"
                    ),
                }
                for cons_str in subtask_data.constraints
            ],
            prompt_template=subtask_data.prompt_template,
            # general_instructions=general_instructions.generate(
            #     m_session, input_str=subtask_data.prompt_template
            # ).parse(),
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
        for subtask_data in subtask_prompts_with_constraints
    ]

    return DecompPipelineResult(
        original_task_prompt=task_prompt,
        subtask_list=[item.subtask for item in subtasks],
        identified_constraints=[
            {
                "constraint": cons_str,
                "validation_strategy": constraint_validation_strategies[cons_str],
            }
            for cons_str in task_prompt_constraints
        ],
        subtasks=decomp_subtask_result,
    )
