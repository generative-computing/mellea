# decompose/pipeline.py
import re
from enum import StrEnum
from typing import Literal, NotRequired, TypedDict

from mellea import MelleaSession
from mellea.backends import ModelOption
from mellea.backends.ollama import OllamaModelBackend
from mellea.backends.openai import OpenAIBackend

from .logging import LogMode, configure_logging, get_logger, log_section
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


def _dedupe_keep_order(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _extract_jinja_vars(prompt_template: str) -> list[str]:
    return re.findall(RE_JINJA_VAR, prompt_template)


def _preview_text(text: str, max_len: int = 240) -> str:
    text = " ".join(text.strip().split())
    if len(text) <= max_len:
        return text
    return text[:max_len] + " ..."


# -------------------------------------------------------------------
# backend
# -------------------------------------------------------------------
def build_backend_session(
    model_id: str = "mistral-small3.2:latest",
    backend: DecompBackend = DecompBackend.ollama,
    backend_req_timeout: int = 300,
    backend_endpoint: str | None = None,
    backend_api_key: str | None = None,
    log_mode: LogMode = LogMode.demo,  # kept for signature compatibility
) -> MelleaSession:
    logger = get_logger("m_decompose.backend")
    log_section(logger, "backend")

    logger.info("backend      : %s", backend.value)
    logger.info("model_id     : %s", model_id)
    logger.info("timeout      : %s", backend_req_timeout)
    if backend_endpoint:
        logger.info("endpoint     : %s", backend_endpoint)

    match backend:
        case DecompBackend.ollama:
            logger.info("initializing Ollama backend")
            session = MelleaSession(
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

            logger.info("initializing OpenAI-compatible backend")
            session = MelleaSession(
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

            logger.info("initializing RITS backend")
            from mellea_ibm.rits import RITSBackend, RITSModelIdentifier  # type: ignore

            session = MelleaSession(
                RITSBackend(
                    RITSModelIdentifier(endpoint=backend_endpoint, model_name=model_id),
                    api_key=backend_api_key,
                    model_options={"timeout": backend_req_timeout},
                )
            )

    logger.info("backend session ready")
    return session


# -------------------------------------------------------------------
# task_decompose
# -------------------------------------------------------------------
def task_decompose(
    m_session: MelleaSession,
    task_prompt: str,
    log_mode: LogMode = LogMode.demo,  # kept for compatibility
) -> tuple[list[SubtaskItem], list[str]]:
    logger = get_logger("m_decompose.task_decompose")
    log_section(logger, "task_decompose")

    logger.info("generating subtask list")
    subtasks: list[SubtaskItem] = subtask_list.generate(m_session, task_prompt).parse()

    logger.info("subtasks found: %d", len(subtasks))
    for i, item in enumerate(subtasks, start=1):
        logger.info("  [%02d] tag=%s | subtask=%s", i, item.tag, item.subtask)

    logger.info("extracting task constraints")
    task_constraints: list[str] = constraint_extractor.generate(
        m_session, task_prompt, enforce_same_words=False
    ).parse()

    logger.info("constraints found: %d", len(task_constraints))
    for i, cons in enumerate(task_constraints, start=1):
        logger.info("  [%02d] %s", i, cons)

    return subtasks, task_constraints


# -------------------------------------------------------------------
# constraint_validate
# -------------------------------------------------------------------
def constraint_validate(
    m_session: MelleaSession,
    task_constraints: list[str],
    log_mode: LogMode = LogMode.demo,  # kept for compatibility
) -> dict[str, ConstraintValData]:
    logger = get_logger("m_decompose.constraint_validate")
    log_section(logger, "constraint_validate")

    constraint_val_data: dict[str, ConstraintValData] = {}

    for idx, cons_key in enumerate(task_constraints, start=1):
        logger.info("constraint [%02d]: %s", idx, cons_key)

        val_strategy = (
            validation_decision.generate(m_session, cons_key).parse() or "llm"
        )
        logger.info("  strategy: %s", val_strategy)

        val_fn: str | None = None
        if val_strategy == "code":
            logger.info("  generating validation code")
            val_fn = validation_code_generator.generate(m_session, cons_key).parse()
            logger.debug("  generated val_fn length: %d", len(val_fn) if val_fn else 0)
        else:
            logger.info("  validation mode: llm")

        constraint_val_data[cons_key] = {"val_strategy": val_strategy, "val_fn": val_fn}

    return constraint_val_data


# -------------------------------------------------------------------
# task_execute
# -------------------------------------------------------------------
def task_execute(
    m_session: MelleaSession,
    task_prompt: str,
    user_input_variable: list[str],
    subtasks: list[SubtaskItem],
    task_constraints: list[str],
    log_mode: LogMode = LogMode.demo,  # kept for compatibility
) -> list[SubtaskPromptConstraintsItem]:
    logger = get_logger("m_decompose.task_execute")
    log_section(logger, "task_execute")

    logger.info("generating prompt templates for subtasks")
    subtask_prompts: list[SubtaskPromptItem] = subtask_prompt_generator.generate(
        m_session,
        task_prompt,
        user_input_var_names=user_input_variable,
        subtasks_and_tags=subtasks,
    ).parse()

    logger.info("subtask prompt templates generated: %d", len(subtask_prompts))
    for i, item in enumerate(subtask_prompts, start=1):
        logger.info("  [%02d] tag=%s", i, item.tag)
        logger.debug("       prompt_template=%s", item.prompt_template)

    logger.info("assigning constraints to subtasks")
    subtask_prompts_with_constraints: list[SubtaskPromptConstraintsItem] = (
        subtask_constraint_assign.generate(
            m_session,
            subtasks_tags_and_prompts=subtask_prompts,
            constraint_list=task_constraints,
        ).parse()
    )

    logger.info(
        "constraint assignment completed: %d", len(subtask_prompts_with_constraints)
    )
    for i, item in enumerate(subtask_prompts_with_constraints, start=1):
        logger.info(
            "  [%02d] tag=%s | assigned_constraints=%d",
            i,
            item.tag,
            len(item.constraints),
        )
        for cons in item.constraints:
            logger.debug("       - %s", cons)

    return subtask_prompts_with_constraints


# -------------------------------------------------------------------
# finalize_result
# -------------------------------------------------------------------
def finalize_result(
    m_session: MelleaSession,
    task_prompt: str,
    user_input_variable: list[str],
    subtasks: list[SubtaskItem],
    task_constraints: list[str],
    constraint_val_data: dict[str, ConstraintValData],
    subtask_prompts_with_constraints: list[SubtaskPromptConstraintsItem],
    log_mode: LogMode = LogMode.demo,  # kept for compatibility
) -> DecompPipelineResult:
    logger = get_logger("m_decompose.finalize_result")
    log_section(logger, "finalize_result")

    decomp_subtask_result: list[DecompSubtasksResult] = []

    for subtask_i, subtask_data in enumerate(subtask_prompts_with_constraints, start=1):
        jinja_vars = _extract_jinja_vars(subtask_data.prompt_template)

        input_vars_required = _dedupe_keep_order(
            [item for item in jinja_vars if item in user_input_variable]
        )
        depends_on = _dedupe_keep_order(
            [item for item in jinja_vars if item not in user_input_variable]
        )

        logger.info("finalizing subtask [%02d] tag=%s", subtask_i, subtask_data.tag)
        logger.info("  input_vars_required: %s", input_vars_required or "[]")
        logger.info("  depends_on         : %s", depends_on or "[]")

        subtask_constraints: list[ConstraintResult] = [
            {
                "constraint": cons_str,
                "val_strategy": constraint_val_data[cons_str]["val_strategy"],
                "val_fn_name": f"val_fn_{task_constraints.index(cons_str) + 1}",
                "val_fn": constraint_val_data[cons_str]["val_fn"],
            }
            for cons_str in subtask_data.constraints
        ]

        subtask_result: DecompSubtasksResult = DecompSubtasksResult(
            subtask=subtask_data.subtask,
            tag=subtask_data.tag,
            constraints=subtask_constraints,
            prompt_template=subtask_data.prompt_template,
            general_instructions=general_instructions.generate(
                m_session, input_str=subtask_data.prompt_template
            ).parse(),
            input_vars_required=input_vars_required,
            depends_on=depends_on,
        )

        logger.debug("  prompt_template=%s", subtask_result["prompt_template"])
        logger.debug(
            "  general_instructions=%s", subtask_result["general_instructions"]
        )

        decomp_subtask_result.append(subtask_result)

    result = DecompPipelineResult(
        original_task_prompt=task_prompt,
        subtask_list=[item.subtask for item in subtasks],
        identified_constraints=[
            {
                "constraint": cons_str,
                "val_strategy": constraint_val_data[cons_str]["val_strategy"],
                "val_fn": constraint_val_data[cons_str]["val_fn"],
                "val_fn_name": f"val_fn_{cons_i + 1}",
            }
            for cons_i, cons_str in enumerate(task_constraints)
        ],
        subtasks=decomp_subtask_result,
    )

    logger.info("pipeline result finalized")
    logger.info("  total_subtasks   : %d", len(result["subtasks"]))
    logger.info("  total_constraints: %d", len(result["identified_constraints"]))
    logger.info("  verify step      : skipped")

    return result


# -------------------------------------------------------------------
# public entry
# -------------------------------------------------------------------
def decompose(
    task_prompt: str,
    user_input_variable: list[str] | None = None,
    model_id: str = "mistral-small3.2:latest",
    backend: DecompBackend = DecompBackend.ollama,
    backend_req_timeout: int = 300,
    backend_endpoint: str | None = None,
    backend_api_key: str | None = None,
    log_mode: LogMode = LogMode.demo,
) -> DecompPipelineResult:
    configure_logging(log_mode)
    logger = get_logger("m_decompose.pipeline")
    log_section(logger, "m_decompose pipeline")

    if user_input_variable is None:
        user_input_variable = []

    logger.info("log_mode       : %s", log_mode.value)
    logger.info("user_input_vars: %s", user_input_variable or "[]")

    if log_mode == LogMode.debug:
        logger.info("user_query     : %s", task_prompt)
    else:
        logger.info("user_query     : %s", _preview_text(task_prompt))

    m_session = build_backend_session(
        model_id=model_id,
        backend=backend,
        backend_req_timeout=backend_req_timeout,
        backend_endpoint=backend_endpoint,
        backend_api_key=backend_api_key,
        log_mode=log_mode,
    )

    subtasks, task_constraints = task_decompose(
        m_session=m_session, task_prompt=task_prompt, log_mode=log_mode
    )

    constraint_val_data = constraint_validate(
        m_session=m_session, task_constraints=task_constraints, log_mode=log_mode
    )

    subtask_prompts_with_constraints = task_execute(
        m_session=m_session,
        task_prompt=task_prompt,
        user_input_variable=user_input_variable,
        subtasks=subtasks,
        task_constraints=task_constraints,
        log_mode=log_mode,
    )

    result = finalize_result(
        m_session=m_session,
        task_prompt=task_prompt,
        user_input_variable=user_input_variable,
        subtasks=subtasks,
        task_constraints=task_constraints,
        constraint_val_data=constraint_val_data,
        subtask_prompts_with_constraints=subtask_prompts_with_constraints,
        log_mode=log_mode,
    )

    logger.info("")
    logger.info("m_decompose pipeline completed successfully")
    return result
