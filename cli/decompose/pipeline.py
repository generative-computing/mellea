# decompose/pipeline.py
"""Core decomposition pipeline that breaks a task prompt into structured subtasks.

Provides the ``decompose()`` function, which orchestrates a series of LLM calls
(subtask listing, constraint extraction, validation strategy selection, prompt
generation, and constraint assignment) to produce a ``DecompPipelineResult``
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
    """A single constraint paired with its assigned validation strategy.

    Args:
        constraint (str): Natural-language description of the constraint.
        validation_strategy (str): Strategy assigned to validate the constraint;
            either ``"code"`` or ``"llm"``.
    """

    constraint: str
    val_strategy: Literal["code", "llm"]
    val_fn: str | None
    val_fn_name: str


class DecompSubtasksResult(TypedDict):
    """The full structured result for one decomposed subtask.

    Args:
        subtask (str): Natural-language description of the subtask.
        tag (str): Short identifier for the subtask, used as a variable name
            in Jinja2 templates and dependency references.
        constraints (list[ConstraintResult]): List of constraints assigned to
            this subtask, each with a validation strategy.
        prompt_template (str): Jinja2 prompt template string for this subtask,
            with ``{{ variable }}`` placeholders for inputs and prior subtask results.
        input_vars_required (list[str]): Ordered list of user-provided input
            variable names referenced in ``prompt_template``.
        depends_on (list[str]): Ordered list of subtask tags whose results are
            referenced in ``prompt_template``.
        generated_response (str): Optional field holding the model response
            produced during execution; not present until the subtask runs.
    """

    subtask: str
    tag: str
    constraints: list[ConstraintResult]
    prompt_template: str
    general_instructions: str
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

    Args:
        ollama (str): Local Ollama inference server backend.
        openai (str): Any OpenAI-compatible HTTP endpoint backend.
        rits (str): IBM RITS (Remote Inference and Training Service) backend.
    """

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
    log_mode: LogMode = LogMode.demo,
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
    m_session: MelleaSession, task_prompt: str, log_mode: LogMode = LogMode.demo
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
    log_mode: LogMode = LogMode.demo,
) -> dict[str, ConstraintValData]:
    logger = get_logger("m_decompose.constraint_validate")
    log_section(logger, "constraint_validate")

    constraint_val_data: dict[str, ConstraintValData] = {}

    for idx, cons_key in enumerate(task_constraints, start=1):
        logger.info("constraint [%02d]: %s", idx, cons_key)

        val_strategy: Literal["code", "llm"] = (
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
    log_mode: LogMode = LogMode.demo,
) -> list[SubtaskPromptConstraintsItem]:
    logger = get_logger("m_decompose.task_execute")
    log_section(logger, "task_execute")

    subtask_prompts: list[SubtaskPromptItem] = subtask_prompt_generator.generate(
        m_session,
        task_prompt,
        user_input_var_names=user_input_variable,
        subtasks_and_tags=subtasks,
    ).parse()

    for i, item in enumerate(subtask_prompts, start=1):
        logger.info("  [%02d] tag=%s", i, item.tag)
        logger.debug("       prompt_template=%s", item.prompt_template)

    logger.info("assigning constraints to subtasks")
    logger.info("  total subtasks   : %d", len(subtask_prompts))
    logger.info("  total constraints: %d", len(task_constraints))

    if log_mode == LogMode.debug:
        for i, item in enumerate(subtask_prompts, start=1):
            logger.debug(
                "  subtask_input[%02d]: subtask=%s | tag=%s | prompt=%s",
                i,
                item.subtask,
                item.tag,
                item.prompt_template,
            )
        for i, cons in enumerate(task_constraints, start=1):
            logger.debug("  constraint[%02d]: %s", i, cons)

    retry_count = 2
    last_exc: Exception | None = None

    for attempt in range(1, retry_count + 1):
        raw_assign_result = None
        raw_text = ""

        try:
            logger.info(
                "subtask_constraint_assign attempt: %d/%d",
                attempt,
                retry_count,
            )

            raw_assign_result = subtask_constraint_assign.generate(
                m_session,
                subtasks_tags_and_prompts=subtask_prompts,
                constraint_list=task_constraints,
                max_new_tokens=8192,
            )
            raw_text = str(raw_assign_result)

            if log_mode == LogMode.debug:
                logger.debug("raw subtask_constraint_assign output:\n%s", raw_text)
            else:
                preview = raw_text[:800] + (" ..." if len(raw_text) > 800 else "")
                logger.info("raw subtask_constraint_assign preview:\n%s", preview)

            subtask_prompts_with_constraints = raw_assign_result.parse()

            logger.info(
                "constraint assignment completed: %d",
                len(subtask_prompts_with_constraints),
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

        except Exception as exc:
            last_exc = exc
            logger.warning(
                "subtask_constraint_assign failed on attempt %d/%d: %s",
                attempt,
                retry_count,
                exc,
            )

            if raw_assign_result is not None:
                if log_mode == LogMode.debug:
                    logger.debug("failed raw output:\n%s", raw_text)
                else:
                    preview = raw_text[:800] + (" ..." if len(raw_text) > 800 else "")
                    logger.info("failed raw output preview:\n%s", preview)

    logger.error("subtask_constraint_assign failed after %d attempts", retry_count)
    raise last_exc if last_exc is not None else RuntimeError(
        "subtask_constraint_assign failed with unknown error"
    )


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
    log_mode: LogMode = LogMode.demo,
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
            for cons_str in subtask_data.constraints or []
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
    """Break a task prompt into structured subtasks using a multi-step LLM pipeline.

    Orchestrates a series of sequential LLM calls to produce a fully structured
    decomposition: subtask listing, constraint extraction, validation strategy
    selection, prompt template generation, and per-subtask constraint assignment.
    The number of calls depends on the number of constraints extracted.

    Args:
        task_prompt: Natural-language description of the task to decompose.
        user_input_variable: Optional list of variable names that will be
            templated into generated prompts as user-provided input data. Pass
            ``None`` or an empty list if the task requires no input variables.
        model_id: Model name or ID used for all pipeline steps.
        backend: Inference backend -- ``"ollama"``, ``"openai"``, or ``"rits"``.
        backend_req_timeout: Request timeout in seconds for model inference calls.
        backend_endpoint: Base URL of the OpenAI-compatible endpoint. Required
            when ``backend`` is ``"openai"`` or ``"rits"``.
        backend_api_key: API key for the configured endpoint. Required when
            ``backend`` is ``"openai"`` or ``"rits"``.
        log_mode: Mode of logging detail to emit during pipeline execution.

    Returns:
        A ``DecompPipelineResult`` containing the original prompt, subtask list,
        identified constraints, and fully annotated subtask objects with prompt
        templates, constraint assignments, and dependency information.
    """

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
