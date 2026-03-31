# decompose/pipeline.py
"""Core decomposition pipeline for turning a task query into scheduled subtasks.

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
    """Validation metadata associated with a single extracted constraint.

    Attributes:
        val_strategy: Validation mode selected for the constraint. ``"code"``
            means the pipeline generated validation code; ``"llm"`` means the
            constraint should be checked by model reasoning rather than code.
        val_fn: Generated validation function source code when ``val_strategy``
            is ``"code"``; otherwise ``None``.
    """

    val_strategy: Literal["code", "llm"]
    val_fn: str | None


class ConstraintResult(TypedDict):
    """A single constraint paired with its validation metadata.

    Attributes:
        constraint: Natural-language description of the constraint.
        val_strategy: Validation mode assigned to the constraint.
        val_fn: Generated validation function source code when the constraint
            uses code-based validation; otherwise ``None``.
        val_fn_name: Stable function name assigned to the generated validation
            function in serialized output.
    """

    constraint: str
    val_strategy: Literal["code", "llm"]
    val_fn: str | None
    val_fn_name: str


class DecompSubtasksResult(TypedDict):
    """Structured decomposition data for one subtask.

    Attributes:
        subtask: Natural-language description of the subtask.
        tag: Short identifier for the subtask, used as a variable name in Jinja2
            templates and dependency references.
        constraints: Constraints assigned to this subtask, each with validation
            metadata.
        prompt_template: Jinja2 prompt template for this subtask, with
            ``{{ variable }}`` placeholders for user inputs and prior subtask
            results.
        general_instructions: Additional general instructions derived from the
            prompt template for this subtask.
        input_vars_required: Ordered list of user-provided input variable names
            referenced in ``prompt_template``.
        depends_on: Ordered list of subtask tags whose results are referenced in
            ``prompt_template``.
        generated_response: Model response produced during execution. This field
            is absent until the subtask has been run.
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
    """Complete output produced by one decomposition run.

    Attributes:
        original_task_prompt: Raw task prompt provided by the user.
        subtask_list: Ordered list of subtask descriptions produced during the
            decomposition stage.
        identified_constraints: Constraints extracted from the original task
            prompt, each with validation metadata.
        subtasks: Fully annotated subtask objects with prompt templates,
            assigned constraints, general instructions, and dependency
            information.
        final_response: Aggregated final response produced during execution.
            This field is absent until the pipeline execution stage is run.
    """

    original_task_prompt: str
    subtask_list: list[str]
    identified_constraints: list[ConstraintResult]
    subtasks: list[DecompSubtasksResult]
    final_response: NotRequired[str]


class DecompBackend(StrEnum):
    """Inference backends supported by the decomposition pipeline.

    Attributes:
        ollama: Local Ollama inference server backend.
        openai: OpenAI-compatible HTTP API backend.
        rits: IBM RITS remote inference backend.
    """

    ollama = "ollama"
    openai = "openai"
    rits = "rits"


RE_JINJA_VAR = re.compile(r"\{\{\s*(.*?)\s*\}\}")


def _dedupe_keep_order(items: list[str]) -> list[str]:
    """Removes duplicates from a list while preserving first-seen order."""
    return list(dict.fromkeys(items))


def _extract_jinja_vars(prompt_template: str) -> list[str]:
    """Extracts raw Jinja variable names referenced in a prompt template.

    Args:
        prompt_template: Prompt template containing zero or more Jinja
            expressions such as ``{{ variable }}``.

    Returns:
        A list of variable expressions captured from the template in match
        order. Duplicates are preserved.
    """
    return re.findall(RE_JINJA_VAR, prompt_template)


def _preview_text(text: str, max_len: int = 240) -> str:
    """Normalizes text whitespace and truncates it for logging preview.

    Args:
        text: Input text to normalize and preview.
        max_len: Maximum preview length in characters before truncation.

    Returns:
        A single-line preview string. Returns the normalized full text when its
        length does not exceed ``max_len``; otherwise returns a truncated
        preview ending with ``" ..."``.
    """
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
    """Builds a model session for the configured inference backend.

    Initializes and returns a ``MelleaSession`` backed by Ollama,
    an OpenAI-compatible endpoint, or RITS, depending on ``backend``.

    Args:
        model_id: Model identifier passed to the selected backend.
        backend: Backend type to use for model inference.
        backend_req_timeout: Request timeout in seconds for backend calls.
        backend_endpoint: Base URL or endpoint required by remote backends.
        backend_api_key: API key required by remote backends.
        log_mode: Logging verbosity for pipeline execution.

    Returns:
        A configured ``MelleaSession`` ready for prompt generation calls.

    Raises:
        AssertionError: If ``backend`` is ``DecompBackend.openai`` or
            ``DecompBackend.rits`` and the required endpoint or API key is not
            provided.
        ImportError: If the RITS backend is selected but the required RITS
            package is not installed.
    """
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
    """Extracts subtasks and top-level constraints from a task prompt.

    This stage runs two prompt modules: one to produce an ordered subtask list
    and one to extract task-level constraints from the original prompt.

    Args:
        m_session: Active model session used to run decomposition prompts.
        task_prompt: Natural-language task description to decompose.
        log_mode: Logging verbosity for this stage.

    Returns:
        A tuple ``(subtasks, task_constraints)``, where ``subtasks`` is the
        ordered list of parsed ``SubtaskItem`` objects and ``task_constraints``
        is the ordered list of extracted natural-language constraint strings.
    """
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
    """Selects a validation mode for each constraint and generates code when needed.

    For every extracted constraint, this stage chooses a validation strategy.
    When the selected strategy is ``"code"``, it also generates a validation
    function body for later serialization or execution.

    Args:
        m_session: Active model session used to run validation prompts.
        task_constraints: Ordered list of extracted natural-language
            constraints.
        log_mode: Logging verbosity for this stage.

    Returns:
        A mapping from each constraint string to its validation metadata,
        including the selected strategy and optional generated validation code.
    """
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
    """Generates per-subtask prompt templates and assigns constraints to them.

    This stage first generates a prompt template for each subtask, then asks the
    model to assign extracted task constraints to the subtasks they apply to.
    Constraint assignment is retried up to two times before failing.

    Args:
        m_session: Active model session used to run prompt-generation and
            constraint-assignment prompts.
        task_prompt: Original task description being decomposed.
        user_input_variable: User-provided input variable names that may appear
            in generated Jinja prompt templates.
        subtasks: Parsed subtasks produced by the decomposition stage.
        task_constraints: Ordered list of extracted task constraints.
        log_mode: Logging verbosity for this stage.

    Returns:
        A list of ``SubtaskPromptConstraintsItem`` objects containing each
        subtask, its generated prompt template, and the constraints assigned to
        that subtask.

    Raises:
        Exception: Re-raises the last exception produced by
            ``subtask_constraint_assign`` after all retry attempts fail.
        RuntimeError: If constraint assignment fails without preserving a final
            exception object.
    """
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
    for i, prompt_item in enumerate(subtask_prompts, start=1):
        logger.info("  [%02d] tag=%s", i, prompt_item.tag)
        if log_mode == LogMode.debug:
            logger.debug("       prompt_template=%s", prompt_item.prompt_template)

    subtasks_tags_and_prompts: list[tuple[str, str, str]] = [
        (prompt_item.subtask, prompt_item.tag, prompt_item.prompt_template)
        for prompt_item in subtask_prompts
    ]

    logger.info("assigning constraints to subtasks")
    logger.info("  total subtasks   : %d", len(subtasks_tags_and_prompts))
    logger.info("  total constraints: %d", len(task_constraints))

    if log_mode == LogMode.debug:
        for i, (subtask, tag, prompt_template) in enumerate(
            subtasks_tags_and_prompts, start=1
        ):
            logger.debug(
                "  subtask_input[%02d]: subtask=%s | tag=%s | prompt=%s",
                i,
                subtask,
                tag,
                prompt_template,
            )
        for i, cons in enumerate(task_constraints, start=1):
            logger.debug("  constraint[%02d]: %s", i, cons)

    retry_count = 2
    last_exc: Exception | None = None

    for attempt in range(1, retry_count + 1):
        try:
            logger.info(
                "subtask_constraint_assign attempt: %d/%d", attempt, retry_count
            )

            subtask_prompts_with_constraints: list[SubtaskPromptConstraintsItem] = (
                subtask_constraint_assign.generate(
                    m_session,
                    subtasks_tags_and_prompts=subtasks_tags_and_prompts,
                    constraint_list=task_constraints,
                ).parse()
            )

            if log_mode == LogMode.debug:
                logger.debug(
                    "parsed subtask_constraint_assign result:\n%s",
                    subtask_prompts_with_constraints,
                )
            else:
                preview_lines: list[str] = []
                for constraint_item in subtask_prompts_with_constraints[:3]:
                    preview_lines.append(
                        f"[{constraint_item.tag}] constraints={len(constraint_item.constraints)}"
                    )
                if len(subtask_prompts_with_constraints) > 3:
                    preview_lines.append("...")
                preview = "\n".join(preview_lines)
                logger.info("parsed subtask_constraint_assign preview:\n%s", preview)

            logger.info(
                "constraint assignment completed: %d",
                len(subtask_prompts_with_constraints),
            )
            for i, constraint_item in enumerate(
                subtask_prompts_with_constraints, start=1
            ):
                logger.info(
                    "  [%02d] tag=%s | assigned_constraints=%d",
                    i,
                    constraint_item.tag,
                    len(constraint_item.constraints),
                )
                if log_mode == LogMode.debug:
                    for cons in constraint_item.constraints:
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

    logger.error("subtask_constraint_assign failed after %d attempts", retry_count)
    raise (
        last_exc
        if last_exc is not None
        else RuntimeError("subtask_constraint_assign failed with unknown error")
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
    """Builds the final structured pipeline result from intermediate outputs.

    This stage resolves Jinja dependencies for each subtask, attaches validation
    metadata to assigned constraints, generates general instructions from each
    prompt template, and assembles the final ``DecompPipelineResult``.

    Args:
        m_session: Active model session used to generate general instructions.
        task_prompt: Original task description provided by the user.
        user_input_variable: User-provided input variable names that should be
            treated as external inputs rather than subtask dependencies.
        subtasks: Parsed subtasks produced by the decomposition stage.
        task_constraints: Ordered list of extracted task constraints.
        constraint_val_data: Validation metadata for each extracted constraint.
        subtask_prompts_with_constraints: Prompt templates with per-subtask
            constraint assignments.
        log_mode: Logging verbosity for this stage.

    Returns:
        A ``DecompPipelineResult`` containing the original prompt, extracted
        constraints, and fully annotated subtasks with dependency and validation
        information.
    """
    logger = get_logger("m_decompose.finalize_result")
    log_section(logger, "finalize_result")

    decomp_subtask_result: list[DecompSubtasksResult] = []

    for subtask_i, subtask_data in enumerate(subtask_prompts_with_constraints, start=1):
        jinja_vars = _extract_jinja_vars(subtask_data.prompt_template)

        input_vars_required = _dedupe_keep_order(
            [var_name for var_name in jinja_vars if var_name in user_input_variable]
        )
        depends_on = _dedupe_keep_order(
            [var_name for var_name in jinja_vars if var_name not in user_input_variable]
        )

        logger.info("finalizing subtask [%02d] tag=%s", subtask_i, subtask_data.tag)
        logger.info("  input_vars_required: %s", input_vars_required or "[]")
        logger.info("  depends_on         : %s", depends_on or "[]")

        if log_mode == LogMode.debug:
            logger.debug("  prompt_template=%s", subtask_data.prompt_template)

        subtask_constraints: list[ConstraintResult] = [
            {
                "constraint": cons_str,
                "val_strategy": constraint_val_data[cons_str]["val_strategy"],
                "val_fn_name": f"val_fn_{task_constraints.index(cons_str) + 1}",
                "val_fn": constraint_val_data[cons_str]["val_fn"],
            }
            for cons_str in subtask_data.constraints
        ]

        parsed_general_instructions: str = general_instructions.generate(
            m_session, input_str=subtask_data.prompt_template
        ).parse()

        if log_mode == LogMode.debug:
            logger.debug("  general_instructions=%s", parsed_general_instructions)

        subtask_result: DecompSubtasksResult = DecompSubtasksResult(
            subtask=subtask_data.subtask,
            tag=subtask_data.tag,
            constraints=subtask_constraints,
            prompt_template=subtask_data.prompt_template,
            general_instructions=parsed_general_instructions,
            input_vars_required=input_vars_required,
            depends_on=depends_on,
        )

        decomp_subtask_result.append(subtask_result)

    result = DecompPipelineResult(
        original_task_prompt=task_prompt,
        subtask_list=[subtask_item.subtask for subtask_item in subtasks],
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
    """Breaks a task prompt into structured subtasks with a staged LLM pipeline.

    Orchestrates sequential prompt calls for subtask listing, constraint
    extraction, validation strategy selection, prompt-template generation, and
    per-subtask constraint assignment. The total number of model calls depends
    in part on how many constraints are extracted from the original task.

    Args:
        task_prompt: Natural-language description of the task to decompose.
        user_input_variable: Optional list of user input variable names that may
            be referenced in generated Jinja prompt templates. Pass ``None`` or
            an empty list when the task requires no external input variables.
        model_id: Model name or identifier used for all pipeline stages.
        backend: Inference backend used for all model calls.
        backend_req_timeout: Request timeout in seconds for backend inference
            calls.
        backend_endpoint: Endpoint URL required by remote backends such as
            OpenAI-compatible APIs and RITS.
        backend_api_key: API key required by remote backends.
        log_mode: Logging verbosity for the pipeline run.

    Returns:
        A ``DecompPipelineResult`` containing the original prompt, extracted
        subtask list, identified constraints, and fully annotated subtask
        records.

    Raises:
        AssertionError: If a selected remote backend is missing a required
            endpoint or API key.
        ImportError: If the RITS backend is selected but its dependency is not
            installed.
        Exception: Propagates backend, generation, parsing, or constraint
            assignment failures raised by lower-level pipeline stages.
    """

    configure_logging(log_mode)
    logger = get_logger("m_decompose.pipeline")
    log_section(logger, "m_decompose pipeline")

    if user_input_variable is None:
        user_input_variable = []

    logger.info("log_mode       : %s", log_mode.value)
    logger.info("user_input_vars: %s", user_input_variable or "[]")

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