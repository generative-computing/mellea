"""SOFAI (Slow and Fast AI) Sampling Strategy.

This module implements a two-solver sampling strategy that uses:
1. S1 Solver (fast model) - Iterative solving with feedback-based repair
2. S2 Solver (slow model) - Single attempt escalation when S1 fails or shows no improvement

The strategy leverages ValidationResult.reason fields to provide targeted
feedback for repair, enabling more effective iterative improvement.
"""

import re
from copy import deepcopy
from typing import Literal

import tqdm

from ...core import (
    Backend,
    BaseModelSubclass,
    Component,
    Context,
    FancyLogger,
    ModelOutputThunk,
    Requirement,
    S,
    SamplingResult,
    SamplingStrategy,
    TemplateRepresentation,
    ValidationResult,
)
from ...stdlib import functional as mfuncs
from ..components import Message
from ..context import ChatContext


class SOFAISamplingStrategy(SamplingStrategy):
    """SOFAI sampling strategy.

    Uses S1 Solver (fast model) in a loop with targeted feedback from validation results.
    If S1 Solver fails after exhausting the budget or shows no improvement,
    escalates to a single attempt with S2 Solver (slow model).

    The strategy leverages ValidationResult.reason fields to provide targeted
    feedback for repair, enabling more effective iterative improvement.
    """

    def __init__(
        self,
        s1_solver_backend: Backend,
        s2_solver_backend: Backend,
        s2_solver_mode: Literal[
            "fresh_start", "continue_chat", "best_attempt"
        ] = "fresh_start",
        *,
        loop_budget: int = 3,
        judge_backend: Backend | None = None,
        feedback_strategy: Literal["simple", "first_error", "all_errors"] = "simple",
    ):
        """Initialize SOFAI sampling strategy with two solvers.

        Args:
            s1_solver_backend: Backend for S1 Solver (fast model for iterative solving).
            s2_solver_backend: Backend for S2 Solver (slow model for escalation).
            s2_solver_mode: How to invoke S2 Solver:
                - "fresh_start": Same prompt as S1 solver (clean slate)
                - "continue_chat": Fresh start input + S1 iteration/feedback history
                - "best_attempt": Fresh start input + best S1 attempt + its feedback
            loop_budget: Maximum attempts for S1 Solver before falling back to S2 Solver.
            judge_backend: Optional third backend for LLM-as-Judge validation.
                If provided, this backend will be used for validation when no custom
                validation_fn is provided. Priority: validation_fn > judge_backend > session backend.
            feedback_strategy: Control detail level of LLM-as-Judge feedback:
                - "simple": Binary yes/no validation, no detailed feedback (default)
                - "first_error": Provide only the first mistake found with detailed feedback
                - "all_errors": Provide comprehensive feedback about all mistakes
                Note: Only used when judge_backend is provided and requirement has no validation_fn.

        Raises:
            TypeError: If backends are not Backend instances.
            ValueError: If loop_budget is not greater than 0.
        """
        if loop_budget <= 0:
            raise ValueError("Loop budget must be at least 1.")

        if not isinstance(s1_solver_backend, Backend):
            raise TypeError(
                f"s1_solver_backend must be an instance of Backend, got {type(s1_solver_backend)}"
            )
        if not isinstance(s2_solver_backend, Backend):
            raise TypeError(
                f"s2_solver_backend must be an instance of Backend, got {type(s2_solver_backend)}"
            )
        if judge_backend is not None and not isinstance(judge_backend, Backend):
            raise TypeError(
                f"judge_backend must be an instance of Backend or None, got {type(judge_backend)}"
            )

        self.s1_solver_backend = s1_solver_backend
        self.s2_solver_backend = s2_solver_backend
        self.s2_solver_mode = s2_solver_mode
        self.loop_budget = loop_budget
        self.judge_backend = judge_backend
        self.feedback_strategy = feedback_strategy

    @staticmethod
    def repair(
        old_ctx: Context,
        new_ctx: Context,
        past_actions: list[Component],
        past_results: list[ModelOutputThunk],
        past_val: list[list[tuple[Requirement, ValidationResult]]],
    ) -> tuple[Component, Context]:
        """Create targeted feedback message from validation results.

        Extracts failed requirements and uses their ValidationResult.reason fields
        to provide specific, actionable feedback for the next attempt.

        Args:
            old_ctx: The context WITHOUT the last action + output.
            new_ctx: The context including the last action + output.
            past_actions: List of actions executed.
            past_results: List of generation results.
            past_val: List of validation results.

        Returns:
            Tuple of (Message component with repair feedback, new context).
        """
        assert isinstance(new_ctx, ChatContext), (
            "SOFAI requires ChatContext for conversation continuity."
        )

        # Extract failed requirements with their validation reasons
        failed_items = [(req, val) for req, val in past_val[-1] if not val.as_bool()]

        # Build targeted feedback from validation reasons
        feedback_lines = []
        for req, val_result in failed_items:
            if val_result.reason:
                # Use detailed feedback from validator
                feedback_lines.append(f"* {val_result.reason}")
            else:
                # Fallback to requirement description
                feedback_lines.append(f"* {req.description}")

        repair_message = (
            "The previous attempt failed. Please fix the following issues:\n"
            + "\n".join(feedback_lines)
        )

        next_action = Message(role="user", content=repair_message)
        return next_action, new_ctx

    @staticmethod
    def select_from_failure(
        sampled_actions: list[Component],
        sampled_results: list[ModelOutputThunk],
        sampled_val: list[list[tuple[Requirement, ValidationResult]]],
    ) -> int:
        """Select the most informed attempt (last) when all fail.

        Returns the last attempt as it has benefited from the most feedback.

        Args:
            sampled_actions: List of actions that have been executed (without success).
            sampled_results: List of (unsuccessful) generation results for these actions.
            sampled_val: List of validation results for the results.

        Returns:
            The index of the result that should be selected as `.value`.
        """
        return -1

    @staticmethod
    def _select_best_attempt(
        sampled_val: list[list[tuple[Requirement, ValidationResult]]],
    ) -> int:
        """Select attempt with most passed requirements.

        Args:
            sampled_val: List of validation results for each attempt.

        Returns:
            Index of attempt with highest number of passed requirements.
            If tie, returns later attempt (more feedback).
        """
        best_idx = 0
        best_count = -1

        for i, validations in enumerate(sampled_val):
            passing_count = sum(1 for _, val in validations if val.as_bool())
            # Use >= to prefer later attempts when tied (more feedback iterations)
            if passing_count >= best_count:
                best_count = passing_count
                best_idx = i

        return best_idx

    @staticmethod
    def _extract_action_prompt(action: Component) -> str:
        """Extract a human-readable prompt from a Component."""
        if isinstance(action, Message):
            return action.content

        for attr in ("description", "_description"):
            if hasattr(action, attr):
                value = getattr(action, attr)
                if value:
                    return str(value)

        try:
            action_repr = action.format_for_llm()
        except Exception:
            return str(action)

        if isinstance(action_repr, str):
            return action_repr

        if isinstance(action_repr, TemplateRepresentation):
            for key in ("description", "content"):
                value = action_repr.args.get(key)
                if value:
                    return str(value)

        return str(action)

    @staticmethod
    def _parse_judgment(llm_response: str) -> bool:
        """Parse Yes/No judgment from LLM response.

        Args:
            llm_response: The LLM's judgment response.

        Returns:
            True if response indicates success (contains 'yes'), False otherwise.
        """
        response_lower = llm_response.strip().lower()
        # Check for explicit yes/no
        if response_lower.startswith("yes"):
            return True
        if response_lower.startswith("no"):
            return False
        # Check for yes/no anywhere in first line
        first_line = response_lower.split("\n")[0]
        return "yes" in first_line

    @staticmethod
    def _extract_feedback(llm_response: str) -> str:
        """Extract feedback from <feedback> tags in LLM response.

        Args:
            llm_response: The LLM's response potentially containing <feedback> tags.

        Returns:
            Extracted feedback text, or full response if no tags found.
        """
        # Try to extract content from <feedback> tags
        match = re.search(
            r"<feedback>(.*?)</feedback>", llm_response, re.DOTALL | re.IGNORECASE
        )
        if match:
            return match.group(1).strip()
        # Fallback to full response
        return llm_response.strip()

    async def _validate_with_judge_backend(
        self,
        requirement: Requirement,
        context: Context,
        model_options: dict | None = None,
    ) -> ValidationResult:
        """Validate using judge backend with configurable feedback strategy.

        Args:
            requirement: The requirement to validate.
            context: The generation context.
            model_options: Optional model options for the judge backend.

        Returns:
            ValidationResult with judgment and feedback.
        """
        assert self.judge_backend is not None

        # Build validation prompt based on feedback_strategy
        if self.feedback_strategy == "simple":
            validation_prompt = (
                f"Does the following output satisfy this requirement?\n\n"
                f"Requirement: {requirement.description}\n\n"
                f"Answer with only 'Yes' or 'No'."
            )
        elif self.feedback_strategy == "first_error":
            validation_prompt = (
                f"Does the following output satisfy this requirement?\n\n"
                f"Requirement: {requirement.description}\n\n"
                f"Instructions: Identify and report only the FIRST mistake you find. "
                f"Answer with 'Yes' or 'No', then provide specific feedback about the first error "
                f"in <feedback></feedback> tags.\n\n"
                f"Example format:\n"
                f"No\n"
                f"<feedback>The first mistake is: X. Please fix it.</feedback>"
            )
        elif self.feedback_strategy == "all_errors":
            validation_prompt = (
                f"Does the following output satisfy this requirement?\n\n"
                f"Requirement: {requirement.description}\n\n"
                f"Instructions: Identify ALL mistakes comprehensively. "
                f"Answer with 'Yes' or 'No', then provide detailed feedback about every error found "
                f"in <feedback></feedback> tags.\n\n"
                f"Example format:\n"
                f"No\n"
                f"<feedback>The following mistakes were found:\n"
                f"1. Issue X\n"
                f"2. Issue Y\n"
                f"3. Issue Z</feedback>"
            )
        else:  # Fallback (shouldn't reach here with current enum)
            validation_prompt = (
                f"Does the following output satisfy this requirement?\n\n"
                f"Requirement: {requirement.description}\n\n"
                f"Answer with 'Yes' or 'No'."
            )

        # Create message for validation
        validation_message = Message(role="user", content=validation_prompt)

        # Generate judgment using judge backend
        judgment_output, _ = await self.judge_backend.generate_from_context(
            validation_message, context, model_options=model_options
        )
        await judgment_output.avalue()

        # Parse judgment
        judgment_text = str(judgment_output.value)
        is_valid = self._parse_judgment(judgment_text)

        # Extract feedback for repair
        if self.feedback_strategy in ["first_error", "all_errors"]:
            feedback = self._extract_feedback(judgment_text)
        else:  # "simple" or fallback
            feedback = (
                "Requirement not satisfied."
                if not is_valid
                else "Requirement satisfied."
            )

        return ValidationResult(result=is_valid, reason=feedback)

    def _create_judge_validate_function(
        self, requirement: Requirement, model_options: dict | None = None
    ):
        """Create a custom validation function that uses judge backend.

        Args:
            requirement: The requirement to wrap with judge validation.
            model_options: Optional model options for the judge backend.

        Returns:
            Callable that validates using judge backend.
        """

        async def validate_with_judge(ctx: Context) -> ValidationResult:
            """Custom validator using judge backend."""
            return await self._validate_with_judge_backend(
                requirement, ctx, model_options
            )

        return validate_with_judge

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    def _prepare_requirements_for_validation(
        self, reqs: list[Requirement], model_options: dict | None = None
    ) -> list[Requirement]:
        """Prepare requirements for validation by wrapping with judge backend if needed.

        If judge_backend is provided, wraps requirements that don't have a custom
        validation_fn to use the judge backend instead.

        Priority: validation_fn > judge_backend > session backend (fallback)

        Args:
            reqs: Original requirements list.
            model_options: Model options to pass to judge backend.

        Returns:
            List of requirements ready for validation (some may be wrapped).
        """
        if self.judge_backend is None:
            return reqs

        reqs_for_validation = []
        for req in reqs:
            if req.validation_fn is None:
                # Wrap with judge backend validation
                wrapped_req = Requirement(
                    description=req.description,
                    validation_fn=self._create_judge_validate_function(
                        req, model_options
                    ),
                    output_to_bool=req.output_to_bool,
                    check_only=req.check_only,
                )
                reqs_for_validation.append(wrapped_req)
            else:
                # Use original requirement with custom validator
                reqs_for_validation.append(req)
        return reqs_for_validation

    def _prepare_s2_context(
        self,
        s2_mode: str,
        original_action: Component,
        original_context: Context,
        last_result_ctx: Context,
        last_action: Component,
        sampled_results: list[ModelOutputThunk],
        sampled_scores: list[list[tuple[Requirement, ValidationResult]]],
        loop_count: int,
    ) -> tuple[Component, Context]:
        """Prepare context and action for S2 Solver based on mode.

        Args:
            s2_mode: One of "fresh_start", "continue_chat", "best_attempt".
            original_action: The original action passed to sample().
            original_context: The original context passed to sample().
            last_result_ctx: Context from last S1 iteration.
            last_action: Action from last S1 iteration.
            sampled_results: All S1 generation results.
            sampled_scores: All S1 validation scores.
            loop_count: Number of S1 iterations completed.

        Returns:
            Tuple of (action_for_s2, context_for_s2).
        """
        flog = FancyLogger.get_logger()

        if s2_mode == "fresh_start":
            # Clean slate: same prompt as S1
            flog.info("SOFAI S2: Starting with fresh context (clean slate).")
            return deepcopy(original_action), original_context

        elif s2_mode == "continue_chat":
            # Fresh start input + S1's iteration and feedback history
            flog.info("SOFAI S2: Continuing with original prompt plus S1 history.")
            return deepcopy(original_action), last_result_ctx

        else:  # best_attempt
            # Find best S1 attempt and build informative prompt
            flog.info("SOFAI S2: Receiving best S1 Solver attempt with feedback.")

            best_idx = self._select_best_attempt(sampled_scores)
            best_result = sampled_results[best_idx]
            best_validations = sampled_scores[best_idx]

            passing_count = sum(1 for _, val in best_validations if val.as_bool())
            flog.info(
                f"SOFAI S2: Best attempt: #{best_idx + 1} with "
                f"{passing_count}/{len(best_validations)} requirements passed."
            )

            # Build feedback summary
            failed_reqs = [
                (req, val) for req, val in best_validations if not val.as_bool()
            ]
            feedback_lines = []
            for req, val in failed_reqs:
                feedback_lines.append(
                    f"  - {val.reason if val.reason else req.description}"
                )

            # Extract original problem statement from original_action
            original_prompt = self._extract_action_prompt(original_action)

            # Build prompt for S2: original problem + best attempt + feedback
            prompt_parts = [
                f"Original task:\n{original_prompt}\n\n"
                f"Previous attempts were made to solve this problem. "
                f"The best attempt (out of {loop_count} attempts) was:\n\n"
                f"{best_result.value}\n\n"
            ]
            if feedback_lines:
                prompt_parts.append(
                    "However, this attempt had the following issues:\n"
                    + "\n".join(feedback_lines)
                    + "\n\n"
                )
            prompt_parts.append(
                "Please improve upon the best attempt above to fully satisfy all requirements."
            )

            s2_action = Message(role="user", content="".join(prompt_parts))
            return s2_action, original_context

    async def _generate_and_validate(
        self,
        solver_backend: Backend,
        action: Component,
        ctx: Context,
        reqs: list[Requirement],
        session_backend: Backend,
        format: type[BaseModelSubclass] | None,
        model_options: dict | None,
        tool_calls: bool,
    ) -> tuple[ModelOutputThunk, Context, list[tuple[Requirement, ValidationResult]]]:
        """Generate with a solver and validate the result.

        Args:
            solver_backend: Backend to use for generation.
            action: Action/prompt to generate from.
            ctx: Context for generation.
            reqs: Requirements to validate against.
            session_backend: Fallback backend for validation.
            format: Output format for structured outputs.
            model_options: Model options for generation.
            tool_calls: Whether to use tool calls.

        Returns:
            Tuple of (result, result_context, validation_scores).
        """
        # Generate
        result, result_ctx = await solver_backend.generate_from_context(
            action,
            ctx=ctx,
            format=format,
            model_options=model_options,
            tool_calls=tool_calls,
        )
        await result.avalue()

        # Note: Unlike base.py which sets result.parsed_repr = action.parse(result),
        # we skip this because MOTs are immutable. The parsed_repr was set by the
        # backend during generation and may differ from what action.parse() returns
        # (e.g., when action is a repair Message vs the original Instruction).

        # Prepare and run validation
        reqs_for_validation = self._prepare_requirements_for_validation(
            reqs, model_options
        )
        val_scores = await mfuncs.avalidate(
            reqs=reqs_for_validation,
            context=result_ctx,
            backend=session_backend,
            output=result,
            format=None,
            model_options=model_options,
        )
        constraint_scores = list(zip(reqs, val_scores))

        return result, result_ctx, constraint_scores

    # =========================================================================
    # Main Sample Method
    # =========================================================================

    async def sample(
        self,
        action: Component[S],
        context: Context,
        backend: Backend,
        requirements: list[Requirement] | None,
        *,
        validation_ctx: Context | None = None,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> SamplingResult[S]:
        """Execute SOFAI two-solver sampling strategy.

        SOFAI Flow Overview:
        ====================
        1. PHASE 1 - S1 Solver Loop:
           - Generate candidate solution with fast S1 model
           - Validate against requirements
           - If success: return immediately
           - If failure: generate repair feedback and iterate
           - If no improvement detected: early exit to Phase 2

        2. PHASE 2 - S2 Solver Escalation:
           - Prepare context based on s2_solver_mode:
             * fresh_start: clean slate with original prompt
             * continue_chat: full S1 conversation history
             * best_attempt: best S1 result with feedback summary
           - Generate single attempt with slow S2 model
           - Validate and return result (success or failure)

        Args:
            action: The component to sample (Instruction, Message, etc.).
            context: The session context (must be ChatContext).
            backend: Session backend (used for validation fallback).
            requirements: Requirements to validate against.
            validation_ctx: Optional separate validation context (unused).
            format: Output format for structured outputs.
            model_options: Model options to pass to backends.
            tool_calls: True if tool calls should be used.

        Returns:
            SamplingResult with success status and all generation history.
        """
        # ---------------------------------------------------------------------
        # Setup and Validation
        # ---------------------------------------------------------------------
        assert isinstance(context, ChatContext), (
            "SOFAI requires ChatContext for conversation management."
        )

        flog = FancyLogger.get_logger()
        reqs: list[Requirement] = list(requirements) if requirements else []

        # State tracking for all attempts
        sampled_results: list[ModelOutputThunk] = []
        sampled_scores: list[list[tuple[Requirement, ValidationResult]]] = []
        sampled_actions: list[Component] = []
        sample_contexts: list[Context] = []

        # ---------------------------------------------------------------------
        # PHASE 1: S1 Solver Loop
        # ---------------------------------------------------------------------
        flog.info(
            f"SOFAI: Starting S1 Solver ({getattr(self.s1_solver_backend, 'model_id', 'unknown')}) "
            f"loop (budget={self.loop_budget})"
        )

        previous_failed_set: set[tuple[str | None, str | None, float | None]] = set()
        loop_count = 0
        next_action = deepcopy(action)
        next_context: Context = context

        show_progress = flog.getEffectiveLevel() <= FancyLogger.INFO
        loop_iterator = (
            tqdm.tqdm(range(self.loop_budget), desc="S1 Solver")
            if show_progress
            else range(self.loop_budget)
        )

        # Exit conditions: success returns immediately; no-improvement breaks
        # early to S2 escalation; loop budget exhaustion flows to S2 escalation.
        for _ in loop_iterator:
            loop_count += 1
            if not show_progress:
                flog.info(f"SOFAI S1: Running loop {loop_count} of {self.loop_budget}")

            # Generate and validate
            result, result_ctx, constraint_scores = await self._generate_and_validate(
                solver_backend=self.s1_solver_backend,
                action=next_action,
                ctx=next_context,
                reqs=reqs,
                session_backend=backend,
                format=format,
                model_options=model_options,
                tool_calls=tool_calls,
            )

            # Store attempt
            sampled_results.append(result)
            sampled_scores.append(constraint_scores)
            sampled_actions.append(next_action)
            sample_contexts.append(result_ctx)

            # Check for success
            if all(bool(score[1]) for score in constraint_scores):
                flog.info(f"SOFAI S1: SUCCESS on attempt {loop_count}")
                assert result._generate_log is not None
                result._generate_log.is_final_result = True

                # Exit with success
                return SamplingResult(
                    result_index=len(sampled_results) - 1,
                    success=True,
                    sample_generations=sampled_results,
                    sample_validations=sampled_scores,
                    sample_contexts=sample_contexts,
                    sample_actions=sampled_actions,
                )

            # Log partial progress
            count_valid = sum(1 for s in constraint_scores if bool(s[1]))
            flog.info(
                f"SOFAI S1: FAILED attempt {loop_count}. "
                f"Valid: {count_valid}/{len(constraint_scores)}"
            )

            # Check for no improvement (early exit to S2)
            current_failed_set = {
                (req.description, val.reason, val.score)
                for req, val in constraint_scores
                if not val.as_bool()
            }
            if loop_count > 1 and current_failed_set == previous_failed_set:
                flog.warning(
                    f"SOFAI S1: No improvement detected between attempt "
                    f"{loop_count - 1} and {loop_count}. Escalating to S2 Solver."
                )
                # Exit with no improvement
                break
            previous_failed_set = current_failed_set

            # Prepare repair for next iteration
            if loop_count < self.loop_budget:
                next_action, next_context = self.repair(
                    next_context,
                    result_ctx,
                    sampled_actions,
                    sampled_results,
                    sampled_scores,
                )
        # Exit due to loop budget exhaustion or no improvement

        # ---------------------------------------------------------------------
        # PHASE 2: S2 Solver Escalation
        # ---------------------------------------------------------------------
        flog.info(
            f"SOFAI: S1 Solver completed {loop_count} attempts. "
            f"Escalating to S2 Solver ({getattr(self.s2_solver_backend, 'model_id', 'unknown')})."
        )

        # Prepare S2 context based on mode
        s2_action, s2_context = self._prepare_s2_context(
            s2_mode=self.s2_solver_mode,
            original_action=action,
            original_context=context,
            last_result_ctx=result_ctx,
            last_action=next_action,
            sampled_results=sampled_results,
            sampled_scores=sampled_scores,
            loop_count=loop_count,
        )

        # Generate and validate with S2
        result, result_ctx, constraint_scores = await self._generate_and_validate(
            solver_backend=self.s2_solver_backend,
            action=s2_action,
            ctx=s2_context,
            reqs=reqs,
            session_backend=backend,
            format=format,
            model_options=model_options,
            tool_calls=tool_calls,
        )

        # Store S2 attempt
        sampled_results.append(result)
        sampled_scores.append(constraint_scores)
        sampled_actions.append(s2_action)
        sample_contexts.append(result_ctx)

        # Check S2 success
        assert result._generate_log is not None
        result._generate_log.is_final_result = True

        if all(bool(score[1]) for score in constraint_scores):
            flog.info("SOFAI S2: SUCCESS")
            return SamplingResult(
                result_index=len(sampled_results) - 1,
                success=True,
                sample_generations=sampled_results,
                sample_validations=sampled_scores,
                sample_contexts=sample_contexts,
                sample_actions=sampled_actions,
            )
        else:
            count_valid = sum(1 for s in constraint_scores if bool(s[1]))
            flog.warning(
                f"SOFAI S2: FAILED. Valid: {count_valid}/{len(constraint_scores)}. "
                f"Returning S2 Solver's attempt as final result."
            )
            return SamplingResult(
                result_index=len(sampled_results) - 1,
                success=False,
                sample_generations=sampled_results,
                sample_validations=sampled_scores,
                sample_contexts=sample_contexts,
                sample_actions=sampled_actions,
            )
