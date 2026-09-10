# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`Requirement` interface for constrained and validated generation.

A `Requirement` pairs a human-readable description with a validation function that
inspects a `Context` (and optionally a backend) to determine whether a model output
meets a constraint. `ValidationResult` carries the pass/fail verdict along with an
optional reason, score, and the `ModelOutputThunk` produced during validation.
`PartialValidationResult` provides a tri-state variant (`"pass"`, `"fail"`,
`"unknown"`) for per-chunk streaming validation.
Helper factories such as `default_output_to_bool` make it easy to build requirements
without boilerplate.
"""

import re
from collections.abc import Callable
from copy import copy
from dataclasses import dataclass
from typing import Literal, final

from .backend import Backend, BaseModelSubclass
from .base import (
    CBlock,
    Component,
    Context,
    ModelOutputThunk,
    Span,
    TemplateRepresentation,
)
from .chunking import Chunker, ChunkingStrategy, resolve_chunking_strategy


class ValidationResult:
    """ValidationResults store the output of a Requirement's validation. They can be used to return additional info from validation functions, which is useful for sampling/repairing.

    Args:
        result (bool): Boolean indicating whether the requirement passed.
        reason (str | None): Optional human-readable explanation for the verdict.
        score (float | None): Optional numeric score returned by the validator.
        thunk (ModelOutputThunk | None): The `ModelOutputThunk` produced during LLM-as-a-Judge validation, if applicable.
        context (Context | None): The context associated with the validation backend call, if applicable.
        error (Exception | None): Set when validation could not produce a verdict because the
            output could not be parsed — for example an `AdapterSchemaMismatchError` from a
            custom `output_to_bool`. When set, `bool(result)` fails closed to `False`
            regardless of `result`, so callers that ignore errors do not silently pass.
            Inspect `error` to distinguish "requirement not met" from "output unparsable".

    """

    def __init__(
        self,
        result: bool,
        *,
        reason: str | None = None,
        score: float | None = None,
        thunk: ModelOutputThunk | None = None,
        context: Context | None = None,
        error: Exception | None = None,
    ):
        """Initialize ValidationResult with a pass/fail boolean and optional metadata."""
        self._result = result
        self._reason = reason
        self._score = score
        self._thunk = thunk
        self._context = context
        self._error = error

    @property
    def reason(self) -> str | None:
        """Reason for the validation result."""
        return self._reason

    @property
    def score(self) -> float | None:
        """An optional score for the validation result."""
        return self._score

    @property
    def thunk(self) -> ModelOutputThunk | None:
        """The ModelOutputThunk associated with the validation func if an llm was used to generate the final result."""
        return self._thunk

    @property
    def context(self) -> Context | None:
        """The context associated with validation if a backend was used to generate the final result."""
        return self._context

    @property
    def error(self) -> Exception | None:
        """The exception raised while parsing the validation output, if any.

        Set when `validate()` could not produce a verdict because the output was
        unparsable (e.g. an `AdapterSchemaMismatchError` from a custom
        `output_to_bool`). `None` for an ordinary pass or fail. When set,
        `as_bool()` returns `False` regardless of the underlying result.
        """
        return self._error

    def as_bool(self) -> bool:
        """Return a boolean value based on the validation result.

        Fails closed when an `error` is set: an unparsable output is never
        treated as a pass, so callers that only check the boolean fail safely
        rather than silently accepting a result that was never computed.

        Returns:
            bool: `True` if the requirement passed and no error is set, `False` otherwise.
        """
        if self._error is not None:
            return False
        return self._result

    def __bool__(self) -> bool:
        """Return a boolean value based on the result."""
        return self.as_bool()

    def __repr__(self) -> str:
        """Return a developer-readable representation of the validation result."""
        return (
            f"ValidationResult({self._result!r}, reason={self._reason!r}, "
            f"score={self._score!r}, error={self._error!r})"
        )


class PartialValidationResult:
    """Tri-state result from per-chunk streaming validation.

    Unlike :class:`ValidationResult`, which stores its verdict as a private
    `_result: bool`, this class exposes `success` as a public property.
    The asymmetry is intentional: the tri-state value cannot be recovered from
    a `bool`, so a public property is the only way to distinguish `"fail"`
    from `"unknown"` after construction.

    Args:
        success: Validation state — `"pass"` (constraint satisfied so far),
            `"fail"` (constraint violated, stop streaming), or
            `"unknown"` (insufficient data yet, continue streaming).
        reason: Optional human-readable explanation.
        score: Optional numeric confidence score.
        thunk: Optional ModelOutputThunk from the validation call.
        context: Optional context associated with the validation call.

    """

    def __init__(
        self,
        success: Literal["pass", "fail", "unknown"],
        *,
        reason: str | None = None,
        score: float | None = None,
        thunk: ModelOutputThunk | None = None,
        context: Context | None = None,
    ):
        """Initialize PartialValidationResult with a tri-state success value."""
        if success not in ("pass", "fail", "unknown"):
            raise ValueError(
                f"success must be 'pass', 'fail', or 'unknown', got {success!r}"
            )
        self._success: Literal["pass", "fail", "unknown"] = success
        self._reason = reason
        self._score = score
        self._thunk = thunk
        self._context = context

    @property
    def success(self) -> Literal["pass", "fail", "unknown"]:
        """The tri-state validation result."""
        return self._success

    @property
    def reason(self) -> str | None:
        """Reason for the validation result."""
        return self._reason

    @property
    def score(self) -> float | None:
        """An optional score for the validation result."""
        return self._score

    @property
    def thunk(self) -> ModelOutputThunk | None:
        """The ModelOutputThunk associated with the validation call, if any."""
        return self._thunk

    @property
    def context(self) -> Context | None:
        """The context associated with the validation call, if any."""
        return self._context

    def as_bool(self) -> bool:
        """Return True for `"pass"`, False for `"fail"` or `"unknown"`.

        `"unknown"` maps to `False` intentionally. In streaming contexts,
        check `pvr.success == "unknown"` before treating `False` as a definitive
        failure — `"unknown"` means insufficient data so far, not a constraint
        violation.

        Returns:
            bool: `True` if the partial result is `"pass"`, `False` otherwise.
        """
        return self._success == "pass"

    def __bool__(self) -> bool:
        """Return a boolean value based on the success state."""
        return self.as_bool()

    def __repr__(self) -> str:
        """Return a developer-readable representation showing the tri-state value."""
        return f"PartialValidationResult({self._success!r}, reason={self._reason!r}, score={self._score!r})"


@dataclass
class PartialValidationSummary:
    """Aggregate of the per-chunk `PartialValidationResult`s a requirement produces for one validated input.

    A requirement re-chunks the text it is given and validates each chunk, so one input can
    yield several results (one when it does not re-chunk). Build one with `from_results`.

    Args:
        results (list[PartialValidationResult]): The per-chunk results, in order.
        success (Literal["pass", "fail", "unknown"]): Aggregate verdict — `"fail"` if any chunk
            failed, `"pass"` if every chunk passed, else `"unknown"` (some chunk undecided, or
            no results).
        failure (PartialValidationResult | None): The failing chunk's result (at most one,
            given `stream_validate` short-circuits at the first failure), or `None`.
        reason (str | None): The failing chunk's `reason`, or `None` when no chunk failed.
    """

    results: list[PartialValidationResult]
    success: Literal["pass", "fail", "unknown"]
    failure: PartialValidationResult | None
    reason: str | None

    @classmethod
    def from_results(
        cls, results: list[PartialValidationResult]
    ) -> "PartialValidationSummary":
        """Summarize per-chunk `results` into a single verdict.

        An empty `results` summarizes to `"unknown"` (no verdict) with no failure.

        Args:
            results: The per-chunk results for one validated input.

        Returns:
            PartialValidationSummary: `success` is `"fail"` if any result failed, `"pass"` if
            every result passed, else `"unknown"`; `failure` (and its `reason`) come from the
            failing result, or are `None`.
        """
        failure = next((r for r in results if r.success == "fail"), None)
        success: Literal["pass", "fail", "unknown"]
        if failure is not None:
            success = "fail"
        elif results and all(r.success == "pass" for r in results):
            success = "pass"
        else:
            success = "unknown"
        reason = failure.reason if failure is not None else None
        return cls(results=results, success=success, failure=failure, reason=reason)


def default_output_to_bool(x: CBlock | ModelOutputThunk | str) -> bool:
    """Convert a model output string to a boolean by checking for a "yes" answer.

    Checks if the output is exactly equal to "yes" or "y" (case-insensitive). If not,
    also checks if any of the words in the output are "yes" (case-insensitive).

    Args:
        x: The model output to evaluate, as a `CBlock`, `ModelOutputThunk`, or plain string.

    Returns:
        `True` if the output indicates a "yes" answer, `False` otherwise.
    """
    output = str(x)

    if output.upper() == "YES" or output.upper() == "Y":
        return True

    word_splits = re.split(r"\W+", output)
    if "YES" in [word.upper() for word in word_splits]:
        return True

    return False


class Requirement(Component[str]):
    """Requirements are a special type of Component used as input to the Validate step in Instruct/Validate/Repair patterns.

    Args:
        description (str | None): A natural-language description of the requirement. Sometimes included in
            `Instruction` prompts; use `check_only=True` to suppress this.
        validation_fn (Callable[[Context], ValidationResult] | None): If provided, this function is executed
            instead of LLM-as-a-Judge. The `bool()` of its return value defines pass/fail.
        output_to_bool (Callable[[CBlock | ModelOutputThunk | str], bool] | None): Translates LLM-as-a-Judge output to a boolean.
            Defaults to a "yes"-detection heuristic. May raise if the output does not match
            the expected format — see `validate` for details.
        check_only (bool): When `True`, the requirement description is excluded from `Instruction` prompts.
        chunking (str | ChunkingStrategy | None): Chunking strategy for streaming validation.
            When set, the requirement re-chunks the stream into its own validation chunks (see
            `stream_validate`); an alias string is resolved. `None` (default) validates each
            stream chunk as-is.

    Attributes:
        description (str | None): A natural-language description of the requirement.
        output_to_bool (Callable[[CBlock | ModelOutputThunk | str], bool] | None): Function used to convert LLM-as-a-Judge
            output into a boolean pass/fail result.
        validation_fn (Callable[[Context], ValidationResult] | None): Optional custom validation
            function that bypasses the LLM-as-a-Judge strategy entirely.
        check_only (bool): When `True`, the requirement description is excluded from `Instruction`
            prompts to avoid influencing model output.
        chunking (ChunkingStrategy | None): The resolved chunking strategy, or `None`.
    """

    def __init__(
        self,
        description: str | None = None,
        validation_fn: Callable[[Context], ValidationResult] | None = None,
        *,
        output_to_bool: Callable[[CBlock | ModelOutputThunk | str], bool]
        | None = default_output_to_bool,
        check_only: bool = False,
        chunking: str | ChunkingStrategy | None = None,
    ):
        """Initialize Requirement with an optional description, validation function, and output converter."""
        self.description = description
        self.output_to_bool = output_to_bool
        self.validation_fn = validation_fn
        self.check_only = check_only
        self.chunking: ChunkingStrategy | None = resolve_chunking_strategy(chunking)

        # Used for validation. Do not manually populate.
        self._output: str | None = None

        # Per-stream chunker for streaming validation, built lazily.
        self._chunker: Chunker | None = None

    def __copy__(self) -> "Requirement":
        """Return a shallow copy with the live `_chunker` reset to `None`.

        The chunker holds per-stream state that must not be shared between copies. Subclasses
        overriding `__copy__` should call `super().__copy__()` to preserve the reset.

        Returns:
            Requirement: A shallow copy whose `_chunker` is `None`.
        """
        clone = self.__class__.__new__(self.__class__)
        clone.__dict__.update(self.__dict__)
        clone._chunker = None
        return clone

    async def validate(
        self,
        backend: Backend,
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
    ) -> ValidationResult:
        """Chooses the appropriate validation strategy and applies it to the given context.

        Uses `validation_fn` if one was provided, otherwise falls back to LLM-as-a-Judge
        by generating a judgement response with the backend.

        Args:
            backend (Backend): The inference backend used when the LLM-as-a-Judge strategy is selected.
            ctx (Context): The context to validate, which must contain a `ModelOutputThunk` as its last output.
            format (type[BaseModelSubclass] | None): Optional structured output format for the judgement call.
            model_options (dict | None): Optional model options to pass to the backend during the judgement call.

        Returns:
            ValidationResult: The result of the validation, including a boolean pass/fail
            and optional metadata. If `output_to_bool` raises while parsing the judgement
            output (e.g. `AdapterSchemaMismatchError` on an unexpected adapter schema), the
            exception is caught and stored on `result.error` rather than propagated: the
            result fails closed (`bool(result)` is `False`), and callers can inspect
            `result.error` to distinguish "requirement not met" from "output unparsable".

        Raises:
            AssertionError: If the LLM-as-a-Judge strategy is selected but `output_to_bool`
                is `None`, or if the context has no `ModelOutputThunk` as its last output.
        """
        if self.validation_fn is not None:
            # Python validation strategy
            return self.validation_fn(ctx)
        else:
            # LLMaJ validation strategy. This includes aLoRA because the backend generate call will appropriately dispatch.
            assert self.output_to_bool is not None
            last_output = ctx.last_output()
            assert isinstance(last_output, ModelOutputThunk), (
                " Context has no appropriate last output"
            )

            # Create a copy of the requirement that holds the output
            # and its template gets populated with the output correctly.
            req_copy = copy(self)
            req_copy._output = last_output.value
            llm_as_a_judge_result, val_ctx = await backend.generate_from_context(
                req_copy, ctx, format=format, model_options=model_options
            )
            await llm_as_a_judge_result.avalue()

            # LLM-as-a-Judge validation often returns only "yes"/"no" because the
            # prompt asks for binary classification. However, repair strategies
            # display the reason to guide the model during repair iterations.
            # A bare "no" is unhelpful; the requirement description is more
            # actionable (e.g., "The email should have a salutation" vs "no").
            judge_output = llm_as_a_judge_result.value
            reason = judge_output
            if judge_output:
                judge_output_str = str(judge_output).strip().lower()
                if judge_output_str in ("yes", "no"):
                    reason = self.description

            try:
                result = self.output_to_bool(llm_as_a_judge_result)
            except Exception as exc:
                # A custom output_to_bool (e.g. the adapter-backed
                # requirement_check_to_bool) can raise on unparsable output.
                # Surface that as a third outcome rather than propagating: the
                # result fails closed and carries the exception for callers that
                # want to distinguish it from an ordinary failure.
                #
                # reason is deliberately None here. Repair strategies treat a
                # truthy reason as literal prompt text; the judge output that
                # triggered the parse error is malformed and would make useless
                # repair guidance. None routes repair to the existing
                # requirement-description fallback, while error and thunk retain
                # the diagnostic for callers that inspect the result directly.
                return ValidationResult(
                    result=False,
                    reason=None,
                    thunk=llm_as_a_judge_result,
                    context=val_ctx,
                    error=exc,
                )

            return ValidationResult(
                result=result,
                reason=reason,
                thunk=llm_as_a_judge_result,
                context=val_ctx,
            )

    @final
    async def stream_validate(
        self, delta: str, *, backend: Backend, ctx: Context, flush: bool = False
    ) -> list[PartialValidationResult]:
        """Validate one stream delta, re-chunked into this requirement's own chunks.

        Feeds `delta` to this requirement's chunker (built from `chunking`) and runs
        `_stream_validate` on each complete chunk until one fails, returning the results up to
        and including that failure — later chunks are skipped. With `chunking=None` the delta is
        validated as-is. When `delta` completes no chunk (the chunker is still accumulating),
        the result is a single `"unknown"`. For a `chunking=None` requirement, an empty `delta`
        returns a single `"unknown"` without invoking `_stream_validate`.

        To add streaming validation, override `_stream_validate`, returning `"pass"`
        (satisfied so far), `"fail"` (constraint violated), or `"unknown"` (no verdict yet)
        for each chunk.

        Args:
            delta: The next piece of stream text to feed this requirement's chunker.
            backend: The inference backend, for backend-assisted checks.
            ctx: The current generation context.
            flush: When `True` also validate the trailing residual withheld by this requirement's chunker.

        Returns:
            list[PartialValidationResult]: One result per validated chunk, ending at the first
            failing chunk, or a single `"unknown"` when `delta` completes no chunk. Never empty.
        """
        if self.chunking is None:
            return (
                [await self._stream_validate(delta, backend=backend, ctx=ctx)]
                if delta
                else [PartialValidationResult("unknown")]
            )
        if self._chunker is None:
            self._chunker = Chunker(self.chunking)
        results: list[PartialValidationResult] = []
        for chunk in self._chunker.feed(delta):
            result = await self._stream_validate(chunk, backend=backend, ctx=ctx)
            results.append(result)
            if result.success == "fail":
                break
        if flush and not any(r.success == "fail" for r in results):
            results.extend(await self.stream_flush(backend=backend, ctx=ctx))
        return results or [PartialValidationResult("unknown")]

    async def _stream_validate(
        self, chunk: str, *, backend: Backend, ctx: Context
    ) -> PartialValidationResult:
        """Validate a single chunk during streaming.

        The default implementation returns `PartialValidationResult("unknown")`
        — meaning insufficient data to decide yet. Subclasses override this method
        to inspect the current chunk and return `"pass"` or `"fail"` early.

        Implementations may accumulate state on `self` across calls within a
        single attempt. The orchestrator clones the requirement (`copy(req)`)
        before each attempt, so state does not bleed across retries.

        Shallow-copy caveat: mutable container fields (e.g. `self._buffer = []`)
        are shared by reference under `copy()`. Reassign rather than mutate in
        place (`self._buffer = self._buffer + [chunk]`, not
        `self._buffer.append(chunk)`), or override `__copy__` for proper isolation.

        Overrides with externally visible side effects (file writes, network
        calls) should perform them only after any logic that could raise, since
        the framework cannot roll them back.

        Implementations must not call `mot.astream()` or otherwise read the
        underlying stream; the stream driver is the single consumer of the MOT
        stream (see `ModelOutputThunk.astream`). Requirements that need access
        to the text seen so far should accumulate it themselves from the
        `chunk` values they receive.

        Args:
            chunk: A single complete, non-empty semantic chunk produced by the chunking
                strategy (e.g. one sentence for `SentenceChunking`). This is
                the delta since the previous call for this attempt, not the
                accumulated output. Requirements that need earlier context
                should retain it on `self` across calls.
            backend: The inference backend, available for backend-assisted checks.
            ctx: The current generation context. During streaming the MOT is
                not yet computed, so `ctx` does not contain the generated
                output; use `chunk` (and any state accumulated on `self`) instead.

        Returns:
            PartialValidationResult: `"unknown"` by default. Subclasses may return
            `"pass"` (constraint satisfied so far) or `"fail"` (constraint violated,
            streaming should be aborted). `"pass"` does not short-circuit the final
            `validate()` call; the orchestrator decides whether to skip it.
        """
        return PartialValidationResult("unknown")

    async def stream_flush(
        self, *, backend: Backend, ctx: Context
    ) -> list[PartialValidationResult]:
        """Validate the trailing residual withheld by this requirement's chunker.

        The end-of-stream counterpart to `stream_validate`: with no new chunk, it releases the
        final chunk the chunker held back (the text after its last boundary) and runs it through
        `_stream_validate`. Returns one result per residual chunk (0 or 1 per the
        `ChunkingStrategy.flush` contract), or an empty list when there is no chunker or nothing
        was withheld.

        Args:
            backend: The inference backend, available for backend-assisted checks.
            ctx: The current generation context.

        Returns:
            list[PartialValidationResult]: The residual chunk's result(s), or empty if none.
        """
        if self._chunker is None:
            return []
        return [
            await self._stream_validate(chunk, backend=backend, ctx=ctx)
            for chunk in self._chunker.flush()
        ]

    def parts(self) -> list[Span]:
        """Returns all of the constituent parts of a Requirement.

        Returns:
            List of constituent components. Empty by default; subclasses override
            to expose their internal structure.
        """
        return []

    def format_for_llm(self) -> TemplateRepresentation | str:
        """Returns a `TemplateRepresentation` for LLM-as-a-Judge evaluation of this requirement.

        Populates the template with the requirement's `description` and the stored model
        `_output`. Must only be called from within a `validate` call for this same requirement,
        after `_output` has been set.

        Returns:
            TemplateRepresentation | str: A `TemplateRepresentation` containing the description
            and the model output to be judged.
        """
        assert self._output is not None, (
            "Object protocol error: should never try to templatize a Requirement except inside of a validate call for that same requirement."
        )
        return TemplateRepresentation(
            obj=self,
            args={"description": self.description, "output": self._output},
            tools=None,
            template_order=["*", "Requirement"],
        )

    def _parse(self, computed: ModelOutputThunk) -> str:
        """Parse the model output. Returns string value for now."""
        return computed.value if computed.value is not None else ""
