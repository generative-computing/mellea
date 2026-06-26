"""Requirements for Python code generation validation."""

import dataclasses
import warnings
from collections.abc import Callable
from typing import Literal

from mellea.stdlib.tools.execution_policy import (
    DOCKER_POLICY,
    LOCAL_POLICY,
    CapabilityPolicy,
    ExecutionTier,
)
from mellea.stdlib.tools.interpreter import (
    ExecutionEnvironment,
    make_execution_environment,
)

from ...core import Context, MelleaLogger, ModelToolCall, Requirement, ValidationResult

logger = MelleaLogger.get_logger()


# region code extraction


def _score_code_block(code: str) -> int:
    """Score a code block to determine if it's likely the main answer.

    Scoring metrics:
    - Length bonus: +1 per line (capped at 10) - longer blocks are generally more substantial
    - Function/class bonus: +5 - indicates complete, structured code
    - Control flow bonus: +3 - presence of if/for/while/try/with suggests meaningful logic
    - Non-trivial content penalty: -5 if fewer than 2 executable lines (filters out import-only or comment-heavy blocks)

    Returns:
        int: Score indicating likelihood this is the primary code block to execute.
    """
    score = 0
    lines = code.split("\n")

    # Longer blocks generally better
    score += min(len(lines), 10)

    # Prefer complete functions/classes
    if "def " in code or "class " in code:
        score += 5

    # Prefer blocks with actual logic
    if any(keyword in code for keyword in ["if ", "for ", "while ", "try:", "with "]):
        score += 3

    # Penalize blocks that are mostly imports/comments without actual logic
    # We want at least 2 lines of executable code to consider it a meaningful code block
    # This helps filter out import-only blocks or heavily commented trivial snippets
    # TODO: Consider using comment-to-code ratio in future iterations
    non_trivial_lines = [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith(("#", "import ", "from "))
    ]
    if len(non_trivial_lines) < 2:
        score -= 5

    return score


def _extract_code_from_tool_call(tool_call: ModelToolCall) -> str | None:
    """Extract Python code from a tool call's arguments using heuristic field lookup.

    Matches tools with "python" in their name, including: standalone "python", "python_tool",
    or "python" paired with execution keywords (executor, interpreter, runner).
    This conservative approach avoids extracting unrelated data from arbitrary tools.

    Tries common code field names in order: 'code', 'script', 'command', 'source'.
    Returns the first string value found, or None if no code-like field exists.

    Args:
        tool_call: The ModelToolCall to extract code from.

    Returns:
        str | None: Extracted code string, or None if not found, not a string, or tool
            name doesn't suggest it handles Python code execution.
    """
    # Only attempt extraction from tools that explicitly handle Python
    if not hasattr(tool_call, "name"):
        logger.debug("Tool call missing 'name' attribute, skipping extraction")
        return None

    tool_name_lower = tool_call.name.lower()
    if tool_name_lower == "python":
        # Standalone "python" tool is a strong match
        pass
    elif tool_name_lower == "python_tool":
        # "python_tool" from mellea is a strong match
        pass
    elif "python" in tool_name_lower and any(
        keyword in tool_name_lower for keyword in ["executor", "interpreter", "runner"]
    ):
        # "python" paired with execution keywords is a strong match
        pass
    else:
        logger.debug(
            "Tool name '%s' does not match python execution pattern, skipping extraction",
            tool_call.name,
        )
        return None

    if not hasattr(tool_call, "args"):
        logger.debug(
            "Tool call '%s' missing 'args' attribute, skipping extraction",
            tool_call.name,
        )
        return None

    if tool_call.args is None:
        logger.debug(
            "Tool call '%s' has None args, skipping extraction", tool_call.name
        )
        return None

    # Try common field names in priority order
    field_names = ["code", "script", "command", "source"]
    for field in field_names:
        if field in tool_call.args:
            value = tool_call.args[field]
            if isinstance(value, str) and value.strip():
                logger.debug(
                    "Extracted code from tool '%s' field '%s'", tool_call.name, field
                )
                return value.strip()
            if not isinstance(value, str):
                logger.debug(
                    "Tool '%s' field '%s' is not a string (type: %s), skipping",
                    tool_call.name,
                    field,
                    type(value).__name__,
                )

    logger.debug(
        "No code-like fields found in tool '%s' args (available fields: %s)",
        tool_call.name,
        list(tool_call.args.keys()),
    )
    return None


def _has_python_code_listing(ctx: Context) -> ValidationResult:
    """Extract Python code from context, checking both text blocks and tool_calls.

    First attempts to extract code from text output (markdown/rst code blocks).
    If no text code is found, falls back to tool_calls (if present), using a
    heuristic search for code-like argument fields in Python-capable tools.

    Text-based extraction has higher priority than tool_calls to preserve
    user-visible code in the response. Tool extraction only targets tools with
    "python" in their name (case-insensitive).

    Args:
        ctx: Context containing model output.

    Returns:
        ValidationResult with extracted code or failure reason.
    """
    last_output = ctx.last_output()
    if last_output is None:
        return ValidationResult(result=False, reason="No output found in context")

    all_blocks: list[tuple[str, int]] = []

    # Step 1: Try extracting from text content (highest priority)
    if last_output.value is not None:
        content = last_output.value

        # Look for code blocks with python specifier
        import re

        # Pattern for ```python / `python blocks (Markdown and RST)
        python_blocks = re.findall(r"```?python\s*\n(.*?)\n```?", content, re.DOTALL)

        # Pattern for generic ``` / ` blocks (Markdown and RST)
        generic_blocks = re.findall(r"```?\s*\n(.*?)\n```?", content, re.DOTALL)

        # Add python blocks with high priority
        for block in python_blocks:
            all_blocks.append((block.strip(), _score_code_block(block.strip()) + 10))

        # Add generic blocks if they look like Python
        for block in generic_blocks:
            block = block.strip()
            if block and any(
                keyword in block
                for keyword in ["def ", "class ", "import ", "print(", "if __name__"]
            ):
                all_blocks.append((block, _score_code_block(block)))

    # Step 2: Fallback to tool_calls if no text code found
    if not all_blocks and last_output.tool_calls:
        for tool_name, tool_call in last_output.tool_calls.items():
            extracted = _extract_code_from_tool_call(tool_call)
            if extracted:
                # Score tool_call code with +5 bonus (vs +10 for text python blocks).
                # This keeps tool_calls below text-visible code in priority, while above
                # generic text blocks. Text blocks are preferred since they're visible to users.
                all_blocks.append((extracted, _score_code_block(extracted) + 5))

    if not all_blocks:
        return ValidationResult(result=False, reason="No Python code blocks found")

    # Return the highest scoring block
    best_block = max(all_blocks, key=lambda x: x[1])
    return ValidationResult(result=True, reason=best_block[0])


# endregion

# region execution validation


def _python_executes_without_error(
    ctx: Context, environment: ExecutionEnvironment, max_output_chars: int | None = None
) -> ValidationResult:
    """Validate that Python code executes without raising exceptions.

    Optionally enforces that captured output stays within size limit.

    Args:
        ctx: Context containing model output with code blocks.
        environment: Execution environment (static, local, or docker).
        max_output_chars: Maximum allowed stdout size in characters.
            None = no size check. Defaults to None.

    Returns:
        ValidationResult with execution success and optional output size details.
    """
    extraction_result = _has_python_code_listing(ctx)
    if not extraction_result.as_bool():
        return ValidationResult(
            result=False,
            reason=f"Could not extract Python code for execution: {extraction_result.reason}",
        )

    code = extraction_result.reason
    if code is None:
        return ValidationResult(
            result=False,
            reason="Code extraction returned None; this should not happen.",
        )

    result = environment.execute(code)

    if not result.success:
        return ValidationResult(
            result=False, reason=result.to_validationresult_reason()
        )

    if max_output_chars is not None:
        output_size = len(result.stdout or "")
        if output_size > max_output_chars:
            return ValidationResult(
                result=False,
                reason=f"Output size ({output_size} chars) exceeds limit ({max_output_chars}). "
                f"Code execution succeeded, but output validation failed.",
            )

    return ValidationResult(result=True, reason=result.to_validationresult_reason())


class PythonExecutionReq(Requirement):
    """Verifies that Python code runs without raising exceptions.

    Optionally validates that captured stdout does not exceed a size limit,
    eliminating the double-execution cost of separate OutputSizeLimit checks.

    Extracts the highest-scoring Python code block from the model's last output
    and validates or executes it according to the configured execution tier.

    Use ``execution_tier`` to select behavior by intent:

    - ``"static"`` (default) — parse and import-check only, no execution.
    - ``"local_unsafe"`` — subprocess execution, no policy restrictions.
    - ``"local"`` — subprocess execution with a declared capability policy.
    - ``"docker_unsafe"`` — Docker-isolated execution, no policy restrictions.
    - ``"docker"`` — Docker-isolated execution with a declared capability policy.

    Args:
        execution_tier (str): One of ``"static"``, ``"local_unsafe"``, ``"local"``,
            ``"docker_unsafe"``, or ``"docker"``.  Defaults to ``"static"``.
        policy (CapabilityPolicy | None): Override the tier's default policy.
            Ignored for ``"static"`` and unsafe tiers unless explicitly provided.
        allowed_imports (list[str] | None): Allowlist of importable top-level
            modules.  ``None`` allows any import.
        max_output_chars (int | None): Maximum allowed stdout size in characters.
            None = no size check (default). When set, adds output size validation
            in the same execution pass, avoiding the double-execution cost of using
            OutputSizeLimit. Only enforced for tiers that execute code (local_unsafe,
            local, docker_unsafe, docker); static tier skips output check.
        timeout (int | None): Deprecated.  Pass ``policy=CapabilityPolicy(timeout=N)``
            instead.  When provided, overrides the policy timeout.
        allow_unsafe_execution (bool): Deprecated.  Use
            ``execution_tier="local_unsafe"`` instead.
        use_sandbox (bool): Deprecated.  Use ``execution_tier="docker"`` instead.

    Attributes:
        validation_fn (Callable[[Context], ValidationResult]): The validation
            function attached to this requirement; always non-`None`.
    """

    def __init__(
        self,
        execution_tier: ExecutionTier = "static",
        *,
        policy: CapabilityPolicy | None = None,
        allowed_imports: list[str] | None = None,
        max_output_chars: int | None = None,
        # Deprecated kwargs — kept for backward compatibility
        timeout: int | None = None,
        allow_unsafe_execution: bool = False,
        use_sandbox: bool = False,
    ):
        """Initialize PythonExecutionReq with an execution tier and optional policy."""
        # Legacy positional-integer shim: old signature was PythonExecutionReq(timeout: int).
        if isinstance(execution_tier, int):
            warnings.warn(
                "Passing an integer as the first argument to PythonExecutionReq() is "
                "deprecated. The first parameter is now execution_tier (a string). "
                "Use PythonExecutionReq(policy=CapabilityPolicy(timeout=N)) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            timeout = execution_tier  # type: ignore[assignment]
            execution_tier = "static"

        # --- Deprecation shims ---
        _local_tiers = ("local_unsafe", "local")
        _docker_tiers = ("docker_unsafe", "docker")

        if allow_unsafe_execution:
            if execution_tier not in _local_tiers:
                if execution_tier in _docker_tiers:
                    # Caller is already on a docker tier — warn but don't downgrade.
                    warnings.warn(
                        f"allow_unsafe_execution is deprecated and has no effect when "
                        f"execution_tier='{execution_tier}' is already set. "
                        "Remove the flag.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                else:
                    warnings.warn(
                        "allow_unsafe_execution is deprecated. Use execution_tier='local_unsafe' instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    if execution_tier == "static":
                        # Promote to "local" when timeout is also set so the
                        # timeout shim below can synthesise a policy — "local_unsafe"
                        # has no policy and would silently discard the timeout value.
                        execution_tier = (
                            "local" if timeout is not None else "local_unsafe"
                        )

        if use_sandbox:
            if execution_tier not in _docker_tiers:
                # Only warn and promote when the flag actually changes something.
                warnings.warn(
                    "use_sandbox is deprecated. Use execution_tier='docker' instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                if execution_tier in ("static", "local_unsafe", "local"):
                    execution_tier = "docker"
            elif execution_tier == "docker_unsafe":
                # Already in Docker but without a policy — nudge toward 'docker'.
                warnings.warn(
                    "use_sandbox is deprecated. Use execution_tier='docker' (with policy) "
                    "instead of 'docker_unsafe' for capability enforcement.",
                    DeprecationWarning,
                    stacklevel=2,
                )

        if timeout is not None:
            if execution_tier == "static":
                warnings.warn(
                    "timeout has no effect on the static tier (no code is executed).",
                    DeprecationWarning,
                    stacklevel=2,
                )
            elif execution_tier in ("local_unsafe", "docker_unsafe"):
                warnings.warn(
                    f"timeout is ignored for the '{execution_tier}' tier (no policy is applied). "
                    "Use execution_tier='local' or 'docker' with policy=CapabilityPolicy(timeout=N) "
                    "to enforce a custom timeout.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    "timeout is deprecated. Pass policy=CapabilityPolicy(timeout=N) instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                if policy is None:
                    base = DOCKER_POLICY if execution_tier == "docker" else LOCAL_POLICY
                    policy = dataclasses.replace(base, timeout=timeout)
                else:
                    policy = dataclasses.replace(policy, timeout=timeout)

        if max_output_chars is not None and max_output_chars <= 0:
            raise ValueError(
                f"max_output_chars must be positive, got {max_output_chars}"
            )

        self._tier = execution_tier
        self._policy = policy
        self._allowed_imports = allowed_imports
        self._max_output_chars = max_output_chars

        environment: ExecutionEnvironment = make_execution_environment(
            tier=execution_tier, policy=policy, allowed_imports=allowed_imports
        )

        if execution_tier in ("local_unsafe", "local"):
            logger.warning(
                "⚠️ UNSAFE: Executing untrusted code without container isolation. "
                "Only use with trusted sources!"
            )

        tier_label = _tier_label(execution_tier, policy)

        output_note = ""
        if max_output_chars is not None and execution_tier != "static":
            output_note = f" Output limit: {max_output_chars} chars."

        super().__init__(
            description=f"The Python code should execute without errors ({tier_label}).{output_note}",
            validation_fn=lambda ctx: _python_executes_without_error(
                ctx, environment, max_output_chars=self._max_output_chars
            ),
            check_only=True,
        )

        self.validation_fn: Callable[[Context], ValidationResult]
        assert self.validation_fn is not None


def _tier_label(tier: str, policy: CapabilityPolicy | None) -> str:
    timeout = policy.timeout if policy is not None else None
    match tier:
        case "static":
            return "validation only"
        case "local_unsafe":
            effective = timeout if timeout is not None else LOCAL_POLICY.timeout
            return f"local execution, no policy (timeout: {effective}s)"
        case "local":
            effective = timeout if timeout is not None else LOCAL_POLICY.timeout
            return f"local execution with policy (timeout: {effective}s)"
        case "docker_unsafe":
            effective = timeout if timeout is not None else DOCKER_POLICY.timeout
            return f"sandbox execution, no policy (timeout: {effective}s)"
        case "docker":
            effective = timeout if timeout is not None else DOCKER_POLICY.timeout
            return f"sandbox execution with policy (timeout: {effective}s)"
        case _:
            return tier


# endregion
