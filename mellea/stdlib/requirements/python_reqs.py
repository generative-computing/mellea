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

from ...core import Context, MelleaLogger, Requirement, ValidationResult

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


def _has_python_code_listing(ctx: Context) -> ValidationResult:
    """Extract Python code from context."""
    last_output = ctx.last_output()
    if last_output is None or last_output.value is None:
        return ValidationResult(result=False, reason="No output found in context")

    content = last_output.value

    # Look for code blocks with python specifier
    import re

    # Pattern for ``python ... `` blocks
    python_blocks = re.findall(r"``python\s*\n(.*?)\n``", content, re.DOTALL)

    # Pattern for generic ``` blocks
    generic_blocks = re.findall(r"``\s*\n(.*?)\n``", content, re.DOTALL)

    all_blocks = []

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

    if not all_blocks:
        return ValidationResult(result=False, reason="No Python code blocks found")

    # Return the highest scoring block
    best_block = max(all_blocks, key=lambda x: x[1])
    return ValidationResult(result=True, reason=best_block[0])


# endregion

# region execution validation


def _python_executes_without_error(
    ctx: Context, environment: ExecutionEnvironment
) -> ValidationResult:
    """Validate that Python code executes without raising exceptions.

    First extracts the highest-scoring Python code block from the context,
    then validates/executes it using the given environment.
    """
    extraction_result = _has_python_code_listing(ctx)
    if not extraction_result.as_bool():
        return ValidationResult(
            result=False,
            reason=f"Could not extract Python code for execution: {extraction_result.reason}",
        )

    code = extraction_result.reason
    assert code is not None

    result = environment.execute(code)
    return ValidationResult(
        result=result.success, reason=result.to_validationresult_reason()
    )


class PythonExecutionReq(Requirement):
    """Verifies that Python code runs without raising exceptions.

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

        self._tier = execution_tier
        self._policy = policy
        self._allowed_imports = allowed_imports

        environment: ExecutionEnvironment = make_execution_environment(
            tier=execution_tier, policy=policy, allowed_imports=allowed_imports
        )

        if execution_tier in ("local_unsafe", "local"):
            logger.warning(
                "⚠️ UNSAFE: Executing untrusted code without container isolation. "
                "Only use with trusted sources!"
            )

        tier_label = _tier_label(execution_tier, policy)

        super().__init__(
            description=f"The Python code should execute without errors ({tier_label}).",
            validation_fn=lambda ctx: _python_executes_without_error(ctx, environment),
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
            return f"docker execution, no policy (timeout: {effective}s)"
        case "docker":
            effective = timeout if timeout is not None else DOCKER_POLICY.timeout
            return f"docker execution with policy (timeout: {effective}s)"
        case _:
            return tier


# endregion
