#
# Tool hook plugins — safety and security policies for tool invocation.
#
# This example demonstrates three enforcement patterns using TOOL_PRE_INVOKE
# and TOOL_POST_INVOKE hooks, built on top of the @tool decorator examples:
#
#   1. Tool allow list     — blocks any tool not on an explicit approved list
#   2. Argument validator  — inspects args before invocation (e.g., blocks
#                            disallowed patterns in calculator expressions)
#   3. Tool audit logger   — fire-and-forget logging of every tool call
#
# Scenarios are driven directly through _acall_tools() with hand-crafted
# ModelToolCall objects so that plugin behavior is always exercised regardless
# of what the LLM chooses to do.
#
# Run:
#   uv run python docs/examples/plugins/tool_hooks.py

import asyncio
import logging

from mellea.backends import tool
from mellea.core.base import ModelOutputThunk, ModelToolCall
from mellea.stdlib.requirements import uses_tool
from mellea.plugins import HookType, PluginMode, PluginSet, block, hook, register
from mellea.stdlib.functional import _acall_tools

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("tool_hooks")


# ---------------------------------------------------------------------------
# Tools (same as tool_decorator_example.py)
# ---------------------------------------------------------------------------


@tool
def get_weather(location: str, days: int = 1) -> dict:
    """Get weather forecast for a location.

    Args:
        location: City name
        days: Number of days to forecast (default: 1)
    """
    return {"location": location, "days": days, "forecast": "sunny", "temperature": 72}


@tool
def search_web(query: str, max_results: int = 5) -> list[str]:
    """Search the web for information.

    Args:
        query: Search query
        max_results: Maximum number of results to return
    """
    return [f"Result {i + 1} for '{query}'" for i in range(max_results)]


@tool(name="calculator")
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 2 * 3")
    """
    # Only digits, whitespace, and basic arithmetic operators are permitted.
    # This is enforced upstream by validate_tool_args, but the function
    # applies its own check as a defence-in-depth measure.
    allowed = set("0123456789 +-*/(). ")
    if not set(expression).issubset(allowed):
        return f"Error: expression contains disallowed characters"
    try:
        # Safe: only reaches here when characters are in the allowed set
        result = _safe_calc(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e!s}"


def _safe_calc(expression: str) -> float:
    """Evaluate a restricted arithmetic expression (no builtins, no names)."""
    import operator as op
    import re

    tokens = re.findall(r"[\d.]+|[+\-*/()]", expression)
    # Build a simple recursive-descent expression for +, -, *, /, ()
    pos = [0]

    def parse_expr():
        left = parse_term()
        while pos[0] < len(tokens) and tokens[pos[0]] in ("+", "-"):
            tok = tokens[pos[0]]
            pos[0] += 1
            right = parse_term()
            left = op.add(left, right) if tok == "+" else op.sub(left, right)
        return left

    def parse_term():
        left = parse_factor()
        while pos[0] < len(tokens) and tokens[pos[0]] in ("*", "/"):
            tok = tokens[pos[0]]
            pos[0] += 1
            right = parse_factor()
            left = op.mul(left, right) if tok == "*" else op.truediv(left, right)
        return left

    def parse_factor():
        tok = tokens[pos[0]]
        if tok == "(":
            pos[0] += 1
            val = parse_expr()
            pos[0] += 1  # consume ")"
            return val
        pos[0] += 1
        return float(tok)

    return parse_expr()


# ---------------------------------------------------------------------------
# Plugin 1 — Tool allow list (enforce)
#
# Only tools explicitly listed in ALLOWED_TOOLS may be called.  Any tool call
# for an unlisted tool is blocked before it reaches the function.
# ---------------------------------------------------------------------------

ALLOWED_TOOLS: frozenset[str] = frozenset({"get_weather", "calculator"})


@hook(HookType.TOOL_PRE_INVOKE, mode=PluginMode.ENFORCE, priority=5)
async def enforce_tool_allowlist(payload, ctx):
    """Block any tool not on the explicit allow list."""
    if payload.tool_name not in ALLOWED_TOOLS:
        log.warning(
            "[allowlist] BLOCKED tool=%r — not in allowed set %s",
            payload.tool_name,
            sorted(ALLOWED_TOOLS),
        )
        return block(
            f"Tool '{payload.tool_name}' is not permitted",
            code="TOOL_NOT_ALLOWED",
            details={"tool": payload.tool_name, "allowed": sorted(ALLOWED_TOOLS)},
        )
    log.info("[allowlist] permitted tool=%r", payload.tool_name)


# ---------------------------------------------------------------------------
# Plugin 2 — Argument validator (enforce)
#
# Inspects the arguments before a tool is invoked.  For the calculator,
# reject expressions that contain characters outside the safe set.
# This runs after the allow list so it only sees permitted tools.
# ---------------------------------------------------------------------------

_CALCULATOR_ALLOWED_CHARS: frozenset[str] = frozenset("0123456789 +-*/(). ")


@hook(HookType.TOOL_PRE_INVOKE, mode=PluginMode.ENFORCE, priority=10)
async def validate_tool_args(payload, ctx):
    """Validate tool arguments before invocation."""
    if payload.tool_name == "calculator":
        expression = payload.tool_args.get("expression", "")
        disallowed = set(expression) - _CALCULATOR_ALLOWED_CHARS
        if disallowed:
            log.warning(
                "[arg-validator] BLOCKED calculator expression=%r (disallowed chars: %s)",
                expression,
                disallowed,
            )
            return block(
                f"Calculator expression contains disallowed characters: {disallowed}",
                code="UNSAFE_EXPRESSION",
                details={"expression": expression, "disallowed": sorted(disallowed)},
            )
        log.info("[arg-validator] calculator expression=%r is safe", expression)
    else:
        log.info(
            "[arg-validator] no arg validation required for tool=%r", payload.tool_name
        )


# ---------------------------------------------------------------------------
# Plugin 3 — Tool audit logger (fire-and-forget)
#
# Records every tool invocation outcome for audit purposes.  Uses
# fire_and_forget so it never adds latency to the main execution path.
# ---------------------------------------------------------------------------


@hook(HookType.TOOL_POST_INVOKE, mode=PluginMode.FIRE_AND_FORGET)
async def audit_tool_calls(payload, ctx):
    """Log the result of every tool call for audit purposes."""
    status = "OK" if payload.success else "ERROR"
    log.info(
        "[audit] tool=%r status=%s latency=%dms args=%s",
        payload.tool_name,
        status,
        payload.execution_time_ms,
        payload.tool_args,
    )
    if not payload.success and payload.error is not None:
        log.error("[audit] tool=%r error=%r", payload.tool_name, str(payload.error))


# ---------------------------------------------------------------------------
# Compose into a PluginSet and register globally
# ---------------------------------------------------------------------------

tool_security = PluginSet(
    "tool-security", [enforce_tool_allowlist, validate_tool_args, audit_tool_calls]
)
register(tool_security)


# ---------------------------------------------------------------------------
# Helpers
#
# _make_result() builds a ModelOutputThunk that looks exactly like what the
# LLM would produce when it decides to call a tool.  This lets us exercise the
# full tool hook pipeline without depending on the model's behaviour.
# ---------------------------------------------------------------------------


def _make_result(tool_obj, **args) -> ModelOutputThunk:
    """Wrap a single tool call into a ModelOutputThunk."""
    tool_call = ModelToolCall(name=tool_obj.name, func=tool_obj, args=args)
    return ModelOutputThunk(value="", tool_calls={tool_obj.name: tool_call})


class _NoOpBackend:
    """Minimal backend stand-in that skips LLM calls (no FormatterBackend)."""

    model_id = "test"


_backend = _NoOpBackend()


# ---------------------------------------------------------------------------
# Main — four scenarios driven directly through _acall_tools()
# ---------------------------------------------------------------------------


async def main() -> None:
    log.info("--- Tool hook plugins example ---")
    log.info("")

    # --- Scenario 1: allowed tool — get_weather ---
    log.info("=== Scenario 1: allowed tool — get_weather ===")
    outputs = await _acall_tools(
        _make_result(get_weather, location="Boston", days=3),
        _backend,  # type: ignore[arg-type]
    )
    if outputs:
        log.info("Tool returned: %s", outputs[0].content)
    log.info("")

    # --- Scenario 2: blocked tool — search_web is not on the allow list ---
    log.info("=== Scenario 2: blocked tool — search_web not on allow list ===")
    outputs = await _acall_tools(
        _make_result(search_web, query="Python news", max_results=3),
        _backend,  # type: ignore[arg-type]
    )
    if not outputs:
        log.info("Tool call was blocked — outputs list is empty, as expected")
    log.info("")

    # --- Scenario 3: allowed tool with safe calculator expression ---
    log.info("=== Scenario 3: safe calculator expression — allowed ===")
    outputs = await _acall_tools(
        _make_result(calculate, expression="6 * 7"),
        _backend,  # type: ignore[arg-type]
    )
    if outputs:
        log.info("Tool returned: %s", outputs[0].content)
    log.info("")

    # --- Scenario 4: calculator with disallowed characters — blocked ---
    log.info("=== Scenario 4: unsafe calculator expression — blocked ===")
    outputs = await _acall_tools(
        _make_result(calculate, expression="__builtins__['open']('/etc/passwd')"),
        _backend,  # type: ignore[arg-type]
    )
    if not outputs:
        log.info("Tool call was blocked — outputs list is empty, as expected")
    log.info("")


if __name__ == "__main__":
    asyncio.run(main())
