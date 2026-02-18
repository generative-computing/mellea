# pytest: ollama, llm
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
# Run:
#   uv run python docs/examples/plugins/tool_hooks.py

import logging
import sys

from mellea import start_session
from mellea.backends import ModelOption, tool
from mellea.stdlib.functional import _call_tools
from mellea.stdlib.requirements import uses_tool
from mellea.plugins import HookType, PluginMode, PluginSet, block, hook

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
    return {
        "location": location,
        "days": str(days),
        "forecast": "sunny",
        "temperature": "72",
    }


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
# Compose into a PluginSet for clean session-scoped registration
# ---------------------------------------------------------------------------

tool_security = PluginSet(
    "tool-security", [enforce_tool_allowlist, validate_tool_args, audit_tool_calls]
)


# ---------------------------------------------------------------------------
# Main — four scenarios
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("--- Tool hook plugins example ---")
    log.info("")

    all_tools = [get_weather, search_web, calculate]

    # --- Scenario 1: allowed tool call (get_weather) ---
    log.info("=== Scenario 1: allowed tool — get_weather ===")
    with start_session(plugins=[tool_security]) as m:
        result = m.instruct(
            description="What is the weather in Boston for the next 3 days?",
            requirements=[uses_tool("get_weather")],
            model_options={ModelOption.TOOLS: all_tools},
            tool_calls=True,
        )
        tool_outputs = _call_tools(result, m.backend)
        if tool_outputs:
            log.info("Tool returned: %s", tool_outputs[0].content)
        else:
            log.error("Expected tool call but none were executed — exiting")
            sys.exit(1)
    log.info("")

    # --- Scenario 2: blocked tool call (search_web is not on the allow list) ---
    log.info("=== Scenario 2: blocked tool — search_web not on allow list ===")
    with start_session(plugins=[tool_security]) as m:
        result = m.instruct(
            description="Search the web for the latest Python news.",
            requirements=[uses_tool("search_web")],
            model_options={ModelOption.TOOLS: all_tools},
            tool_calls=True,
        )
        tool_outputs = _call_tools(result, m.backend)
        if not tool_outputs:
            log.info("Tool call was blocked — outputs list is empty, as expected")
        else:
            log.warning("Expected tool to be blocked but it executed: %s", tool_outputs)
    log.info("")

    # --- Scenario 3: safe calculator expression goes through ---
    log.info("=== Scenario 3: safe calculator expression ===")
    with start_session(plugins=[tool_security]) as m:
        result = m.instruct(
            description="Use the calculator to compute 6 * 7.",
            requirements=[uses_tool("calculator")],
            model_options={ModelOption.TOOLS: all_tools},
            tool_calls=True,
        )
        tool_outputs = _call_tools(result, m.backend)
        if tool_outputs:
            log.info("Tool returned: %s", tool_outputs[0].content)
        else:
            log.error("Expected tool call but none were executed — exiting")
            sys.exit(1)
    log.info("")

    # --- Scenario 4: unsafe calculator expression is blocked ---
    log.info("=== Scenario 4: unsafe calculator expression blocked ===")
    with start_session(plugins=[tool_security]) as m:
        result = m.instruct(
            description=(
                "Use the calculator on this expression: "
                "__builtins__['print']('injected')"
            ),
            requirements=[uses_tool("calculator")],
            model_options={ModelOption.TOOLS: all_tools},
            tool_calls=True,
        )
        tool_outputs = _call_tools(result, m.backend)
        if not tool_outputs:
            log.info("Tool call was blocked — outputs list is empty, as expected")
        else:
            log.warning("Expected tool to be blocked but it executed: %s", tool_outputs)
