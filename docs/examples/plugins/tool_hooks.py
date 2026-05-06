# pytest: ollama, e2e
#
# Tool hook plugins — safety and security policies for tool invocation.
#
# This example demonstrates four enforcement / repair patterns using
# TOOL_PRE_INVOKE and TOOL_POST_INVOKE hooks, built on top of the @tool
# decorator examples:
#
#   1. Tool allow list     — blocks any tool not on an explicit approved list
#   2. Argument validator  — inspects args before invocation (e.g., blocks
#                            disallowed patterns in calculator expressions)
#   3. Tool audit logger   — fire-and-forget logging of every tool call
#   4. Arg sanitizer       — auto-fixes tool args before invocation instead of
#                            blocking (e.g., strips unsafe chars from calculator
#                            expressions and normalises location strings)
#
# Run:
#   uv run python docs/examples/plugins/tool_hooks.py

import dataclasses
import logging

from mellea import start_session
from mellea.backends import ModelOption, tool
from mellea.plugins import (
    HookType,
    PluginMode,
    PluginResult,
    PluginSet,
    PluginViolationError,
    block,
    hook,
)
from mellea.stdlib.functional import _call_tools
from mellea.stdlib.requirements import uses_tool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("mellea").setLevel(logging.ERROR)
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
        return "Error: expression contains disallowed characters"
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


@hook(HookType.TOOL_PRE_INVOKE, mode=PluginMode.CONCURRENT, priority=5)
async def enforce_tool_allowlist(payload, _):
    """Block any tool not on the explicit allow list."""
    if payload.is_control_flow:
        return  # framework control-flow tools (e.g. final_answer) are exempt
    tool_name = payload.model_tool_call.name
    if tool_name not in ALLOWED_TOOLS:
        log.warning(
            "[allowlist] BLOCKED tool=%r — not in allowed set %s",
            tool_name,
            sorted(ALLOWED_TOOLS),
        )
        return block(
            f"Tool '{tool_name}' is not permitted",
            code="TOOL_NOT_ALLOWED",
            details={"tool": tool_name, "allowed": sorted(ALLOWED_TOOLS)},
        )
    log.info("[allowlist] permitted tool=%r", tool_name)


# ---------------------------------------------------------------------------
# Plugin 2 — Argument validator (enforce)
#
# Inspects the arguments before a tool is invoked.  For the calculator,
# reject expressions that contain characters outside the safe set.
# This runs after the allow list so it only sees permitted tools.
# ---------------------------------------------------------------------------

_CALCULATOR_ALLOWED_CHARS: frozenset[str] = frozenset("0123456789 +-*/(). ")


@hook(HookType.TOOL_PRE_INVOKE, mode=PluginMode.CONCURRENT, priority=10)
async def validate_tool_args(payload, _):
    """Validate tool arguments before invocation."""
    tool_name = payload.model_tool_call.name
    tool_args = payload.model_tool_call.args or {}
    if tool_name == "calculator":
        expression = tool_args.get("expression", "")
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
        log.info("[arg-validator] no arg validation required for tool=%r", tool_name)


# ---------------------------------------------------------------------------
# Plugin 3 — Tool audit logger (fire-and-forget)
#
# Records every tool invocation outcome for audit purposes.  Uses
# fire_and_forget so it never adds latency to the main execution path.
# ---------------------------------------------------------------------------


@hook(HookType.TOOL_POST_INVOKE, mode=PluginMode.FIRE_AND_FORGET)
async def audit_tool_calls(payload, _):
    """Log the result of every tool call for audit purposes."""
    status = "OK" if payload.success else "ERROR"
    tool_name = payload.model_tool_call.name
    log.info(
        "[audit] tool=%r status=%s latency=%dms args=%s",
        tool_name,
        status,
        payload.execution_time_ms,
        payload.model_tool_call.args,
    )
    if not payload.success and payload.error is not None:
        log.error("[audit] tool=%r error=%r", tool_name, str(payload.error))


# ---------------------------------------------------------------------------
# Plugin 4 — Arg sanitizer (repair)
#
# Instead of blocking, this plugin auto-fixes tool arguments before
# invocation.  Two repairs are applied:
#
#   calculator  — strips any character outside the safe arithmetic set so
#                 that the expression can still be evaluated.  A warning is
#                 logged showing what was removed.
#
#   get_weather — normalises the location string to title-case and strips
#                 leading/trailing whitespace (e.g. "  NEW YORK " → "New York").
#
# The plugin returns a modified ModelToolCall via model_copy so that the
# corrected args are what actually reaches the tool function.
# ---------------------------------------------------------------------------


@hook(HookType.TOOL_PRE_INVOKE, mode=PluginMode.CONCURRENT, priority=15)
async def sanitize_tool_args(payload, _) -> PluginResult:
    """Auto-fix tool arguments rather than blocking on unsafe input."""
    mtc = payload.model_tool_call
    tool_name = mtc.name
    args = dict(mtc.args or {})
    updated: dict[str, object] = {}

    if tool_name == "calculator":
        raw_expr = str(args.get("expression", ""))
        sanitized = "".join(c for c in raw_expr if c in _CALCULATOR_ALLOWED_CHARS)
        if sanitized != raw_expr:
            removed = set(raw_expr) - _CALCULATOR_ALLOWED_CHARS
            log.warning(
                "[sanitizer] calculator: stripped disallowed chars %s from expression=%r → %r",
                sorted(removed),
                raw_expr,
                sanitized,
            )
            updated["expression"] = sanitized

    elif tool_name == "get_weather":
        raw_location = str(args.get("location", ""))
        normalised = raw_location.strip().title()
        if normalised != raw_location:
            log.info(
                "[sanitizer] get_weather: normalised location %r → %r",
                raw_location,
                normalised,
            )
            updated["location"] = normalised

    if not updated:
        return None  # nothing changed — pass through as-is

    new_args = {**args, **updated}
    new_call = dataclasses.replace(mtc, args=new_args)
    modified = payload.model_copy(update={"model_tool_call": new_call})
    return PluginResult(continue_processing=True, modified_payload=modified)


# ---------------------------------------------------------------------------
# Compose into PluginSets for clean session-scoped registration
# ---------------------------------------------------------------------------

tool_security = PluginSet(
    "tool-security", [enforce_tool_allowlist, validate_tool_args, audit_tool_calls]
)

tool_sanitizer = PluginSet("tool-sanitizer", [sanitize_tool_args, audit_tool_calls])


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def _run_scenario(name: str, fn) -> None:
    """Run a scenario function, logging any PluginViolationError without halting."""
    log.info("=== %s ===", name)
    try:
        fn()
    except PluginViolationError as e:
        log.warning(
            "Execution blocked on %s: [%s] %s (plugin=%s)",
            e.hook_type,
            e.code,
            e.reason,
            e.plugin_name,
        )
    log.info("")


def scenario_1_allowed_tool(all_tools):
    """Scenario 1: allowed tool call (get_weather)."""
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
            log.error("Expected tool call but none were executed")


def scenario_2_blocked_tool(all_tools):
    """Scenario 2: blocked tool call (search_web not on allow list)."""
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


def scenario_3_safe_calculator(all_tools):
    """Scenario 3: safe calculator expression goes through."""
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
            log.error("Expected tool call but none were executed")


def scenario_4_blocked_calculator(all_tools):
    """Scenario 4: unsafe calculator expression is blocked."""
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


def scenario_5_sanitizer_calculator(all_tools):
    """Scenario 5: arg sanitizer auto-fixes an unsafe calculator expression."""
    with start_session(plugins=[tool_sanitizer]) as m:
        result = m.instruct(
            description=(
                "Use the calculator on this expression: "
                "6 * 7 + __import__('os').getpid()"
            ),
            requirements=[uses_tool("calculator")],
            model_options={ModelOption.TOOLS: all_tools},
            tool_calls=True,
        )
        tool_outputs = _call_tools(result, m.backend)
        if tool_outputs:
            log.info(
                "Sanitized expression evaluated — tool returned: %s",
                tool_outputs[0].content,
            )
        else:
            log.error("Expected sanitized tool call but none were executed")


def scenario_6_sanitizer_location(all_tools):
    """Scenario 6: arg sanitizer normalises a messy location string."""
    with start_session(plugins=[tool_sanitizer]) as m:
        result = m.instruct(
            description="What is the weather in '  NEW YORK  '?",
            requirements=[uses_tool("get_weather")],
            model_options={ModelOption.TOOLS: all_tools},
            tool_calls=True,
        )
        tool_outputs = _call_tools(result, m.backend)
        if tool_outputs:
            log.info(
                "Weather fetched with normalised location — tool returned: %s",
                tool_outputs[0].content,
            )
        else:
            log.error("Expected tool call but none were executed")


# ---------------------------------------------------------------------------
# Main — six scenarios
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("--- Tool hook plugins example ---")
    log.info("")

    all_tools = [get_weather, search_web, calculate]

    _run_scenario(
        "Scenario 1: allowed tool — get_weather",
        lambda: scenario_1_allowed_tool(all_tools),
    )
    _run_scenario(
        "Scenario 2: blocked tool — search_web not on allow list",
        lambda: scenario_2_blocked_tool(all_tools),
    )
    _run_scenario(
        "Scenario 3: safe calculator expression",
        lambda: scenario_3_safe_calculator(all_tools),
    )
    _run_scenario(
        "Scenario 4: unsafe calculator expression blocked",
        lambda: scenario_4_blocked_calculator(all_tools),
    )
    _run_scenario(
        "Scenario 5: arg sanitizer auto-fixes calculator expression",
        lambda: scenario_5_sanitizer_calculator(all_tools),
    )
    _run_scenario(
        "Scenario 6: arg sanitizer normalises location in get_weather",
        lambda: scenario_6_sanitizer_location(all_tools),
    )
