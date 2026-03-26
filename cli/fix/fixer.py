"""AST-based detection and source transformation for async call fixes.

Targets: aact, ainstruct, aquery calls that return uncomputed ModelOutputThunk.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from cli.fix import FixMode

# Functions that need fixing.
TARGET_FUNCTIONS = {"aact", "ainstruct", "aquery"}

# The canonical module path for the functional API.
FUNCTIONAL_MODULE = "mellea.stdlib.functional"

# Modules that export session classes / factory functions.
SESSION_MODULES = {"mellea", "mellea.stdlib.session"}
# Names from those modules that produce session objects with aact/ainstruct/aquery methods.
SESSION_FACTORY_NAMES = {"MelleaSession", "start_session"}

# Directories to skip during traversal.
SKIP_DIRS = {"__pycache__", ".git", ".venv", "node_modules"}


@dataclass
class FixLocation:
    """Location of the fix along with some additional details."""
    filepath: Path
    line: int
    col_offset: int
    function_name: str  # "aact", "ainstruct", or "aquery"
    call_style: str  # "functional" or "session"
    target_variable: str | None  # MOT variable name from assignment
    context_variable: str | None


@dataclass
class FixResult:
    """Full summary of fixes."""
    locations: list[FixLocation]
    total_fixes: int
    files_affected: int


@dataclass
class _MelleaImports:
    """Names resolved from imports of mellea functional and session APIs in a file."""

    # Bare names imported directly: `from mellea.stdlib.functional import aact`
    functional_bare_names: set[str]
    # Module aliases: `import mellea.stdlib.functional as mfuncs` or
    # `from mellea.stdlib import functional [as alias]`
    functional_module_aliases: set[str]
    # Whether the file imports any mellea session types (MelleaSession, start_session).
    # If True, `obj.aact(...)` calls on unknown objects are treated as session calls.
    has_session_imports: bool


def _resolve_mellea_imports(tree: ast.Module) -> _MelleaImports:
    """Walk top-level imports and resolve which names refer to mellea APIs."""
    functional_bare: set[str] = set()
    functional_aliases: set[str] = set()
    has_session = False

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # `import mellea.stdlib.functional` or
                # `import mellea.stdlib.functional as mfuncs`
                if alias.name == FUNCTIONAL_MODULE:
                    local = (
                        alias.asname if alias.asname else alias.name.rsplit(".", 1)[-1]
                    )
                    functional_aliases.add(local)
                # `import mellea` or `import mellea.stdlib.session`
                if alias.name in SESSION_MODULES:
                    has_session = True

        elif isinstance(node, ast.ImportFrom):
            full_module = node.module or ""

            if full_module == FUNCTIONAL_MODULE:
                # `from mellea.stdlib.functional import aact, ainstruct`
                for alias in node.names:
                    local = alias.asname if alias.asname else alias.name
                    if alias.name in TARGET_FUNCTIONS:
                        functional_bare.add(local)

            elif full_module == "mellea.stdlib":
                # `from mellea.stdlib import functional [as alias]`
                # `from mellea.stdlib import session [as alias]`
                for alias in node.names:
                    if alias.name == "functional":
                        local = alias.asname if alias.asname else alias.name
                        functional_aliases.add(local)
                    elif alias.name == "session":
                        has_session = True

            # Session imports: `from mellea import MelleaSession` or
            # `from mellea.stdlib.session import start_session`
            if full_module in SESSION_MODULES:
                for alias in node.names:
                    if alias.name in SESSION_FACTORY_NAMES:
                        has_session = True

    return _MelleaImports(
        functional_bare_names=functional_bare,
        functional_module_aliases=functional_aliases,
        has_session_imports=has_session,
    )


def _get_call_info(
    call_node: ast.Call, imports: _MelleaImports
) -> tuple[str, str] | None:
    """Extract (function_name, call_style) from a Call node, or None if not a target."""
    func = call_node.func

    # Bare call: `aact(...)` — functional only if name was imported from the functional module.
    if isinstance(func, ast.Name) and func.id in imports.functional_bare_names:
        return func.id, "functional"

    # Attribute call: `something.aact(...)`
    if isinstance(func, ast.Attribute) and func.attr in TARGET_FUNCTIONS:
        if isinstance(func.value, ast.Name):
            if func.value.id in imports.functional_module_aliases:
                return func.attr, "functional"
            # Only treat as session method if the file imports mellea session types.
            if imports.has_session_imports:
                return func.attr, "session"

    return None


def _get_keyword_value(call_node: ast.Call, name: str) -> ast.expr | None:
    """Get the value node of a keyword argument, or None if absent."""
    for kw in call_node.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _is_none_literal(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _is_true_literal(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _should_skip(call_node: ast.Call, func_name: str) -> bool:
    """Return True if this call should NOT be fixed."""
    # Already has await_result=True
    ar = _get_keyword_value(call_node, "await_result")
    if ar is not None and _is_true_literal(ar):
        return True

    # For aact/ainstruct: skip if no explicit strategy=None
    # (absent strategy = default non-None = already computed)
    if func_name in ("aact", "ainstruct"):
        strategy_val = _get_keyword_value(call_node, "strategy")
        if strategy_val is None:
            # No strategy keyword at all — default is non-None
            return True
        if not _is_none_literal(strategy_val):
            # Has strategy but it's not None (e.g., strategy=my_strategy)
            return True

    # aquery: no strategy param, always affected — don't skip based on strategy

    # Has return_sampling_results=True
    rsr = _get_keyword_value(call_node, "return_sampling_results")
    if rsr is not None and _is_true_literal(rsr):
        return True

    return False


def _extract_variables(
    stmt: ast.stmt, call_style: str
) -> tuple[str | None, str | None]:
    """Extract (target_variable, context_variable) from the assignment statement."""
    targets = None
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        targets = stmt.targets[0]
    elif isinstance(stmt, ast.AnnAssign) and stmt.target:
        targets = stmt.target

    if targets is None:
        return None, None

    # Tuple unpack: result, ctx = await ...
    if isinstance(targets, ast.Tuple) and len(targets.elts) >= 2:
        first = targets.elts[0]
        second = targets.elts[1]
        target_var = first.id if isinstance(first, ast.Name) else None
        ctx_var = second.id if isinstance(second, ast.Name) else None
        return target_var, ctx_var

    # Simple assign: result = await session.aact(...)
    if isinstance(targets, ast.Name):
        return targets.id, None

    return None, None


def _has_existing_consumption(following_stmts: list[ast.stmt], var_name: str) -> bool:
    """Check if subsequent statements already consume/await the thunk variable.

    Looks for:
    - ``while not <var>.is_computed(): ...``
    - ``await <var>.avalue()``
    - ``await <var>.astream()``

    Recurses into nested blocks (if/for/try/with) via ast.walk.
    """
    for stmt in following_stmts:
        for node in ast.walk(stmt):
            # Pattern A: while not var.is_computed(): ...
            if isinstance(node, ast.While):
                test = node.test
                if (
                    isinstance(test, ast.UnaryOp)
                    and isinstance(test.op, ast.Not)
                    and isinstance(test.operand, ast.Call)
                    and isinstance(test.operand.func, ast.Attribute)
                    and test.operand.func.attr == "is_computed"
                    and isinstance(test.operand.func.value, ast.Name)
                    and test.operand.func.value.id == var_name
                ):
                    return True

            # Pattern B/C: await var.avalue() or await var.astream()
            if isinstance(node, ast.Await):
                awaited = node.value
                if (
                    isinstance(awaited, ast.Call)
                    and isinstance(awaited.func, ast.Attribute)
                    and awaited.func.attr in ("avalue", "astream")
                    and isinstance(awaited.func.value, ast.Name)
                    and awaited.func.value.id == var_name
                ):
                    return True
    return False


def _iter_bodies(node: ast.AST):
    """Yield each statement-list body contained in *node*."""
    for attr in ("body", "orelse", "finalbody", "handlers"):
        child = getattr(node, attr, None)
        if child is None:
            continue
        if attr == "handlers":
            for handler in child:
                yield from _iter_bodies(handler)
        elif isinstance(child, list):
            yield child


def find_fixable_calls(source: str, filepath: Path) -> list[FixLocation]:
    """Analyze source code and return locations of calls that need fixing."""
    if not source.strip():
        return []

    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        return []

    imports = _resolve_mellea_imports(tree)
    locations: list[FixLocation] = []

    def _visit_body(stmts: list[ast.stmt]) -> None:
        for i, node in enumerate(stmts):
            # Recurse into nested bodies first (if/for/with/try/async for/async with/class/func)
            for body in _iter_bodies(node):
                _visit_body(body)

            # We only care about statements that contain Await expressions
            if not isinstance(node, (ast.Expr, ast.Assign, ast.AnnAssign)):
                continue

            # Find the Await node
            await_node = None
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Await):
                await_node = node.value
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                val = node.value
                if isinstance(val, ast.Await):
                    await_node = val

            if await_node is None:
                continue

            # The awaited expression should be a Call
            if not isinstance(await_node.value, ast.Call):
                continue

            call_node = await_node.value
            info = _get_call_info(call_node, imports)
            if info is None:
                continue

            func_name, call_style = info

            if _should_skip(call_node, func_name):
                continue

            target_var, ctx_var = _extract_variables(node, call_style)

            # Check if subsequent statements already handle the thunk
            if target_var and _has_existing_consumption(stmts[i + 1 :], target_var):
                continue

            locations.append(
                FixLocation(
                    filepath=filepath,
                    line=call_node.lineno,
                    col_offset=call_node.col_offset,
                    function_name=func_name,
                    call_style=call_style,
                    target_variable=target_var,
                    context_variable=ctx_var,
                )
            )

    _visit_body(tree.body)
    return locations


def _apply_add_await_result(lines: list[str], call_node: ast.Call) -> None:
    """Insert await_result=True into the call, modifying lines in place."""
    # Check if there's an existing await_result=False to replace
    for kw in call_node.keywords:
        if kw.arg == "await_result":
            # Replace the value (False -> True)
            line_idx = kw.value.lineno - 1
            line = lines[line_idx]
            # Find the keyword in the line and replace the value
            col = kw.value.col_offset
            end_col = kw.value.end_col_offset
            lines[line_idx] = line[:col] + "True" + line[end_col:]
            return

    # Find the closing paren of the call
    assert call_node.end_lineno is not None and call_node.end_col_offset is not None
    end_line = call_node.end_lineno - 1
    end_col = call_node.end_col_offset - 1  # points to ')'

    line = lines[end_line]

    # Check if it's a multi-line call (call spans multiple lines)
    is_multiline = call_node.lineno != call_node.end_lineno

    if is_multiline:
        # For multi-line calls, insert a new line before the closing paren
        # The closing paren is at lines[end_line][end_col]
        paren_line = lines[end_line]
        indent = ""
        for ch in paren_line:
            if ch in (" ", "\t"):
                indent += ch
            else:
                break

        # Determine the indentation of keyword args (one level deeper than closing paren)
        # Look at the line above the closing paren for reference
        kw_indent = indent + "    "

        # Check if line before closing paren has trailing comma
        prev_content_line_idx = end_line - 1
        prev_line = lines[prev_content_line_idx].rstrip()
        if prev_line.endswith(","):
            # Already has trailing comma, just add new kwarg line
            lines[end_line] = kw_indent + "await_result=True,\n" + paren_line
        else:
            # Add comma to previous line, then new kwarg line
            lines[prev_content_line_idx] = prev_line + ",\n"
            lines[end_line] = kw_indent + "await_result=True,\n" + paren_line
    else:
        # Single-line call: insert before the closing paren
        before_paren = line[:end_col].rstrip()
        after_paren = line[end_col:]

        if before_paren.endswith(","):
            # Trailing comma: insert with space
            lines[end_line] = before_paren + " await_result=True" + after_paren
        else:
            lines[end_line] = before_paren + ", await_result=True" + after_paren


def _apply_add_stream_loop(
    lines: list[str], stmt_node: ast.stmt, loc: FixLocation
) -> None:
    """Insert a stream loop after the assignment statement."""
    if loc.target_variable is None:
        return

    # Find indentation of the statement
    stmt_line = lines[stmt_node.lineno - 1]
    indent = ""
    for ch in stmt_line:
        if ch in (" ", "\t"):
            indent += ch
        else:
            break

    var = loc.target_variable
    loop_code = (
        f"{indent}while not {var}.is_computed():\n{indent}    await {var}.astream()\n"
    )

    # Insert after the end of the statement
    assert stmt_node.end_lineno is not None
    lines.insert(stmt_node.end_lineno, loop_code)


def fix_file(filepath: Path, mode: FixMode, dry_run: bool = False) -> list[FixLocation]:
    """Fix a single file. Returns list of fix locations found."""
    source = filepath.read_text()
    locations = find_fixable_calls(source, filepath)

    if not locations or dry_run:
        return locations

    # Re-parse to get AST nodes for transformation
    tree = ast.parse(source, filename=str(filepath))

    # Collect (call_node, stmt_node) pairs for each location
    call_stmt_pairs: list[tuple[ast.Call, ast.stmt, FixLocation]] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Expr, ast.Assign, ast.AnnAssign)):
            continue

        await_node = None
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Await):
            await_node = node.value
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            if isinstance(node.value, ast.Await):
                await_node = node.value

        if await_node is None or not isinstance(await_node.value, ast.Call):
            continue

        call_node = await_node.value

        # Match by line number
        for loc in locations:
            if call_node.lineno == loc.line and call_node.col_offset == loc.col_offset:
                call_stmt_pairs.append((call_node, node, loc))
                break

    # Sort by line number descending so edits don't shift positions
    call_stmt_pairs.sort(key=lambda x: x[0].lineno, reverse=True)

    lines = source.splitlines(keepends=True)
    # Ensure last line has newline
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"

    for call_node, stmt_node, loc in call_stmt_pairs:
        if mode == FixMode.ADD_AWAIT_RESULT:
            _apply_add_await_result(lines, call_node)
        elif mode == FixMode.ADD_STREAM_LOOP:
            _apply_add_stream_loop(lines, stmt_node, loc)

    filepath.write_text("".join(lines))
    return locations


def fix_path(path: Path, mode: FixMode, dry_run: bool = False) -> FixResult:
    """Fix a file or directory recursively. Returns aggregated results."""
    all_locations: list[FixLocation] = []
    files_affected = 0

    if path.is_file():
        files = [path]
    else:
        files = sorted(path.rglob("*.py"))

    for f in files:
        # Skip excluded directories
        parts = f.relative_to(path).parts if path.is_dir() else ()
        if any(part in SKIP_DIRS for part in parts):
            continue

        locs = fix_file(f, mode, dry_run=dry_run)
        if locs:
            all_locations.extend(locs)
            files_affected += 1

    return FixResult(
        locations=all_locations,
        total_fixes=len(all_locations),
        files_affected=files_affected,
    )
