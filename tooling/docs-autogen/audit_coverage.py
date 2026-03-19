#!/usr/bin/env python3
"""Audit API documentation coverage and docstring quality.

Discovers all public classes and functions in mellea/ and cli/ using Griffe,
then checks which ones have generated MDX documentation. Constants and module
attributes are excluded from the count — they are not expected to have
standalone documentation.

With --quality, also audits docstring quality: flags missing docstrings,
very short docstrings, and functions whose Args/Returns sections are absent.
"""

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path

try:
    import griffe
except ImportError:
    print("ERROR: griffe not installed. Run: uv pip install griffe", file=sys.stderr)
    sys.exit(1)


# Modules that are confirmed internal but whose parent __init__.py imports nothing
# (making the import-based check indeterminate).  Must stay in sync with generate-ast.py.
_CONFIRMED_INTERNAL_MODULES: frozenset[str] = frozenset(
    {"json_util", "backend_instrumentation"}
)


def _imported_submodule_names(init_path: Path) -> set[str] | None:
    """Return submodule names imported via relative imports in init_path.

    Returns None if the file cannot be parsed (treat as indeterminate).
    Returns an empty set if the file is readable but has no relative imports.
    """
    try:
        tree = ast.parse(init_path.read_text())
    except Exception:
        return None
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level > 0 and node.module:
            result.add(node.module.split(".")[0])
    return result


def _is_public_submodule(submodule_name: str, submodule_filepath: Path | None) -> bool:
    """Return True if a submodule should be treated as part of the public API.

    Mirrors the filter in generate-ast.py's remove_internal_modules().
    """
    if submodule_name in _CONFIRMED_INTERNAL_MODULES:
        return False
    if submodule_filepath is None:
        return True  # conservative: keep if we can't determine
    # The submodule filepath is either a .py file or a package __init__.py.
    # The parent __init__.py to check is one level up.
    parent_init = submodule_filepath.parent.parent / "__init__.py"
    if not parent_init.exists():
        return True  # conservative
    subs = _imported_submodule_names(parent_init)
    if subs is None or not subs:
        return True  # conservative: can't determine, keep
    return submodule_name in subs


def _load_package(source_dir: Path, package_name: str):
    """Load a package with Griffe. Returns the package object or None on failure."""
    try:
        # try_relative_path=False ensures Griffe only searches the explicit
        # search_paths and does not fall back to CWD, which avoids loading a
        # same-named package from the project root when --source-dir points
        # elsewhere (e.g. auditing mellea-b while running from mellea-d).
        search_path = str(source_dir.parent.resolve())
        return griffe.load(
            source_dir.name, search_paths=[search_path], try_relative_path=False
        )
    except Exception as e:
        print(f"WARNING: Failed to load {source_dir}: {e}", file=sys.stderr)
        return None


def discover_public_symbols(
    source_dir: Path, package_name: str
) -> dict[str, list[str]]:
    """Discover all public symbols using Griffe.

    Args:
        source_dir: Root directory to scan (e.g., mellea/ or cli/)
        package_name: Package name to prepend (e.g., "mellea" or "cli")

    Returns:
        Dict mapping full symbol paths to empty lists (for compatibility)
        Example: {"mellea.core.base.Component": [], "mellea.core.base.generative": []}
    """
    symbols: dict[str, list[str]] = {}
    package = _load_package(source_dir, package_name)
    if package is None:
        return symbols

    def walk_module(module, module_path: str):
        """Recursively walk through module and submodules."""
        # Skip internal modules (starting with _)
        if any(part.startswith("_") for part in module_path.split(".")):
            return

        # Get public classes and functions (not starting with _).
        # Constants/attributes are excluded — they are not expected to have
        # standalone documentation and would skew the coverage metric.
        # Aliases (re-exports from other modules) are also excluded — they are
        # documented at their canonical definition, not at each re-export site.
        for name, member in module.members.items():
            if not name.startswith("_"):
                try:
                    if getattr(member, "is_alias", False):
                        continue
                    if member.is_class or member.is_function:
                        full_path = f"{module_path}.{name}"
                        symbols[full_path] = []
                except Exception:
                    # Skip members that can't be resolved (e.g., aliases to stdlib)
                    pass

        # Recursively walk submodules (but skip internal ones)
        if hasattr(module, "modules"):
            for submodule_name, submodule in module.modules.items():
                if submodule_name.startswith("_"):
                    continue
                fp = getattr(submodule, "filepath", None)
                if not _is_public_submodule(submodule_name, fp):
                    continue
                submodule_path = f"{module_path}.{submodule_name}"
                walk_module(submodule, submodule_path)

    # Walk through all top-level modules
    for module_name, module in package.modules.items():
        if not module_name.startswith("_"):
            module_path = f"{package_name}.{module_name}"
            walk_module(module, module_path)

    return symbols


# ---------------------------------------------------------------------------
# Docstring quality audit
# ---------------------------------------------------------------------------

_ARGS_RE = re.compile(r"^\s*(Args|Arguments|Parameters)\s*:", re.MULTILINE)
_TYPEDDICT_BASES = re.compile(r"\bTypedDict\b")
_RETURNS_RE = re.compile(r"^\s*Returns\s*:", re.MULTILINE)
_YIELDS_RE = re.compile(r"^\s*Yields\s*:", re.MULTILINE)
_RAISES_RE = re.compile(r"^\s*Raises\s*:", re.MULTILINE)
_ATTRIBUTES_RE = re.compile(r"^\s*Attributes\s*:", re.MULTILINE)
# Matches an indented param entry inside an Args block: "    param_name:" or "    param_name (type):"
# The colon must be followed by whitespace to avoid matching Sphinx cross-reference
# continuation lines like "        through :func:`...`".
_ARGS_ENTRY_RE = re.compile(r"^\s{4,}(\w+)\s*(?:\([^)]*\))?\s*:\s", re.MULTILINE)
# Matches an indented param entry with an EXPLICIT type: "    param_name (SomeType):"
_ARGS_ENTRY_WITH_TYPE_RE = re.compile(r"^\s{4,}(\w+)\s*\(([^)]+)\)\s*:\s", re.MULTILINE)
# Matches the type prefix on the first content line of a Returns: or Yields: section.
# Format: "    SomeType: description" — only matches unambiguous type-like prefixes
# (word chars, brackets, pipes, commas, dots; no sentence-starting punctuation).
_SECTION_TYPE_LINE_RE = re.compile(
    r"^\s{4,}([\w][\w\[\] |,.*]*(?:\[[\w\[\] |,.*]+\])?):\s", re.MULTILINE
)
# Return annotations that need no Returns section
_TRIVIAL_RETURNS = {"None", "NoReturn", "Never", "never", ""}

# Typing-module aliases → modern builtin equivalents for normalization.
_TYPING_ALIAS_RE = re.compile(
    r"\b(List|Dict|Tuple|Set|FrozenSet|Type|Sequence|Mapping|Iterable|Iterator"
    r"|Generator|AsyncGenerator|AsyncIterator|Awaitable|Callable)\b"
)
_TYPING_ALIAS_MAP = {
    "List": "list",
    "Dict": "dict",
    "Tuple": "tuple",
    "Set": "set",
    "FrozenSet": "frozenset",
    "Type": "type",
}


def _normalize_type(t: str) -> str:
    """Normalize a type string for loose equality comparison.

    Handles the most common differences between docstring-stated types and
    Python annotations:
    - typing-module aliases: List → list, Dict → dict, etc.
    - Optional[X] → X | None
    - Union[A, B] → A | B
    - Union/pipe ordering: components are sorted so str|None == None|str
    - incidental whitespace

    Known simplifications (will NOT flag as mismatch):
    - Deeply nested Union/Optional combinations may not fully flatten
    - typing.X (qualified) vs X (unqualified) are NOT normalised
    - Callable[[A], B] argument ordering is not normalised
    """
    t = t.strip()
    # Strip typing. prefix
    t = re.sub(r"\btyping\.", "", t)
    # typing → builtin aliases
    t = _TYPING_ALIAS_RE.sub(lambda m: _TYPING_ALIAS_MAP.get(m.group(0), m.group(0)), t)
    # Optional[X] → X | None (non-nested only — avoids false positives on complex nesting)
    t = re.sub(r"\bOptional\[([^\[\]]*)\]", r"\1 | None", t)
    # Union[A, B, ...] → A | B | ... (non-nested)
    t = re.sub(
        r"\bUnion\[([^\[\]]+)\]",
        lambda m: " | ".join(x.strip() for x in m.group(1).split(",")),
        t,
    )
    # normalise whitespace around type operators
    t = re.sub(r"\s*\|\s*", " | ", t)
    t = re.sub(r"\s*,\s*", ", ", t)
    t = re.sub(r"\s*\[\s*", "[", t)
    t = re.sub(r"\s*\]\s*", "]", t)
    t = t.strip()
    # Sort pipe-union components so str|None == None|str.
    # Only apply at the top level (not inside brackets) to avoid reordering
    # tuple/generic args, which would cause false positives.
    if " | " in t and "[" not in t:
        t = " | ".join(sorted(t.split(" | ")))
    return t


def _types_match(a: str, b: str) -> bool:
    """Return True if two type strings are equivalent after normalisation.

    Returns True (no mismatch reported) when normalisation is uncertain:
    if either side still contains unexpanded Optional[...] or Union[...] after
    normalisation (indicating a nested generic we couldn't fully expand), we
    suppress the comparison to avoid false positives.
    """
    na, nb = _normalize_type(a), _normalize_type(b)
    # If either side has unexpanded Optional/Union (e.g. Optional[list[str]]),
    # skip — we can't reliably compare and must not emit a false positive.
    if re.search(r"\b(Optional|Union)\[", na) or re.search(r"\b(Optional|Union)\[", nb):
        return True
    return na == nb


def _extract_section_type(doc_text: str, section_re: re.Pattern[str]) -> str | None:
    """Extract the type prefix from the first content line of a docstring section.

    Returns the raw type string if found, or None if the section is absent or
    the first content line has no recognisable type prefix.
    """
    m = section_re.search(doc_text)
    if not m:
        return None
    # Grab text after the section heading up to the next blank line.
    after = doc_text[m.end() :]
    type_match = _SECTION_TYPE_LINE_RE.search(after.split("\n\n")[0])
    return type_match.group(1).strip() if type_match else None


# Return annotations that indicate a generator (should use Yields, not Returns)
_GENERATOR_RETURN_PATTERNS = re.compile(
    r"Generator|Iterator|AsyncGenerator|AsyncIterator"
)


def _check_member(member, full_path: str, short_threshold: int) -> list[dict]:
    """Return quality issues for a single class or function member.

    Each returned dict has keys: ``path``, ``kind``, ``detail``, ``file``, ``line``.
    ``file`` is the absolute source path; ``line`` is the 1-based line number of the
    symbol definition (the ``def`` / ``class`` keyword line).
    """
    issues: list[dict] = []
    _raw_file = getattr(member, "filepath", None)
    _abs_file = str(_raw_file) if _raw_file is not None else ""
    # Convert to a repo-relative path for readability and GHA annotations.
    try:
        _rel_file = str(Path(_abs_file).relative_to(Path.cwd())) if _abs_file else ""
    except ValueError:
        _rel_file = _abs_file
    _line = getattr(member, "lineno", None) or 0

    doc = getattr(member, "docstring", None)
    doc_text = doc.value.strip() if (doc and doc.value) else ""

    if not doc_text:
        issues.append({"path": full_path, "kind": "missing", "detail": "no docstring"})
        return issues  # no further checks without a docstring

    word_count = len(doc_text.split())
    if word_count < short_threshold:
        preview = doc_text[:70].replace("\n", " ")
        issues.append(
            {
                "path": full_path,
                "kind": "short",
                "detail": f'{word_count} word(s): "{preview}"',
            }
        )

    if getattr(member, "is_function", False):
        # Args section check: only flag when there are meaningful parameters.
        # Use Griffe ParameterKind to correctly exclude *args / **kwargs — their
        # names are stored without the leading '*', so a startswith("*") check
        # would not filter them.
        params = getattr(member, "parameters", None)
        _variadic_kinds = {
            griffe.ParameterKind.var_positional,
            griffe.ParameterKind.var_keyword,
        }
        concrete = [
            p.name
            for p in (params or [])
            if p.name not in ("self", "cls")
            and getattr(p, "kind", None) not in _variadic_kinds
        ]
        # A function whose only non-self params are *args/**kwargs is a variadic
        # forwarder (e.g. def f(*args, **kwargs)). Its docstring Args: section
        # documents accepted kwargs by convention, not a concrete signature —
        # skip both no_args and param_mismatch for these.
        is_variadic_forwarder = (not concrete) and any(
            getattr(p, "kind", None) in _variadic_kinds for p in (params or [])
        )
        if concrete and not _ARGS_RE.search(doc_text):
            sample = ", ".join(concrete[:3]) + ("..." if len(concrete) > 3 else "")
            issues.append(
                {
                    "path": full_path,
                    "kind": "no_args",
                    "detail": f"params [{sample}] have no Args section",
                }
            )
        elif concrete and not is_variadic_forwarder and _ARGS_RE.search(doc_text):
            # Param name mismatch: documented names that don't exist in the signature
            args_block = re.search(
                r"(?:Args|Arguments|Parameters)\s*:(.*?)(?:\n\s*\n|\Z)",
                doc_text,
                re.DOTALL,
            )
            if args_block:
                doc_param_names = set(_ARGS_ENTRY_RE.findall(args_block.group(1)))
                actual_names = set(concrete)
                phantom = doc_param_names - actual_names
                if phantom:
                    issues.append(
                        {
                            "path": full_path,
                            "kind": "param_mismatch",
                            "detail": f"documented params {sorted(phantom)} not in signature",
                        }
                    )

            # Missing type annotation: Args: section exists but parameters lack Python
            # annotations — the type column will be absent from the generated API docs.
            # Only checked here (when Args: exists) to avoid duplicating no_args reports.
            # Variadic *args/**kwargs are excluded via ParameterKind (same filter as concrete).
            unannotated = [
                p.name
                for p in (params or [])
                if p.name not in ("self", "cls")
                and getattr(p, "kind", None) not in _variadic_kinds
                and p.annotation is None
            ]
            if unannotated:
                sample = ", ".join(unannotated[:3]) + (
                    "..." if len(unannotated) > 3 else ""
                )
                issues.append(
                    {
                        "path": full_path,
                        "kind": "missing_param_type",
                        "detail": f"params [{sample}] have no Python type annotation — type absent from API docs",
                    }
                )

            # Param type consistency: docstring states a type but it differs from
            # the signature annotation.  Only fires when BOTH sides have an explicit
            # type — absence is already handled by missing_param_type / no_args.
            # Uses _types_match() which normalises aliases, Optional/Union, and pipe
            # ordering.  Comparisons involving nested generics (e.g. Optional[list[X]])
            # that cannot be fully expanded are silently skipped to avoid false positives.
            if args_block:
                for p in params or []:
                    if p.name in ("self", "cls"):
                        continue
                    if getattr(p, "kind", None) in _variadic_kinds:
                        continue
                    if p.annotation is None:
                        continue  # missing_param_type already handles this
                    # Find the docstring type for this param
                    param_type_match = re.search(
                        rf"^\s{{4,}}{re.escape(p.name)}\s*\(([^)]+)\)\s*:\s",
                        args_block.group(1),
                        re.MULTILINE,
                    )
                    if param_type_match:
                        if not _types_match(
                            param_type_match.group(1), str(p.annotation)
                        ):
                            issues.append(
                                {
                                    "path": full_path,
                                    "kind": "param_type_mismatch",
                                    "detail": (
                                        f"param '{p.name}': docstring says"
                                        f" '{param_type_match.group(1).strip()}'"
                                        f" but annotation is '{p.annotation}'"
                                    ),
                                }
                            )

        # Returns/Yields section check: only flag when there is an explicit non-trivial annotation.
        # Generator return types (Generator, Iterator, etc.) require Yields:, not Returns:.
        returns = getattr(member, "returns", None)
        ret_str = str(returns).strip() if returns else ""
        if ret_str and ret_str not in _TRIVIAL_RETURNS:
            is_generator = bool(_GENERATOR_RETURN_PATTERNS.search(ret_str))
            if is_generator and not _YIELDS_RE.search(doc_text):
                issues.append(
                    {
                        "path": full_path,
                        "kind": "no_yields",
                        "detail": f"return type {ret_str!r} is a generator — needs Yields section, not Returns",
                    }
                )
            elif not is_generator and not _RETURNS_RE.search(doc_text):
                issues.append(
                    {
                        "path": full_path,
                        "kind": "no_returns",
                        "detail": f"return type {ret_str!r} has no Returns section",
                    }
                )
            elif not is_generator:
                # Return type consistency: docstring Returns: states a type prefix but
                # it differs from the signature annotation.
                # Only fires when BOTH sides have an explicit type — missing_return_type
                # and no_returns already handle one-sided absence.
                # See _normalize_type() docstring for known simplifications that may
                # suppress edge cases (e.g. deeply nested Union, qualified typing.X).
                doc_ret_type = _extract_section_type(doc_text, _RETURNS_RE)
                if doc_ret_type:
                    if not _types_match(doc_ret_type, ret_str):
                        issues.append(
                            {
                                "path": full_path,
                                "kind": "return_type_mismatch",
                                "detail": (
                                    f"Returns: says '{doc_ret_type}'"
                                    f" but annotation is '{ret_str}'"
                                ),
                            }
                        )

        # Missing return annotation: Returns: section documented but no Python annotation.
        # Naturally non-overlapping with no_returns (which fires when annotation exists
        # but Returns: section is absent). When both are missing, no_returns fires first.
        if not ret_str and _RETURNS_RE.search(doc_text):
            issues.append(
                {
                    "path": full_path,
                    "kind": "missing_return_type",
                    "detail": "Returns: section documented but no return type annotation — type absent from API docs",
                }
            )

        # Raises section check: only flag when the source contains explicit raise statements
        source = getattr(member, "source", None) or ""
        if "raise " in source and not _RAISES_RE.search(doc_text):
            issues.append(
                {
                    "path": full_path,
                    "kind": "no_raises",
                    "detail": "function raises but has no Raises section",
                }
            )

    if getattr(member, "is_class", False):
        # Args section check for classes: look at __init__ typed parameters.
        # Variadic *args/**kwargs are excluded by kind (same logic as function check).
        _variadic_kinds = {
            griffe.ParameterKind.var_positional,
            griffe.ParameterKind.var_keyword,
        }
        init = member.members.get("__init__")
        if init:
            init_params = getattr(init, "parameters", None)
            typed_params = [
                p.name
                for p in (init_params or [])
                if p.name not in ("self", "cls")
                and getattr(p, "kind", None) not in _variadic_kinds
                and p.annotation is not None
            ]
            if typed_params and not _ARGS_RE.search(doc_text):
                sample = ", ".join(typed_params[:3]) + (
                    "..." if len(typed_params) > 3 else ""
                )
                issues.append(
                    {
                        "path": full_path,
                        "kind": "no_class_args",
                        "detail": f"__init__ params [{sample}] have no Args section",
                    }
                )

            # Duplicate Args check: Option C requires Args: on the class docstring only.
            # Flag when __init__ has its own Args: section in addition to the class's.
            init_doc = getattr(init, "docstring", None)
            init_doc_text = (
                init_doc.value.strip() if (init_doc and init_doc.value) else ""
            )
            if _ARGS_RE.search(init_doc_text) and _ARGS_RE.search(doc_text):
                issues.append(
                    {
                        "path": full_path,
                        "kind": "duplicate_init_args",
                        "detail": (
                            "Args: in both class and __init__ docstrings "
                            "(Option C: place Args: on class docstring only)"
                        ),
                    }
                )

        # TypedDict field mismatch check.
        # Unlike regular classes (where Attributes: is optional under Option C),
        # TypedDict fields *are* the entire public contract. When an Attributes:
        # section exists, every entry must match an actual declared field and every
        # declared field must appear — stale or missing entries are always a bug.
        is_typeddict = any(
            _TYPEDDICT_BASES.search(str(base)) for base in getattr(member, "bases", [])
        )
        if is_typeddict and _ATTRIBUTES_RE.search(doc_text):
            attrs_block = re.search(
                r"Attributes\s*:(.*?)(?:\n\s*\n|\Z)", doc_text, re.DOTALL
            )
            if attrs_block:
                doc_field_names = set(_ARGS_ENTRY_RE.findall(attrs_block.group(1)))
                actual_fields = {
                    name
                    for name, m in member.members.items()
                    if not name.startswith("_") and getattr(m, "is_attribute", False)
                }
                phantom = doc_field_names - actual_fields
                if phantom:
                    issues.append(
                        {
                            "path": full_path,
                            "kind": "typeddict_phantom",
                            "detail": f"Attributes: documents {sorted(phantom)} not declared in TypedDict",
                        }
                    )
                undocumented = actual_fields - doc_field_names
                if undocumented:
                    issues.append(
                        {
                            "path": full_path,
                            "kind": "typeddict_undocumented",
                            "detail": f"TypedDict fields {sorted(undocumented)} missing from Attributes: section",
                        }
                    )

    # Inject file location into every issue so callers can display and annotate.
    for issue in issues:
        issue["file"] = _rel_file
        issue["line"] = _line

    return issues


def audit_docstring_quality(
    source_dir: Path,
    package_name: str,
    short_threshold: int = 5,
    include_methods: bool = True,
    documented: set[str] | None = None,
) -> list[dict]:
    """Audit docstring quality for all public classes and functions.

    Checks each public symbol for:
    - missing: no docstring at all
    - short: docstring below short_threshold words
    - no_args: function with parameters but no Args/Parameters section
    - no_returns: function with a non-trivial return annotation but no Returns section
    - no_yields: generator function (Generator/Iterator return type) but no Yields section
    - no_raises: function whose source contains raise but has no Raises section
    - no_class_args: class whose __init__ has typed params but no Args section on the class
    - duplicate_init_args: Args: present in both class docstring and __init__ (Option C violation)
    - param_mismatch: Args section documents names absent from the real signature
    - typeddict_phantom: TypedDict Attributes: section documents fields not declared in the class
    - typeddict_undocumented: TypedDict has declared fields absent from its Attributes: section
    - missing_param_type: Args: section exists but one or more parameters lack Python type
      annotations — the type column will be absent from the generated API docs. Only checked
      when no no_args issue exists (avoids duplicate reports).
    - missing_return_type: Returns: section exists but the function has no return type
      annotation — the return type will be absent from the generated API docs. Only checked
      when no no_returns issue exists (avoids duplicate reports).

    Note: Attributes: sections are intentionally not enforced for regular classes. Under
    the Option C convention, Attributes: is only used when stored values differ in type or
    behaviour from the constructor inputs (e.g. type transforms, computed values, class
    constants). Pure-echo entries that repeat Args: verbatim are omitted. TypedDicts are
    a carve-out: their fields are the entire public contract, so when an Attributes:
    section is present it must exactly match the declared fields.

    Only symbols (and methods whose parent class) present in `documented` are
    checked when that set is provided — ensuring the audit is scoped to what is
    actually surfaced in the API reference.

    Args:
        source_dir: Root directory to scan (e.g., mellea/ or cli/)
        package_name: Package name (e.g., "mellea" or "cli")
        short_threshold: Word count below which a docstring is flagged as short
        include_methods: Whether to audit public methods on classes in addition
            to top-level functions and classes
        documented: Set of symbol paths present in the generated MDX docs (from
            find_documented_symbols()). When provided, only documented symbols
            are audited. Pass None to audit all public symbols.

    Returns:
        List of issue dicts, each with keys: path, kind, detail
    """
    issues: list[dict] = []
    package = _load_package(source_dir, package_name)
    if package is None:
        return issues

    def walk_module(module, module_path: str) -> None:
        if any(part.startswith("_") for part in module_path.split(".")):
            return

        for name, member in module.members.items():
            if name.startswith("_"):
                continue
            try:
                if getattr(member, "is_alias", False):
                    continue
                if not (member.is_class or member.is_function):
                    continue
            except Exception:
                continue

            full_path = f"{module_path}.{name}"

            # Skip symbols not in the API reference when a filter is provided
            if documented is not None and full_path not in documented:
                continue

            issues.extend(_check_member(member, full_path, short_threshold))

            if include_methods and getattr(member, "is_class", False):
                for mname, method in member.members.items():
                    if mname.startswith("_"):
                        continue
                    try:
                        if getattr(method, "is_alias", False):
                            continue
                        if not getattr(method, "is_function", False):
                            continue
                    except Exception:
                        continue
                    issues.extend(
                        _check_member(method, f"{full_path}.{mname}", short_threshold)
                    )

        if hasattr(module, "modules"):
            for submodule_name, submodule in module.modules.items():
                if not submodule_name.startswith("_"):
                    walk_module(submodule, f"{module_path}.{submodule_name}")

    for module_name, module in package.modules.items():
        if not module_name.startswith("_"):
            walk_module(module, f"{package_name}.{module_name}")

    return issues


_IN_GHA = os.environ.get("GITHUB_ACTIONS") == "true"

# Base URLs for documentation references emitted in fix hints.
# These point to upstream main; anchors for the CI checks reference section
# will resolve once the PR introducing them is merged.
_CONTRIB_DOCS_URL = (
    "https://github.com/generative-computing/mellea/blob/main"
    "/docs/docs/guide/CONTRIBUTING.md"
)
_COVERAGE_DOCS_URL = (
    "https://github.com/generative-computing/mellea/blob/main"
    "/CONTRIBUTING.md#validating-docstrings"
)

# Per-kind fix hints: (one-line fix text, CONTRIBUTING.md anchor)
_KIND_FIX_HINTS: dict[str, tuple[str, str]] = {
    "missing": (
        "Add a Google-style summary sentence.",
        f"{_CONTRIB_DOCS_URL}#missing-or-short-docstrings",
    ),
    "short": (
        "Expand the summary — aim for at least 5 meaningful words.",
        f"{_CONTRIB_DOCS_URL}#missing-or-short-docstrings",
    ),
    "no_args": (
        "Add an Args: section listing each parameter with a description.",
        f"{_CONTRIB_DOCS_URL}#args-returns-yields-and-raises",
    ),
    "no_returns": (
        "Add a Returns: section describing the return value.",
        f"{_CONTRIB_DOCS_URL}#args-returns-yields-and-raises",
    ),
    "no_yields": (
        "Add a Yields: section (generator functions use Yields:, not Returns:).",
        f"{_CONTRIB_DOCS_URL}#args-returns-yields-and-raises",
    ),
    "no_raises": (
        "Add a Raises: section listing each exception and the condition that triggers it.",
        f"{_CONTRIB_DOCS_URL}#args-returns-yields-and-raises",
    ),
    "no_class_args": (
        "Add Args: to the class docstring (not __init__) — Option C convention.",
        f"{_CONTRIB_DOCS_URL}#class-docstrings-option-c",
    ),
    "duplicate_init_args": (
        "Remove Args: from __init__ docstring; place it on the class docstring only (Option C).",
        f"{_CONTRIB_DOCS_URL}#class-docstrings-option-c",
    ),
    "param_mismatch": (
        "Remove or rename phantom entries to match the actual parameter names.",
        f"{_CONTRIB_DOCS_URL}#class-docstrings-option-c",
    ),
    "typeddict_phantom": (
        "Remove documented fields that are not declared in the TypedDict.",
        f"{_CONTRIB_DOCS_URL}#typeddict-classes",
    ),
    "typeddict_undocumented": (
        "Add the missing fields to the Attributes: section.",
        f"{_CONTRIB_DOCS_URL}#typeddict-classes",
    ),
    "missing_param_type": (
        "Add a Python type annotation to each listed parameter in the function signature.",
        f"{_CONTRIB_DOCS_URL}#args-returns-yields-and-raises",
    ),
    "missing_return_type": (
        "Add a return type annotation (-> SomeType) to the function signature.",
        f"{_CONTRIB_DOCS_URL}#args-returns-yields-and-raises",
    ),
    "param_type_mismatch": (
        "Align the type in the Args: entry with the signature annotation (or vice versa).",
        f"{_CONTRIB_DOCS_URL}#args-returns-yields-and-raises",
    ),
    "return_type_mismatch": (
        "Align the type prefix in Returns: with the signature annotation (or vice versa).",
        f"{_CONTRIB_DOCS_URL}#args-returns-yields-and-raises",
    ),
}


def _gha_cmd(level: str, title: str, message: str) -> None:
    """Emit a GitHub Actions workflow command annotation."""
    # Escape special characters required by the GHA annotation format
    message = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    title = title.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"::{level} title={title}::{message}")


def _gha_file_annotation(
    level: str, title: str, message: str, file: str, line: int
) -> None:
    """Emit a GitHub Actions annotation anchored to a specific file and line.

    When ``file`` and ``line`` are provided the annotation appears inline in the
    PR diff view, making it immediately obvious which symbol needs fixing.

    Args:
        level: Annotation level — ``"error"``, ``"warning"``, or ``"notice"``.
        title: Short label shown in bold in the annotation.
        message: Body text for the annotation.
        file: Repo-relative file path (e.g. ``mellea/core/base.py``).
        line: 1-based line number of the symbol definition.
    """
    for s in (message, title, file):
        s = s.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    if file and line:
        print(f"::{level} file={file},line={line},title={title}::{message}")
    else:
        print(f"::{level} title={title}::{message}")


# GitHub Actions silently drops inline diff annotations beyond ~10 per step.
# This per-kind cap ensures every category gets at least one visible annotation
# rather than early kinds consuming the entire budget.
_GHA_ANNOTATIONS_PER_KIND = 10


def _print_quality_report(issues: list[dict], *, fail_on_quality: bool = False) -> None:
    """Print a grouped quality report to stdout and emit GHA annotations.

    When running in GitHub Actions (``GITHUB_ACTIONS=true``), each issue is
    also emitted as a file-level annotation (``::error`` or ``::warning``)
    anchored to the source line, so they appear inline in the PR diff.

    GitHub Actions caps inline diff annotations at roughly 10 per step; issues
    beyond that cap are silently dropped from the diff view (they still appear
    in the full job log and in the JSON artifact).  To ensure every check
    category gets at least one visible annotation, this function emits at most
    :data:`_GHA_ANNOTATIONS_PER_KIND` annotations per kind and prints a
    ``"... and N more"`` notice for the remainder.  The complete list is always
    written to the job log regardless of the GHA cap.

    Args:
        issues: List of issue dicts from :func:`audit_docstring_quality`.
            Each dict must have keys: ``path``, ``kind``, ``detail``,
            ``file``, ``line``.
        fail_on_quality: When ``True`` annotations are emitted as ``error``
            (red); otherwise as ``warning`` (yellow).
    """
    by_kind: dict[str, list[dict]] = {}
    for issue in issues:
        by_kind.setdefault(issue["kind"], []).append(issue)

    kind_labels = {
        "missing": "Missing docstrings",
        "short": "Short docstrings",
        "no_args": "Missing Args section",
        "no_returns": "Missing Returns section",
        "no_yields": "Missing Yields section (generator)",
        "no_raises": "Missing Raises section",
        "no_class_args": "Missing class Args section",
        "duplicate_init_args": "Duplicate Args: in class + __init__ (Option C violation)",
        "param_mismatch": "Param name mismatches (documented but not in signature)",
        "typeddict_phantom": "TypedDict phantom fields (documented but not declared)",
        "typeddict_undocumented": "TypedDict undocumented fields (declared but missing from Attributes:)",
        "missing_param_type": "Missing parameter type annotations (type absent from API docs)",
        "missing_return_type": "Missing return type annotations (type absent from API docs)",
        "param_type_mismatch": "Param type mismatch (docstring vs annotation)",
        "return_type_mismatch": "Return type mismatch (docstring vs annotation)",
    }

    annotation_level = "error" if fail_on_quality else "warning"

    total = len(issues)
    print(f"\n{'=' * 60}")
    print("Docstring Quality Report")
    print(f"{'=' * 60}")
    print(f"Total issues found: {total}")

    for kind in (
        "missing",
        "short",
        "no_args",
        "no_returns",
        "no_yields",
        "no_raises",
        "no_class_args",
        "duplicate_init_args",
        "param_mismatch",
        "typeddict_phantom",
        "typeddict_undocumented",
        "missing_param_type",
        "missing_return_type",
        "param_type_mismatch",
        "return_type_mismatch",
    ):
        items = by_kind.get(kind, [])
        if not items:
            continue
        label = kind_labels.get(kind, kind)
        fix_hint, ref_url = _KIND_FIX_HINTS.get(kind, ("", ""))
        print(f"\n{'─' * 50}")
        print(f"  {label} ({len(items)})")
        if fix_hint:
            print(f"  Fix: {fix_hint}")
        if ref_url:
            print(f"  Ref: {ref_url}")
        print(f"{'─' * 50}")
        sorted_items = sorted(items, key=lambda x: x["path"])
        gha_emitted = 0
        for item in sorted_items:
            file_ref = item.get("file", "")
            line_num = item.get("line", 0)
            loc = f"  [{file_ref}:{line_num}]" if file_ref and line_num else ""
            print(f"  {item['path']}{loc}")
            print(f"    {item['detail']}")
            if _IN_GHA and file_ref and gha_emitted < _GHA_ANNOTATIONS_PER_KIND:
                _gha_file_annotation(
                    annotation_level,
                    f"[{kind}] {item['path']}",
                    item["detail"],
                    file_ref,
                    line_num,
                )
                gha_emitted += 1
        if _IN_GHA and len(sorted_items) > _GHA_ANNOTATIONS_PER_KIND:
            remainder = len(sorted_items) - _GHA_ANNOTATIONS_PER_KIND
            print(
                f"  (GHA annotations capped at {_GHA_ANNOTATIONS_PER_KIND} per kind — "
                f"{remainder} more {kind!r} issue(s) in job log and JSON artifact)"
            )


def audit_nav_orphans(docs_dir: Path, source_dir: Path) -> list[str]:
    """Find MDX files that exist on disk but are not linked in mint.json navigation.

    An orphaned module has a generated MDX file but no entry in the Mintlify
    navigation tree, so it is unreachable from the docs site.

    Args:
        docs_dir: Directory containing generated MDX files (e.g. docs/docs/api)
        source_dir: Project root, used to locate docs/mint.json

    Returns:
        Sorted list of orphaned module paths relative to docs_dir (no extension)
    """
    # Support both Mintlify v1 (mint.json at docs/mint.json) and
    # v2 (docs.json at docs/docs/docs.json).  docs.json uses plain string
    # entries in "pages" arrays; mint.json uses {"page": "..."} dicts.
    nav_config: Path | None = None
    for candidate in (
        source_dir / "docs" / "docs" / "docs.json",
        source_dir / "docs" / "mint.json",
    ):
        if candidate.exists():
            nav_config = candidate
            break

    mdx_files: set[str] = set()
    for mdx_file in docs_dir.rglob("*.mdx"):
        mdx_files.add(str(mdx_file.relative_to(docs_dir).with_suffix("")))

    nav_refs: set[str] = set()
    if nav_config is not None:
        config = json.loads(nav_config.read_text())

        def _extract(obj: object) -> None:
            if isinstance(obj, str):
                # docs.json / mint.json plain string page entry
                if obj.startswith("api/"):
                    nav_refs.add(obj[len("api/") :])
            elif isinstance(obj, dict):
                # mint.json {"page": "api/..."} dict entry
                if "page" in obj:
                    page = obj["page"]
                    if isinstance(page, str) and page.startswith("api/"):
                        nav_refs.add(page[len("api/") :])
                for v in obj.values():
                    _extract(v)
            elif isinstance(obj, list):
                for item in obj:
                    _extract(item)

        _extract(config)

    return sorted(mdx_files - nav_refs)


def discover_cli_commands(cli_dir: Path) -> list[str]:
    """Discover CLI commands from Typer applications.

    Args:
        cli_dir: Path to cli/ directory

    Returns:
        List of command names (e.g., ["m serve", "m alora", "m decompose"])
    """
    commands = []

    # Look for Typer app definitions
    # This is a simplified version - full implementation would parse the CLI structure
    main_file = cli_dir / "m.py"
    if main_file.exists():
        content = main_file.read_text()

        # Simple heuristic: look for @app.command() decorators or add_typer() calls

        # Find command decorators
        command_pattern = r'@app\.command\(["\']([^"\']+)["\']\)'
        commands.extend(re.findall(command_pattern, content))

        # Find subcommand additions
        typer_pattern = r'app\.add_typer\([^,]+,\s*name=["\']([^"\']+)["\']\)'
        commands.extend(re.findall(typer_pattern, content))

    return sorted(set(commands))


def find_documented_symbols(docs_dir: Path) -> set[str]:
    """Find which symbols have MDX documentation.

    Args:
        docs_dir: Path to docs/docs/api/ directory

    Returns:
        Set of documented symbol paths (e.g., {"mellea.core.base.Base"})
    """
    documented: set[str] = set()

    if not docs_dir.exists():
        return documented

    # Walk through all .mdx files
    for mdx_file in docs_dir.rglob("*.mdx"):
        # Convert file path to module path
        # e.g., mellea/core/base.mdx -> mellea.core.base
        rel_path = mdx_file.relative_to(docs_dir)
        module_path = str(rel_path.with_suffix("")).replace("/", ".")

        # Read file to find documented symbols
        content = mdx_file.read_text()

        # Look for heading patterns that indicate symbol documentation.
        # mdxify output format (0.2.37+): ### `SymbolName` <sup>...</sup>
        # Legacy formats kept for compatibility with older generated files.
        patterns = [
            r"^##\s+(?:class|function|attribute)\s+(\w+)",  # very old
            r"###\s+<span[^>]*>(?:FUNC|CLASS|ATTR)</span>\s+`(\w+)`",  # intermediate
            r"^###\s+`(\w+)`",  # current mdxify 0.2.37+
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                symbol_name = match.group(1)
                documented.add(f"{module_path}.{symbol_name}")

    return documented


def generate_coverage_report(
    discovered: dict[str, list[str]], documented: set[str], cli_commands: list[str]
) -> dict:
    """Generate coverage report.

    Args:
        discovered: Dict of full symbol paths (keys are "module.symbol", values are empty lists)
        documented: Set of documented symbols (full paths like "module.symbol")
        cli_commands: List of CLI commands

    Returns:
        Coverage report dict with statistics and missing symbols
    """
    # discovered is now a dict where keys are full paths like "mellea.core.base.Component"
    total_symbols = len(discovered)

    # Count how many discovered symbols are documented
    documented_count = len(discovered.keys() & documented)

    # Find missing symbols grouped by module
    missing: dict[str, list[str]] = {}
    for full_path in discovered.keys():
        if full_path not in documented:
            # Extract module and symbol name
            parts = full_path.rsplit(".", 1)
            if len(parts) == 2:
                module_path, symbol_name = parts
                if module_path not in missing:
                    missing[module_path] = []
                missing[module_path].append(symbol_name)

    coverage_pct = (documented_count / total_symbols * 100) if total_symbols > 0 else 0

    return {
        "total_symbols": total_symbols,
        "documented_symbols": documented_count,
        "coverage_percentage": round(coverage_pct, 2),
        "missing_symbols": missing,
        "cli_commands": cli_commands,
        "cli_documented": [],  # TODO: check CLI documentation
    }


def main():
    parser = argparse.ArgumentParser(description="Audit API documentation coverage")
    parser.add_argument("--source-dir", default=".", help="Project root directory")
    parser.add_argument(
        "--docs-dir",
        default=None,
        help="Generated docs directory (default: <source-dir>/docs/docs/api)",
    )
    parser.add_argument("--output", help="Output JSON file for report")
    parser.add_argument(
        "--threshold", type=float, default=80.0, help="Minimum coverage threshold"
    )
    parser.add_argument(
        "--quality",
        action="store_true",
        help="Run docstring quality audit (missing, short, no Args/Returns sections)",
    )
    parser.add_argument(
        "--short-threshold",
        type=int,
        default=5,
        metavar="N",
        help="Flag docstrings with fewer than N words as short (default: 5)",
    )
    parser.add_argument(
        "--no-methods",
        action="store_true",
        help="Exclude class methods from quality audit (check top-level symbols only)",
    )
    parser.add_argument(
        "--orphans",
        action="store_true",
        help="Check for MDX files not linked in docs/mint.json navigation",
    )
    parser.add_argument(
        "--fail-on-quality",
        action="store_true",
        help="Exit 1 if any quality issues are found (for CI/pre-commit use)",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    docs_dir = Path(args.docs_dir) if args.docs_dir else source_dir / "docs/docs/api"

    print("🔍 Discovering public symbols...")
    mellea_symbols = discover_public_symbols(source_dir / "mellea", "mellea")
    cli_symbols = discover_public_symbols(source_dir / "cli", "cli")

    print("🔍 Discovering CLI commands...")
    cli_commands = discover_cli_commands(source_dir / "cli")

    print("📚 Finding documented symbols...")
    documented = find_documented_symbols(docs_dir)

    print("📊 Generating coverage report...")
    all_symbols = {**mellea_symbols, **cli_symbols}
    report = generate_coverage_report(all_symbols, documented, cli_commands)

    # Print coverage report
    print(f"\n{'=' * 60}")
    print("API Documentation Coverage Report")
    print(f"{'=' * 60}")
    print(f"Total classes + functions: {report['total_symbols']}")
    print(f"Documented: {report['documented_symbols']}")
    print(f"Coverage: {report['coverage_percentage']}%")
    print(f"CLI commands: {len(report['cli_commands'])}")

    if report["missing_symbols"]:
        total_missing = sum(len(s) for s in report["missing_symbols"].values())
        print(f"\n{'─' * 60}")
        print(
            f"  Missing API docs — {total_missing} symbol(s) across "
            f"{len(report['missing_symbols'])} module(s)"
        )
        print(
            "  Fix: Run the doc generation pipeline to produce MDX for new symbols,\n"
            "       then add entries to docs/docs/docs.json navigation.\n"
            "       uv run python tooling/docs-autogen/generate-ast.py"
        )
        print(f"  Ref: {_COVERAGE_DOCS_URL}")
        print(f"{'─' * 60}")
        for module, symbols in sorted(report["missing_symbols"].items()):
            print(f"  {module}")
            for sym in sorted(symbols):
                print(f"    {sym}")
        if _IN_GHA:
            _gha_cmd(
                "error"
                if report["coverage_percentage"] < args.threshold
                else "warning",
                "API Coverage",
                f"{total_missing} symbol(s) undocumented in "
                f"{len(report['missing_symbols'])} module(s) — "
                f"coverage {report['coverage_percentage']}% "
                f"(threshold {args.threshold}%)",
            )

    # Quality audit — scoped to documented (API reference) symbols only
    quality_issues: list[dict] = []
    if args.quality:
        print("\n🔬 Running docstring quality audit (documented symbols only)...")
        include_methods = not args.no_methods
        for pkg, pkg_name in [("mellea", "mellea"), ("cli", "cli")]:
            pkg_dir = source_dir / pkg
            if pkg_dir.exists():
                quality_issues.extend(
                    audit_docstring_quality(
                        pkg_dir,
                        pkg_name,
                        short_threshold=args.short_threshold,
                        include_methods=include_methods,
                        documented=documented,
                    )
                )
        _print_quality_report(quality_issues, fail_on_quality=args.fail_on_quality)

    # Nav orphan check — MDX files not referenced in mint.json navigation
    orphans: list[str] = []
    if args.orphans:
        print("\n🔗 Checking navigation orphans...")
        orphans = audit_nav_orphans(docs_dir, source_dir)
        print(f"\n{'=' * 60}")
        print("Navigation Orphans Report")
        print(f"{'=' * 60}")
        if orphans:
            print(f"⚠️  {len(orphans)} MDX file(s) not linked in navigation:")
            for orphan in orphans:
                print(f"  • {orphan}")
        else:
            print("✅ All MDX files are linked in navigation.")

    # Save report if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        full_report = {**report}
        if args.quality:
            full_report["quality_issues"] = quality_issues
        if args.orphans:
            full_report["nav_orphans"] = orphans
        output_path.write_text(json.dumps(full_report, indent=2))
        print(f"\n✅ Report saved to {output_path}")

    # Check threshold
    failed = False
    if report["coverage_percentage"] < args.threshold:
        print(
            f"\n❌ Coverage {report['coverage_percentage']}% below threshold {args.threshold}%"
        )
        failed = True
    else:
        print(f"\n✅ Coverage meets threshold {args.threshold}%")

    if args.fail_on_quality and quality_issues:
        print(
            f"\n❌ {len(quality_issues)} quality issue(s) found (--fail-on-quality set)"
        )
        failed = True

    if _IN_GHA:
        if not args.quality:
            pass  # quality step is a separate CI job; no annotation needed here
        elif quality_issues:
            _gha_cmd(
                "error" if args.fail_on_quality else "warning",
                "Docstring quality",
                f"{len(quality_issues)} issue(s) found — see job summary for details",
            )
        else:
            _gha_cmd(
                "notice",
                "Docstring quality",
                "All documented symbols pass quality checks",
            )

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
