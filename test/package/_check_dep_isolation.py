"""Subprocess helper for dependency isolation tests.

Usage: python _check_dep_isolation.py <group_name> <module1> [module2 ...]

Exits 0 if all imports are from declared dependencies, 1 if violations found.
"""

import importlib
import importlib.metadata
import re
import sys
import tomllib
from pathlib import Path

# Packages that are part of the Python standard library or otherwise
# should never be flagged as undeclared dependencies.
STDLIB_AND_INFRASTRUCTURE = {
    # Build/install infrastructure that leaks into sys.modules
    "_distutils_hack",
    "pkg_resources",
    "setuptools",
    "pip",
    "wheel",
    "distutils",
}

# Packages that third-party libraries opportunistically import via
# `try/except ImportError` when installed.  These are extras of core
# networking and serialization libraries — not declared by mellea, but
# they appear in sys.modules when present in the environment.
OPPORTUNISTIC_IMPORTS = {
    # urllib3 / httpx extras (compression & protocol upgrades)
    "brotli",
    "brotlicffi",
    "zstandard",
    "h2",
    "hpack",
    "hyperframe",
    "socksio",
    # Widely used utility imported opportunistically by many packages
    "packaging",
    # Fast JSON — used by pydantic/fastapi when available
    "orjson",
}


def parse_dep_name(dep_spec: str) -> str | None:
    """Extract the distribution name from a dependency specifier.

    Strips version constraints, extras, and environment markers.
    Returns None for self-references like 'mellea[hooks]'.
    """
    # Remove environment markers (e.g., "; sys_platform != 'darwin'")
    dep_spec = dep_spec.split(";")[0].strip()
    # Extract just the package name (before any version/extras specifiers)
    match = re.match(r"^([A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?)", dep_spec)
    if not match:
        return None
    name = match.group(1).lower()
    # Skip self-references (handled separately by extract_self_ref_groups)
    if name == "mellea":
        return None
    return name


def extract_self_ref_groups(dep_spec: str) -> list[str]:
    """Extract optional-dependency group names from self-references.

    e.g. 'mellea[hooks]' → ['hooks'], 'mellea[watsonx,hf,vllm]' → ['watsonx', 'hf', 'vllm']
    Returns an empty list for non-self-references.
    """
    dep_spec = dep_spec.split(";")[0].strip()
    match = re.match(r"^mellea\[([^\]]+)\]", dep_spec, re.IGNORECASE)
    if not match:
        return []
    return [g.strip() for g in match.group(1).split(",")]


def get_top_level_names(dist_name: str) -> set[str]:
    """Get the importable top-level module names for a distribution."""
    try:
        dist = importlib.metadata.distribution(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return set()

    # Try top_level.txt first
    top_level = dist.read_text("top_level.txt")
    if top_level:
        return {line.strip() for line in top_level.splitlines() if line.strip()}

    # Fall back to packages listed in RECORD
    names = set()
    if dist.files:
        for f in dist.files:
            parts = str(f).split("/")
            if len(parts) > 1 and not parts[0].endswith(".dist-info"):
                name = parts[0].replace(".py", "")
                if name and not name.startswith("_") and name != "__pycache__":
                    names.add(name)
    if names:
        return names

    # Last resort: normalize the dist name itself
    return {dist_name.replace("-", "_").lower()}


def get_transitive_deps(dist_name: str, seen: set[str] | None = None) -> set[str]:
    """Recursively resolve all transitive dependencies of a distribution.

    Returns a set of normalized distribution names.
    """
    if seen is None:
        seen = set()

    normalized = dist_name.lower().replace("-", "_")
    if normalized in seen:
        return set()
    seen.add(normalized)

    result = {normalized}
    try:
        dist = importlib.metadata.distribution(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return result

    reqs = dist.requires
    if not reqs:
        return result

    for req in reqs:
        # For extras-only requirements (e.g., 'brotli ; extra == "brotli"'),
        # include them if actually installed.  These are legitimate transitive
        # deps of declared packages — e.g., urllib3[brotli] pulls in brotli,
        # datasets[s3] pulls in boto3, transformers[torch] pulls in torchvision.
        if "extra ==" in req:
            dep = parse_dep_name(req)
            if dep:
                try:
                    importlib.metadata.distribution(dep)
                    result |= get_transitive_deps(dep, seen)
                except importlib.metadata.PackageNotFoundError:
                    pass
            continue
        dep = parse_dep_name(req)
        if dep:
            result |= get_transitive_deps(dep, seen)

    return result


def build_allowed_set(
    group_name: str, also_allow_groups: list[str] | None = None
) -> set[str]:
    """Build the set of allowed top-level import names for a dependency group.

    Args:
        group_name: The optional-dependency group (or "core" for base only).
        also_allow_groups: Extra optional-dependency groups whose packages
            should also be allowed.  Use this for groups that are imported
            opportunistically via ``try/except ImportError`` guards — the
            code works without them, but they *will* appear in
            ``sys.modules`` when installed.
    """
    # Parse pyproject.toml
    pyproject_path = Path(__file__).resolve().parent.parent.parent / "pyproject.toml"

    with open(pyproject_path, "rb") as f:
        pyproject = tomllib.load(f)

    # Collect declared distribution names: core + specified group
    core_deps = pyproject.get("project", {}).get("dependencies", [])
    optional_deps = pyproject.get("project", {}).get("optional-dependencies", {})
    # "core" is a special pseudo-group meaning core deps only
    group_deps = [] if group_name == "core" else optional_deps.get(group_name, [])

    # Include deps from additionally-allowed groups
    extra_deps: list[str] = []
    for g in also_allow_groups or []:
        extra_deps.extend(optional_deps.get(g, []))

    # Expand self-references like 'mellea[hooks]' into that group's deps
    all_dep_specs = core_deps + group_deps + extra_deps
    expanded: list[str] = []
    seen_groups: set[str] = set()
    queue = list(all_dep_specs)
    while queue:
        spec = queue.pop(0)
        refs = extract_self_ref_groups(spec)
        if refs:
            for ref in refs:
                if ref not in seen_groups:
                    seen_groups.add(ref)
                    queue.extend(optional_deps.get(ref, []))
        else:
            expanded.append(spec)

    declared_dists: set[str] = set()
    for dep_spec in expanded:
        name = parse_dep_name(dep_spec)
        if name:
            declared_dists.add(name)

    # Resolve transitive dependencies
    all_allowed_dists: set[str] = set()
    for dist_name in declared_dists:
        all_allowed_dists |= get_transitive_deps(dist_name)

    # Map all allowed distributions to their importable top-level names
    allowed_imports: set[str] = set()
    for dist_name in all_allowed_dists:
        allowed_imports |= get_top_level_names(dist_name)

    # Also add the normalized dist names themselves (common pattern)
    for dist_name in all_allowed_dists:
        allowed_imports.add(dist_name.replace("-", "_").lower())

    return allowed_imports


def is_third_party(module_name: str) -> bool:
    """Check if a module name appears to be third-party (not stdlib, not local)."""
    top = module_name.split(".")[0]

    if top in STDLIB_AND_INFRASTRUCTURE or top in OPPORTUNISTIC_IMPORTS:
        return False

    # Skip internal/private modules
    if top.startswith("_"):
        return False

    # Skip mellea and cli (our own packages)
    if top in ("mellea", "cli", "test"):
        return False

    # Check if it's a known distribution
    try:
        importlib.metadata.distribution(top)
        return True
    except importlib.metadata.PackageNotFoundError:
        pass

    # Try with hyphens replaced
    try:
        importlib.metadata.distribution(top.replace("_", "-"))
        return True
    except importlib.metadata.PackageNotFoundError:
        pass

    # Not a known distribution — likely stdlib
    return False


def main() -> int:
    # Parse --allow-group flags before positional args
    also_allow: list[str] = []
    positional: list[str] = []
    args = sys.argv[1:]
    while args:
        # Iterates over all the args until the list is empty.
        if args[0] == "--allow-group" and len(args) >= 2:
            # Grabs <group> from ["--allow-group", "<group>", ...]
            also_allow.append(args[1])
            args = args[2:]
        else:
            positional.append(args[0])
            args = args[1:]

    if len(positional) < 2:
        print(
            f"Usage: {sys.argv[0]} [--allow-group GROUP ...] <group_name> <module1> [module2 ...]",
            file=sys.stderr,
        )
        return 2

    group_name = positional[0]
    target_modules = positional[1:]

    # Build allowed set
    allowed = build_allowed_set(group_name, also_allow_groups=also_allow)

    # Snapshot modules before import
    before = set(sys.modules.keys())

    # Import target modules
    for mod in target_modules:
        try:
            importlib.import_module(mod)
        except ImportError as e:
            print(f"IMPORT_ERROR: Could not import {mod}: {e}", file=sys.stderr)
            return 2

    # Find new third-party modules
    after = set(sys.modules.keys())
    new_modules = after - before

    violations: list[str] = []
    for mod in sorted(new_modules):
        top = mod.split(".")[0]
        if not is_third_party(top):
            # It's a standard python package.
            continue
        if top.lower() in allowed or top.replace("-", "_").lower() in allowed:
            # It's allowed by the current group or an explicitly allowed group.
            continue
        violations.append(top)

    # Deduplicate
    violations = sorted(set(violations))

    if violations:
        print(f"VIOLATIONS for group '{group_name}':")
        for v in violations:
            print(f"  - {v}")
        return 1

    print(f"OK: group '{group_name}' imports only declared dependencies")
    return 0


if __name__ == "__main__":
    sys.exit(main())
