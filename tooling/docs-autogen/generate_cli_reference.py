#!/usr/bin/env python3
"""Generate a CLI reference page from Typer command metadata.

Imports the ``cli.m`` Typer application, introspects its Click command tree,
and emits a single Markdown reference page (``reference/cli.md``) documenting
every command, its flags, defaults, and descriptions.

Structured docstring sections (``Prerequisites:``, ``See Also:``) in command
functions are extracted and rendered as admonitions and cross-links.

Run via::

    uv run python tooling/docs-autogen/generate_cli_reference.py
"""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

import click

# ---------------------------------------------------------------------------
# Docstring parsing
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(
    r"^(Prerequisites|See Also|Output|Examples|Args|Raises|Returns|Attributes|Yields):\s*$",
    re.MULTILINE,
)

# RST-style double-backtick → markdown single-backtick
_RST_BACKTICK_RE = re.compile(r"``([^`]+)``")


def _parse_docstring_sections(docstring: str | None) -> dict[str, str]:
    """Parse a Google-style docstring into named sections.

    Returns a dict with keys ``"summary"``, ``"body"``, and any structured
    section names found (e.g. ``"Prerequisites"``, ``"See Also"``).
    """
    if not docstring:
        return {"summary": "", "body": ""}

    lines = textwrap.dedent(docstring).strip().splitlines()

    # First non-empty line is the summary
    summary = lines[0].strip() if lines else ""

    # Collect body lines until we hit a structured section
    body_lines: list[str] = []
    sections: dict[str, list[str]] = {}
    current_section: str | None = None
    i = 1

    while i < len(lines):
        line = lines[i]
        match = _SECTION_RE.match(line.strip())
        if match:
            section_name: str = match.group(1)
            current_section = section_name
            sections[section_name] = []
        elif current_section is not None:
            sections[current_section].append(line)
        else:
            body_lines.append(line)
        i += 1

    result: dict[str, str] = {"summary": summary, "body": "\n".join(body_lines).strip()}
    for name, section_lines in sections.items():
        result[name] = textwrap.dedent("\n".join(section_lines)).strip()
    return result


def _parse_see_also(see_also_text: str) -> list[tuple[str, str]]:
    """Parse ``See Also`` entries into ``(kind, path)`` tuples.

    Expected format::

        guide: getting-started/quickstart
        guide: how-to/refactor-prompts-with-cli
    """
    links: list[tuple[str, str]] = []
    for line in see_also_text.strip().splitlines():
        line = line.strip()
        if ":" in line:
            kind, _, path = line.partition(":")
            links.append((kind.strip(), path.strip()))
    return links


def _rst_to_md(text: str) -> str:
    """Convert RST-style double-backticks to Markdown single-backticks."""
    return _RST_BACKTICK_RE.sub(r"`\1`", text)


_TITLE_LOWERCASE = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "vs",
    "with",
}


def _slug_to_title(slug: str) -> str:
    """Convert a URL slug to a human-readable title with proper capitalisation."""
    words = slug.replace("-", " ").split()
    result = []
    for i, word in enumerate(words):
        if i == 0 or word.lower() not in _TITLE_LOWERCASE:
            result.append(word.capitalize())
        else:
            result.append(word.lower())
    return " ".join(result)


# ---------------------------------------------------------------------------
# Click model traversal
# ---------------------------------------------------------------------------


def _get_click_app():
    """Import and return the Click command tree for the ``m`` CLI."""
    try:
        import typer.main

        from cli.m import cli
    except ImportError as e:
        raise SystemExit(
            f"❌ Failed to import CLI application: {e}\n"
            "   Install all extras: uv sync --all-extras --group dev"
        ) from e

    return typer.main.get_command(cli)


def _extract_verbatim_blocks(help_text: str) -> list[str]:
    """Extract Click ``\\b`` verbatim blocks from help text.

    Click uses the backspace character (``\\x08``, written as ``\\b`` in Python
    source string literals) as a marker to prevent paragraph rewrapping in
    ``--help`` output.  The generator must extract these blocks independently
    because the section parser buries them inside whatever named section
    (e.g. ``Raises``) happens to precede them, and that section is never
    rendered.

    Returns a list of stripped block strings (first line is typically the
    block title, e.g. ``"Modes:"``, followed by indented content lines).
    """
    # In memory the docstring contains actual \x08 chars; split on them.
    parts = re.split(r"\x08\s*\n", help_text)
    blocks: list[str] = []
    for part in parts[1:]:  # parts[0] is content before the first \b
        block = part.rstrip()
        if block.strip():
            blocks.append(block)
    return blocks


def _format_default(value: Any) -> str:
    """Format a parameter default for display."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    # Handle enum defaults (e.g. DecompBackend.ollama)
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def _format_type(param: click.Parameter) -> str:
    """Format a Click parameter type for display."""
    type_name = param.type.name
    if hasattr(param.type, "choices"):
        return " \\| ".join(param.type.choices)
    return type_name


def _format_flags(param: click.Parameter) -> str:
    """Format parameter flags (e.g. ``--backend, -b``)."""
    is_arg = isinstance(param, click.Argument)
    if is_arg:
        name = param.name or ""
        return f"`{name.upper()}`"
    opts = list(getattr(param, "opts", []))
    secondary = list(getattr(param, "secondary_opts", []))
    # Filter out --no-* boolean counterparts
    all_opts = [o for o in opts + secondary if not o.startswith("--no-")]
    return ", ".join(f"`{o}`" for o in all_opts)


def _is_help_param(param: click.Parameter) -> bool:
    """Return True for Click's auto-generated --help parameter."""
    return param.name == "help" and not isinstance(param, click.Argument)


def _build_synopsis(full_name: str, cmd: click.BaseCommand) -> str:
    """Build a usage synopsis line for a command."""
    parts = [full_name]
    for param in cmd.params:
        if _is_help_param(param):
            continue
        is_arg = isinstance(param, click.Argument)
        if is_arg:
            name = (param.name or "").upper()
            if param.required:
                parts.append(f"<{name}>")
            else:
                parts.append(f"[{name}]")
        else:
            opts = list(getattr(param, "opts", []))
            long_opt = next(
                (o for o in opts if o.startswith("--")), opts[0] if opts else ""
            )
            if param.required:
                parts.append(f"{long_opt} <value>")
            else:
                parts.append(f"[{long_opt}]")
    return " ".join(parts)


def _render_command(
    full_name: str, cmd: click.BaseCommand, heading_level: int
) -> list[str]:
    """Render a single command as Markdown lines."""
    lines: list[str] = []
    heading = "#" * heading_level
    lines.append(f"{heading} `{full_name}`")
    lines.append("")

    # Parse docstring
    sections = _parse_docstring_sections(cmd.help)

    # Summary
    summary = _rst_to_md(sections.get("summary", ""))
    if summary:
        lines.append(summary)
        lines.append("")

    # Extended description
    body = _rst_to_md(sections.get("body", ""))
    if body:
        # Strip Click's \b formatting markers
        body = body.replace("\b", "").strip()
        if body:
            lines.append(body)
            lines.append("")

    # Prerequisites — render as bulleted blockquote list
    prereqs = _rst_to_md(sections.get("Prerequisites", ""))
    if prereqs:
        # Join continuation lines into single string, then split on sentences
        prereqs_joined = " ".join(prereqs.split())
        items = re.split(r"\.\s+(?=[A-Z])", prereqs_joined.strip())
        items = [item.rstrip(".").strip() for item in items if item.strip()]
        if len(items) == 1:
            lines.append(f"> **Prerequisites:** {items[0]}.")
        else:
            lines.append("> **Prerequisites:**")
            lines.append(">")
            for item in items:
                lines.append(f"> - {item}.")
        lines.append("")

    # Synopsis
    lines.append("```bash")
    lines.append(_build_synopsis(full_name, cmd))
    lines.append("```")
    lines.append("")

    # Options table — split into arguments and options, exclude --help
    arguments = [p for p in cmd.params if isinstance(p, click.Argument)]
    options = [
        p
        for p in cmd.params
        if not isinstance(p, click.Argument) and not _is_help_param(p)
    ]

    if arguments:
        lines.append("**Arguments:**")
        lines.append("")
        lines.append("| Name | Type | Required | Description |")
        lines.append("| ---- | ---- | -------- | ----------- |")
        for p in arguments:
            flags = _format_flags(p)
            ptype = _format_type(p)
            required = "yes" if p.required else "no"
            help_text = _rst_to_md(getattr(p, "help", "") or "")
            lines.append(f"| {flags} | {ptype} | {required} | {help_text} |")
        lines.append("")

    if options:
        lines.append("**Options:**")
        lines.append("")
        lines.append("| Flag | Type | Default | Description |")
        lines.append("| ---- | ---- | ------- | ----------- |")
        for p in options:
            flags = _format_flags(p)
            ptype = _format_type(p)
            default_str = _format_default(p.default)
            if p.required:
                default = "*required*"
            elif default_str:
                default = f"`{default_str}`"
            else:
                default = "—"
            help_text = _rst_to_md(getattr(p, "help", "") or "—")
            lines.append(f"| {flags} | {ptype} | {default} | {help_text} |")
        lines.append("")

    # \b verbatim blocks — Click-style preformatted sections (e.g. "Modes:",
    # "Best practices:") that appear in --help but are buried inside the
    # Raises/Args sections of the parsed docstring and would otherwise be lost.
    if cmd.help:
        vb_blocks = _extract_verbatim_blocks(cmd.help)
        for block in vb_blocks:
            first_line, _, rest = block.partition("\n")
            title = first_line.strip()
            if title:
                lines.append(f"**{title}**")
                lines.append("")
            if rest.strip():
                lines.append("```")
                lines.append(rest.rstrip())
                lines.append("```")
                lines.append("")

    # Output
    output = _rst_to_md(sections.get("Output", ""))
    if output:
        # Join continuation lines
        output_joined = " ".join(output.split())
        lines.append(f"**Output:** {output_joined}")
        lines.append("")

    # Examples
    examples = sections.get("Examples", "")
    if examples:
        lines.append("**Example:**")
        lines.append("")
        lines.append("```bash")
        for ex_line in examples.strip().splitlines():
            lines.append(ex_line.strip())
        lines.append("```")
        lines.append("")

    # See Also
    see_also = sections.get("See Also", "")
    if see_also:
        links = _parse_see_also(see_also)
        if links:
            see_parts = []
            for kind, path in links:
                if kind == "guide":
                    title = _slug_to_title(path.split("/")[-1])
                    see_parts.append(f"[{title}](../{path})")
            if see_parts:
                lines.append(f"**See also:** {', '.join(see_parts)}")
                lines.append("")

    return lines


# ---------------------------------------------------------------------------
# Full page generation
# ---------------------------------------------------------------------------

FRONTMATTER = """\
---
title: "CLI Reference"
sidebarTitle: "CLI Reference"
description: "Complete reference for the m command-line tool — all subcommands, flags, and defaults."
---
"""


def generate_cli_reference(click_app: click.BaseCommand) -> str:
    """Generate the full CLI reference page as a Markdown string."""
    lines: list[str] = [FRONTMATTER]

    # Intro from root callback docstring
    root_sections = _parse_docstring_sections(click_app.help)
    root_summary = _rst_to_md(root_sections.get("summary", ""))
    if root_summary:
        lines.append(root_summary)
        lines.append("")
    root_body = _rst_to_md(root_sections.get("body", ""))
    if root_body:
        lines.append(root_body)
        lines.append("")

    # Iterate commands
    if not hasattr(click_app, "commands"):
        return "\n".join(lines)

    for cmd_name in sorted(click_app.commands):
        cmd = click_app.commands[cmd_name]

        if hasattr(cmd, "commands") and cmd.commands:
            # Command group — render group heading then subcommands
            group_summary = ""
            if cmd.help:
                group_summary = cmd.help.split("\n")[0].strip()

            lines.append(f"## `m {cmd_name}`")
            lines.append("")
            if group_summary:
                lines.append(group_summary)
                lines.append("")

            for sub_name in sorted(cmd.commands):
                sub_cmd = cmd.commands[sub_name]
                lines.extend(
                    _render_command(
                        f"m {cmd_name} {sub_name}", sub_cmd, heading_level=3
                    )
                )
        else:
            # Top-level command
            lines.extend(_render_command(f"m {cmd_name}", cmd, heading_level=2))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Strict validation
# ---------------------------------------------------------------------------


def validate_cli_reference(click_app: click.BaseCommand) -> list[str]:
    """Validate CLI command docstrings for completeness.

    Returns a list of error messages. Empty list means all checks pass.
    """
    errors: list[str] = []

    if not hasattr(click_app, "commands"):
        errors.append("Root CLI app has no commands")
        return errors

    def _check_command(full_name: str, cmd: click.BaseCommand) -> None:
        sections = _parse_docstring_sections(cmd.help)
        summary = sections.get("summary", "").strip()

        if not summary:
            errors.append(f"{full_name}: missing docstring summary")

        if not sections.get("Prerequisites", "").strip():
            errors.append(f"{full_name}: missing Prerequisites section")

        if not sections.get("Output", "").strip():
            errors.append(f"{full_name}: missing Output section")

        # Check all options have help text (skip auto-generated --help)
        for param in cmd.params:
            if _is_help_param(param):
                continue
            help_text = getattr(param, "help", None)
            if not help_text:
                param_name = param.name or "(unnamed)"
                errors.append(f"{full_name}: option --{param_name} has no help text")

    for cmd_name in sorted(click_app.commands):
        cmd = click_app.commands[cmd_name]
        if hasattr(cmd, "commands") and cmd.commands:
            for sub_name in sorted(cmd.commands):
                sub_cmd = cmd.commands[sub_name]
                _check_command(f"m {cmd_name} {sub_name}", sub_cmd)
        else:
            _check_command(f"m {cmd_name}", cmd)

    return errors


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CLI reference documentation from Typer metadata."
    )
    parser.add_argument(
        "--docs-root",
        default=None,
        help="Docs root directory (defaults to docs/docs relative to repo root).",
    )
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Repository root (for sys.path setup). Defaults to two parents up.",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print to stdout instead of writing a file.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail with non-zero exit if any CLI docstring is incomplete.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = (
        Path(args.source_dir).resolve() if args.source_dir else script_dir.parents[1]
    )

    # Ensure repo root is on sys.path so cli/ can be imported
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    docs_root = Path(args.docs_root) if args.docs_root else repo_root / "docs" / "docs"
    output_path = docs_root / "reference" / "cli.md"

    print("🔧 Importing CLI application...", flush=True)
    click_app = _get_click_app()

    # Validate before writing — fail early with --strict
    errors = validate_cli_reference(click_app)
    if errors:
        print(f"\n⚠️  CLI docstring validation: {len(errors)} issue(s):", flush=True)
        for err in errors:
            print(f"  • {err}")
        if args.strict:
            print("\n❌ Strict mode: failing due to incomplete CLI docstrings.")
            sys.exit(1)
    else:
        print("✅ CLI docstring validation passed.")

    print("📝 Generating CLI reference...", flush=True)
    content = generate_cli_reference(click_app)

    if args.stdout:
        print(content)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)
        print(f"✅ CLI reference written to {output_path}")


if __name__ == "__main__":
    main()
