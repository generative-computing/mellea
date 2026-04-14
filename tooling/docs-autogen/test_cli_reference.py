"""Tests for generate-cli-reference.py."""

from __future__ import annotations

import re

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def click_app():
    """Import and return the Click command tree for the ``m`` CLI."""
    import typer.main

    from cli.m import cli

    return typer.main.get_command(cli)


@pytest.fixture(scope="module")
def generated_md(click_app):
    """Generate the full CLI reference Markdown string."""
    from generate_cli_reference import generate_cli_reference

    return generate_cli_reference(click_app)


# ---------------------------------------------------------------------------
# Import / introspection tests
# ---------------------------------------------------------------------------


def test_click_app_is_group(click_app):
    """Root app should be a Click Group with commands."""
    assert hasattr(click_app, "commands")
    assert len(click_app.commands) > 0


EXPECTED_TOP_LEVEL = {"serve", "alora", "decompose", "eval", "fix"}


def test_all_top_level_commands_present(click_app):
    """All expected top-level commands must be discovered."""
    assert EXPECTED_TOP_LEVEL <= set(click_app.commands.keys())


EXPECTED_SUBCOMMANDS = {
    "alora": {"train", "upload", "add-readme"},
    "decompose": {"run"},
    "eval": {"run"},
    "fix": {"async", "genslots"},
}


@pytest.mark.parametrize(
    "group,expected_subs", list(EXPECTED_SUBCOMMANDS.items()), ids=EXPECTED_SUBCOMMANDS
)
def test_subcommands_present(click_app, group, expected_subs):
    """Each command group must contain its expected subcommands."""
    cmd = click_app.commands[group]
    assert hasattr(cmd, "commands"), f"{group} should be a command group"
    assert expected_subs <= set(cmd.commands.keys())


def test_serve_is_not_a_group(click_app):
    """serve is a direct command, not a group."""
    serve = click_app.commands["serve"]
    assert not hasattr(serve, "commands")


# ---------------------------------------------------------------------------
# Docstring parsing tests
# ---------------------------------------------------------------------------


def test_parse_docstring_sections_basic():
    from generate_cli_reference import _parse_docstring_sections

    result = _parse_docstring_sections(
        """One-line summary.

    Extended body here.

    Prerequisites:
        Some prereq.

    See Also:
        guide: foo/bar
    """
    )
    assert result["summary"] == "One-line summary."
    assert "Extended body" in result["body"]
    assert "Some prereq" in result["Prerequisites"]
    assert "guide: foo/bar" in result["See Also"]


def test_parse_docstring_sections_empty():
    from generate_cli_reference import _parse_docstring_sections

    result = _parse_docstring_sections(None)
    assert result["summary"] == ""
    assert result["body"] == ""


def test_parse_see_also():
    from generate_cli_reference import _parse_see_also

    links = _parse_see_also("guide: how-to/my-page\nguide: advanced/other")
    assert links == [("guide", "how-to/my-page"), ("guide", "advanced/other")]


# ---------------------------------------------------------------------------
# Generated output tests
# ---------------------------------------------------------------------------


def test_frontmatter_present(generated_md):
    assert generated_md.startswith("---\n")
    assert 'title: "CLI Reference"' in generated_md


def test_all_commands_in_output(generated_md):
    """Every expected command should appear as a heading in the output."""
    assert "## `m serve`" in generated_md
    assert "## `m alora`" in generated_md
    assert "## `m decompose`" in generated_md
    assert "## `m eval`" in generated_md
    assert "## `m fix`" in generated_md


def test_subcommands_in_output(generated_md):
    """Subcommands should appear as H3 headings."""
    assert "### `m alora train`" in generated_md
    assert "### `m alora upload`" in generated_md
    assert "### `m decompose run`" in generated_md
    assert "### `m eval run`" in generated_md
    assert "### `m fix async`" in generated_md
    assert "### `m fix genslots`" in generated_md


def test_options_tables_present(generated_md):
    """Options tables should be present for commands with flags."""
    assert "| Flag | Type | Default | Description |" in generated_md


def test_prerequisites_rendered(generated_md):
    """Prerequisites should be rendered as blockquote callouts."""
    assert "> **Prerequisites:**" in generated_md


def test_see_also_links_rendered(generated_md):
    """See also links should be rendered as markdown links."""
    assert "**See also:**" in generated_md
    # Should contain at least one relative link
    assert re.search(r"\[.*?\]\(\.\./.*?\)", generated_md)


def test_synopsis_present(generated_md):
    """Each command should have a code-fenced synopsis."""
    assert "```bash\nm serve" in generated_md
    assert "```bash\nm alora train" in generated_md


def test_output_sections_rendered(generated_md):
    """Output sections should be rendered for commands that define them."""
    assert "**Output:**" in generated_md


def test_rst_backticks_converted(generated_md):
    """RST-style double-backticks from docstrings should be converted."""
    # The description column text should not contain rst-style ``backticks``
    # (flag names in the Name column use markdown ``code`` which is fine)
    # Check that description text in tables doesn't have rst backticks
    for line in generated_md.splitlines():
        if line.startswith("|") and "| ``--" not in line and "| ``" not in line:
            continue
        # Check descriptions (last column) for rst backticks
        if line.startswith("|"):
            cols = line.split("|")
            if len(cols) >= 5:
                desc = cols[-2]  # Description is second-to-last (before trailing |)
                assert "``" not in desc, f"RST backticks in description: {desc}"


def test_strict_validation_passes(click_app):
    """Strict validation should pass with no errors for all current commands."""
    from generate_cli_reference import validate_cli_reference

    errors = validate_cli_reference(click_app)
    assert errors == [], f"Strict validation errors: {errors}"


def test_no_mdx_or_framework_specific_syntax(generated_md):
    """Output should be standard Markdown, no MDX components."""
    assert "<Callout" not in generated_md
    assert "<Tab" not in generated_md
    assert "import " not in generated_md.split("---")[-1]  # After frontmatter
