#!/usr/bin/env python3
# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for validate.py validation functions."""

import tempfile
from pathlib import Path

import pytest
from validate import (
    generate_report,
    validate_doc_imports,
    validate_examples_catalogue,
    validate_internal_links,
    validate_mdx_syntax,
    validate_rst_docstrings,
    validate_source_links,
    validate_stale_files,
)


def test_validate_source_links_pass():
    """Test source link validation passes with correct links."""
    with tempfile.TemporaryDirectory() as tmpdir:
        docs_dir = Path(tmpdir)
        test_file = docs_dir / "test.mdx"
        test_file.write_text(
            "---\ntitle: Test\n---\n\n"
            "[View source](https://github.com/ibm-granite/mellea/blob/v0.5.0/mellea/core/base.py#L10)"
        )

        error_count, errors = validate_source_links(docs_dir, "0.5.0")
        assert error_count == 0
        assert len(errors) == 0


def test_validate_source_links_fail():
    """Test source link validation fails with incorrect links."""
    with tempfile.TemporaryDirectory() as tmpdir:
        docs_dir = Path(tmpdir)
        test_file = docs_dir / "test.mdx"
        test_file.write_text(
            "---\ntitle: Test\n---\n\n"
            "[View source](https://github.com/wrong-org/mellea/blob/v0.5.0/mellea/core/base.py#L10)"
        )

        error_count, errors = validate_source_links(docs_dir, "0.5.0")
        assert error_count == 1
        assert len(errors) == 1
        assert "Invalid source link" in errors[0]


def test_validate_mdx_syntax_pass():
    """Test MDX syntax validation passes with valid MDX."""
    with tempfile.TemporaryDirectory() as tmpdir:
        docs_dir = Path(tmpdir)
        test_file = docs_dir / "test.mdx"
        test_file.write_text(
            "---\ntitle: Test\n---\n\n# Header\n\n```python\ncode\n```\n"
        )

        error_count, errors = validate_mdx_syntax(docs_dir)
        assert error_count == 0
        assert len(errors) == 0


def test_validate_mdx_syntax_missing_frontmatter():
    """Test MDX syntax validation fails without frontmatter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        docs_dir = Path(tmpdir)
        test_file = docs_dir / "test.mdx"
        test_file.write_text("# Header\n\nContent")

        error_count, errors = validate_mdx_syntax(docs_dir)
        assert error_count == 1
        assert "Missing frontmatter" in errors[0]


def test_validate_mdx_syntax_unclosed_code_block():
    """Test MDX syntax validation fails with unclosed code block."""
    with tempfile.TemporaryDirectory() as tmpdir:
        docs_dir = Path(tmpdir)
        test_file = docs_dir / "test.mdx"
        test_file.write_text("---\ntitle: Test\n---\n\n```python\ncode\n")

        error_count, errors = validate_mdx_syntax(docs_dir)
        assert error_count == 1
        assert "Unclosed code block" in errors[0]


def test_validate_internal_links_pass():
    """Test internal link validation passes with valid links."""
    with tempfile.TemporaryDirectory() as tmpdir:
        docs_dir = Path(tmpdir)

        # Create two files
        file1 = docs_dir / "file1.mdx"
        file2 = docs_dir / "file2.mdx"

        file1.write_text("---\ntitle: File 1\n---\n\n[Link to file 2](file2.mdx)")
        file2.write_text("---\ntitle: File 2\n---\n\nContent")

        error_count, errors = validate_internal_links(docs_dir)
        assert error_count == 0
        assert len(errors) == 0


def test_validate_internal_links_broken():
    """Test internal link validation fails with broken links."""
    with tempfile.TemporaryDirectory() as tmpdir:
        docs_dir = Path(tmpdir)
        test_file = docs_dir / "test.mdx"
        test_file.write_text("---\ntitle: Test\n---\n\n[Broken link](nonexistent.mdx)")

        error_count, errors = validate_internal_links(docs_dir)
        assert error_count == 1
        assert "Broken link" in errors[0]


def test_validate_internal_links_external_ignored():
    """Test internal link validation ignores external links."""
    with tempfile.TemporaryDirectory() as tmpdir:
        docs_dir = Path(tmpdir)
        test_file = docs_dir / "test.mdx"
        test_file.write_text(
            "---\ntitle: Test\n---\n\n"
            "[External](https://example.com)\n"
            "[Anchor](#section)"
        )

        error_count, errors = validate_internal_links(docs_dir)
        assert error_count == 0
        assert len(errors) == 0


def test_validate_rst_docstrings_detects_multiline_markup():
    """Test RST literal detection across docstring lines."""
    with tempfile.TemporaryDirectory() as tmpdir:
        source_dir = Path(tmpdir) / "source"
        source_dir.mkdir()
        (source_dir / "example.py").write_text(
            'def example():\n    """Use ``multi-line\n    value`` in prose."""\n',
            encoding="utf-8",
        )

        error_count, errors = validate_rst_docstrings(source_dir)

        assert error_count == 1
        assert errors[0]["file"] == "source/example.py"
        assert errors[0]["line"] == 2


def test_validate_rst_docstrings_detects_embedded_backticks():
    """Test RST literal detection when its content contains backticks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        source_dir = Path(tmpdir) / "source"
        source_dir.mkdir()
        (source_dir / "example.py").write_text(
            'def example():\n    """Avoid ``value with `inner` marker`` in prose."""\n',
            encoding="utf-8",
        )

        error_count, errors = validate_rst_docstrings(source_dir)

        assert error_count == 1
        assert errors[0]["line"] == 2


def test_validate_rst_docstrings_detects_attribute_descriptions():
    """Test attribute, explicit module, and field documentation strings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        source_dir = Path(tmpdir) / "source"
        source_dir.mkdir()
        (source_dir / "example.py").write_text(
            '__doc__ = "Module uses ``literal``."\n'
            'SETTING = "value"\n'
            '"""Setting uses ``literal``."""\n'
            "\n"
            "class Options:\n"
            '    field: str = Field(description="Field uses ``literal``.")\n'
            "\n"
            "    def __init__(self):\n"
            '        self.value = "value"\n'
            '        """Value uses ``literal``."""\n',
            encoding="utf-8",
        )

        error_count, errors = validate_rst_docstrings(source_dir)

        assert error_count == 4
        assert [error["line"] for error in errors] == [1, 3, 6, 10]


def test_validate_rst_docstrings_handles_explicit_doc_scopes():
    """Test explicit documentation assignments without matching local fixtures."""
    with tempfile.TemporaryDirectory() as tmpdir:
        source_dir = Path(tmpdir) / "source"
        source_dir.mkdir()
        (source_dir / "example.py").write_text(
            "class Documented:\n"
            "    pass\n"
            "\n"
            'Documented.__doc__ = "Class uses ``literal``."\n'
            "\n"
            "def fixture():\n"
            '    __doc__ = "Runtime ``fixture`` input."\n'
            "    return __doc__\n",
            encoding="utf-8",
        )

        error_count, errors = validate_rst_docstrings(source_dir)

        assert error_count == 1
        assert errors[0]["line"] == 4


def test_validate_rst_docstrings_reports_source_line_for_escaped_newline():
    """Test line reporting uses source text rather than the decoded string value."""
    with tempfile.TemporaryDirectory() as tmpdir:
        source_dir = Path(tmpdir) / "source"
        source_dir.mkdir()
        (source_dir / "example.py").write_text(
            'def example():\n    """Before \\n``literal`` on one source line."""\n',
            encoding="utf-8",
        )

        error_count, errors = validate_rst_docstrings(source_dir)

        assert error_count == 1
        assert errors[0]["line"] == 2


def test_validate_rst_docstrings_reports_source_line_for_concatenated_literal():
    """Test line reporting across implicitly concatenated string literals."""
    with tempfile.TemporaryDirectory() as tmpdir:
        source_dir = Path(tmpdir) / "source"
        source_dir.mkdir()
        (source_dir / "example.py").write_text(
            "def example():\n"
            "    (\n"
            '        "Before "\n'
            '        "``literal`` in prose."\n'
            "    )\n",
            encoding="utf-8",
        )

        error_count, errors = validate_rst_docstrings(source_dir)

        assert error_count == 1
        assert errors[0]["line"] == 4


def test_validate_rst_docstrings_reports_source_line_for_split_literal():
    """Test line reporting when RST markup spans concatenated string tokens."""
    with tempfile.TemporaryDirectory() as tmpdir:
        source_dir = Path(tmpdir) / "source"
        source_dir.mkdir()
        (source_dir / "example.py").write_text(
            "def example():\n"
            "    (\n"
            '        "Before "\n'
            '        "``literal "\n'
            '        "continued`` in prose."\n'
            "    )\n",
            encoding="utf-8",
        )

        error_count, errors = validate_rst_docstrings(source_dir)

        assert error_count == 1
        assert errors[0]["line"] == 4


def test_validate_rst_docstrings_ignores_non_docstrings_and_fences():
    """Test that only prose outside fenced blocks is validated."""
    with tempfile.TemporaryDirectory() as tmpdir:
        source_dir = Path(tmpdir) / "source"
        source_dir.mkdir()
        (source_dir / "example.py").write_text(
            'TEXT = "Use ``literal`` as fixture input"\n'
            "\n"
            "def example():\n"
            '    """Show a fenced fixture.\n'
            "\n"
            "    ```python\n"
            '    value = "``literal``"\n'
            "    ```\n"
            "\n"
            "    ~~~~text\n"
            "    ``tilde literal``\n"
            "    ~~~~\n"
            "\n"
            "    ````text\n"
            "    ```\n"
            "    ``long fence literal``\n"
            "    ```\n"
            "    ````\n"
            '    """\n'
            '    return "``literal``"\n'
            "\n"
            "def direct_fence():\n"
            '    """```text\n'
            "    ``direct fence literal``\n"
            "    ```\n"
            '    """\n'
            "\n"
            "def escaped_fence():\n"
            '    """```text\\n``escaped fence literal``\\n```"""\n'
            "\n"
            "def concatenated_fixture():\n"
            "    (\n"
            '        "Safe "\n'
            "        # ``comment fixture``\n"
            '        "documentation."\n'
            "    )\n",
            encoding="utf-8",
        )

        error_count, errors = validate_rst_docstrings(source_dir)

        assert error_count == 0
        assert errors == []


@pytest.mark.parametrize("source_root", ["mellea", "cli", "test", "docs", "tooling"])
def test_repository_docstrings_use_markdown_backticks(source_root):
    """Test that repository documentation strings follow Markdown style."""
    repo_root = Path(__file__).resolve().parents[2]

    error_count, errors = validate_rst_docstrings(repo_root / source_root)

    details = "\n".join(
        f"{error['file']}:{error['line']}: {error['message']}" for error in errors
    )
    assert error_count == 0, details


def test_validate_stale_files_clean():
    """Test stale-file check passes when no stale files exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        docs_root = Path(tmpdir)
        (docs_root / "docs").mkdir()
        (docs_root / "PUBLISHING.md").write_text("legit file")

        error_count, errors = validate_stale_files(docs_root)
        assert error_count == 0
        assert len(errors) == 0


def test_validate_stale_files_review_artifact():
    """Test stale-file check catches review tracker files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        docs_root = Path(tmpdir)
        (docs_root / "PR601-REVIEW.md").write_text("review notes")

        error_count, errors = validate_stale_files(docs_root)
        assert error_count == 1
        assert "review artifact" in errors[0].lower()


def test_validate_stale_files_superseded_index():
    """Test stale-file check catches superseded index.md."""
    with tempfile.TemporaryDirectory() as tmpdir:
        docs_root = Path(tmpdir)
        (docs_root / "index.md").write_text("old landing page")
        (docs_root / "docs").mkdir()
        (docs_root / "docs" / "index.mdx").write_text("new landing page")

        error_count, errors = validate_stale_files(docs_root)
        assert error_count == 1
        assert "superseded" in errors[0].lower()


def test_validate_stale_files_superseded_tutorial():
    """Test stale-file check catches superseded tutorial.md."""
    with tempfile.TemporaryDirectory() as tmpdir:
        docs_root = Path(tmpdir)
        (docs_root / "tutorial.md").write_text("old tutorial")
        (docs_root / "docs").mkdir()
        (docs_root / "docs" / "tutorials").mkdir()

        error_count, errors = validate_stale_files(docs_root)
        assert error_count == 1
        assert "superseded" in errors[0].lower()


def test_validate_doc_imports_pass():
    """Test doc import check passes with valid imports."""
    with tempfile.TemporaryDirectory() as tmpdir:
        docs_dir = Path(tmpdir)
        test_file = docs_dir / "test.md"
        test_file.write_text(
            "# Example\n\n```python\nimport os\nfrom pathlib import Path\n```\n"
        )

        error_count, errors = validate_doc_imports(docs_dir)
        assert error_count == 0
        assert len(errors) == 0


def test_validate_doc_imports_bad_symbol():
    """Test doc import check catches missing symbols."""
    with tempfile.TemporaryDirectory() as tmpdir:
        docs_dir = Path(tmpdir)
        test_file = docs_dir / "test.md"
        # mellea.core exists but NoSuchSymbol does not
        test_file.write_text("```python\nfrom mellea.core import NoSuchSymbol\n```\n")

        error_count, errors = validate_doc_imports(docs_dir)
        assert error_count == 1
        assert "symbol not found" in errors[0]


def test_validate_doc_imports_skips_non_python_blocks():
    """Test doc import check ignores non-python code blocks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        docs_dir = Path(tmpdir)
        test_file = docs_dir / "test.md"
        test_file.write_text("```bash\nfrom mellea.nonexistent import Foo\n```\n")

        error_count, _errors = validate_doc_imports(docs_dir)
        assert error_count == 0


def test_generate_report():
    """Test report generation."""
    report = generate_report(
        source_link_errors=["error1"],
        coverage_passed=False,
        coverage_report={
            "coverage_percentage": 50,
            "total_symbols": 10,
            "documented_symbols": 5,
        },
        mdx_errors=["error2"],
        link_errors=["error3"],
        anchor_errors=["error4"],
    )

    assert report["source_links"]["passed"] is False
    assert report["source_links"]["error_count"] == 1
    assert report["coverage"]["passed"] is False
    assert report["coverage"]["percentage"] == 50
    assert report["mdx_syntax"]["passed"] is False
    assert report["internal_links"]["passed"] is False
    assert report["anchor_collisions"]["passed"] is False
    assert report["overall_passed"] is False


def test_generate_report_all_pass():
    """Test report generation with all checks passing."""
    report = generate_report(
        source_link_errors=[],
        coverage_passed=True,
        coverage_report={
            "coverage_percentage": 90,
            "total_symbols": 10,
            "documented_symbols": 9,
        },
        mdx_errors=[],
        link_errors=[],
        anchor_errors=[],
    )

    assert report["source_links"]["passed"] is True
    assert report["coverage"]["passed"] is True
    assert report["mdx_syntax"]["passed"] is True
    assert report["internal_links"]["passed"] is True
    assert report["overall_passed"] is True


def test_validate_examples_catalogue_pass():
    """Test examples catalogue check passes when all dirs are listed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        docs_root = Path(tmpdir)
        examples_dir = docs_root / "examples"
        examples_dir.mkdir()
        index_dir = docs_root / "docs" / "examples"
        index_dir.mkdir(parents=True)

        # Create an example directory with a .py file
        (examples_dir / "my_example").mkdir()
        (examples_dir / "my_example" / "demo.py").write_text("# demo")

        # Create index listing it
        (index_dir / "index.md").write_text("| `my_example/` | A demo example |")

        error_count, errors = validate_examples_catalogue(docs_root)
        assert error_count == 0
        assert len(errors) == 0


def test_validate_examples_catalogue_missing():
    """Test examples catalogue check catches unlisted directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        docs_root = Path(tmpdir)
        examples_dir = docs_root / "examples"
        examples_dir.mkdir()
        index_dir = docs_root / "docs" / "examples"
        index_dir.mkdir(parents=True)

        # Create an example directory with a .py file
        (examples_dir / "unlisted_example").mkdir()
        (examples_dir / "unlisted_example" / "demo.py").write_text("# demo")

        # Create index that does NOT mention it
        (index_dir / "index.md").write_text("| `other/` | Something else |")

        error_count, errors = validate_examples_catalogue(docs_root)
        assert error_count == 1
        assert "unlisted_example" in errors[0]


def test_validate_examples_catalogue_skips_helper():
    """Test examples catalogue check skips the helper directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        docs_root = Path(tmpdir)
        examples_dir = docs_root / "examples"
        examples_dir.mkdir()
        index_dir = docs_root / "docs" / "examples"
        index_dir.mkdir(parents=True)

        # Create helper directory (should be skipped)
        (examples_dir / "helper").mkdir()
        (examples_dir / "helper" / "utils.py").write_text("# utils")

        (index_dir / "index.md").write_text("# Examples")

        error_count, _errors = validate_examples_catalogue(docs_root)
        assert error_count == 0


def test_validate_examples_catalogue_skips_empty_dirs():
    """Test examples catalogue check skips directories with no .py files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        docs_root = Path(tmpdir)
        examples_dir = docs_root / "examples"
        examples_dir.mkdir()
        index_dir = docs_root / "docs" / "examples"
        index_dir.mkdir(parents=True)

        # Create directory with only non-Python files
        (examples_dir / "data_only").mkdir()
        (examples_dir / "data_only" / "data.json").write_text("{}")

        (index_dir / "index.md").write_text("# Examples")

        error_count, _errors = validate_examples_catalogue(docs_root)
        assert error_count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
