"""Tests for the `m fix async` CLI tool.

Tests AST-based detection and fixup of async calls (aact, ainstruct, aquery) after changing contract to no longer always await.
"""

import textwrap
from pathlib import Path

from cli.fix import _FixMode
from cli.fix.async_fixer import (
    FixLocation,
    FixResult,
    find_fixable_calls,
    fix_file,
    fix_path,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dedent(source: str) -> str:
    """Dedent and strip leading newline for inline test sources."""
    return textwrap.dedent(source).lstrip("\n")


def _loc_names(locs: list[FixLocation]) -> list[str]:
    return [loc.function_name for loc in locs]


# Common import line for session-based test snippets.
_SESSION_IMPORT = "from mellea import MelleaSession\n"


# ===================================================================
# Detection tests — bare / module / session calls
# ===================================================================


class TestDetection:
    """Calls with explicit strategy=None should be detected."""

    def test_bare_aact_strategy_none(self):
        src = _dedent("""\
            from mellea.stdlib.functional import aact
            import asyncio
            async def main():
                result, ctx = await aact(action, context, backend, strategy=None)
        """)
        locs = find_fixable_calls(src, Path("test.py"))
        assert len(locs) == 1
        assert locs[0].function_name == "aact"
        assert locs[0].call_style == "functional"

    def test_bare_ainstruct_strategy_none(self):
        src = _dedent("""\
            from mellea.stdlib.functional import ainstruct
            async def main():
                result, ctx = await ainstruct("do something", context, backend, strategy=None)
        """)
        locs = find_fixable_calls(src, Path("test.py"))
        assert len(locs) == 1
        assert locs[0].function_name == "ainstruct"

    def test_bare_aquery(self):
        """aquery has no strategy param — ALL calls are affected."""
        src = _dedent("""\
            from mellea.stdlib.functional import aquery
            async def main():
                result, ctx = await aquery(obj, "question", context, backend)
        """)
        locs = find_fixable_calls(src, Path("test.py"))
        assert len(locs) == 1
        assert locs[0].function_name == "aquery"

    def test_module_qualified_mfuncs(self):
        src = _dedent("""\
            import mellea.stdlib.functional as mfuncs
            async def main():
                result, ctx = await mfuncs.aact(action, context, backend, strategy=None)
        """)
        locs = find_fixable_calls(src, Path("test.py"))
        assert len(locs) == 1
        assert locs[0].call_style == "functional"

    def test_module_qualified_functional(self):
        src = _dedent("""\
            from mellea.stdlib import functional
            async def main():
                result, ctx = await functional.ainstruct("desc", ctx, be, strategy=None)
        """)
        locs = find_fixable_calls(src, Path("test.py"))
        assert len(locs) == 1
        assert locs[0].call_style == "functional"

    def test_module_qualified_functional_aquery(self):
        src = _dedent("""\
            from mellea.stdlib import functional
            async def main():
                result, ctx = await functional.aquery(obj, "q", ctx, be)
        """)
        locs = find_fixable_calls(src, Path("test.py"))
        assert len(locs) == 1
        assert locs[0].call_style == "functional"

    def test_session_method_aact(self):
        src = _dedent("""\
            from mellea import MelleaSession
            async def main():
                result = await session.aact(action, strategy=None)
        """)
        locs = find_fixable_calls(src, Path("test.py"))
        assert len(locs) == 1
        assert locs[0].call_style == "session"

    def test_session_method_aquery(self):
        src = _dedent("""\
            from mellea.stdlib.session import start_session
            async def main():
                result = await session.aquery(obj, "question")
        """)
        locs = find_fixable_calls(src, Path("test.py"))
        assert len(locs) == 1
        assert locs[0].call_style == "session"

    def test_multiple_calls_in_one_file(self):
        src = _dedent("""\
            from mellea.stdlib.functional import aact, ainstruct, aquery
            async def main():
                r1, c1 = await aact(a, ctx, be, strategy=None)
                r2, c2 = await ainstruct("x", ctx, be, strategy=None)
                r3, c3 = await aquery(obj, "q", ctx, be)
        """)
        locs = find_fixable_calls(src, Path("test.py"))
        assert len(locs) == 3
        assert set(_loc_names(locs)) == {"aact", "ainstruct", "aquery"}

    def test_custom_alias_import(self):
        """import mellea.stdlib.functional as f — custom alias works."""
        src = _dedent("""\
            import mellea.stdlib.functional as f
            async def main():
                result, ctx = await f.aact(action, context, backend, strategy=None)
        """)
        locs = find_fixable_calls(src, Path("test.py"))
        assert len(locs) == 1
        assert locs[0].call_style == "functional"

    def test_from_import_with_alias(self):
        """from mellea.stdlib.functional import aact as my_aact."""
        src = _dedent("""\
            from mellea.stdlib.functional import aact as my_aact
            async def main():
                result, ctx = await my_aact(action, context, backend, strategy=None)
        """)
        locs = find_fixable_calls(src, Path("test.py"))
        assert len(locs) == 1
        assert locs[0].function_name == "my_aact"
        assert locs[0].call_style == "functional"

    def test_from_stdlib_import_functional_with_alias(self):
        """from mellea.stdlib import functional as fn."""
        src = _dedent("""\
            from mellea.stdlib import functional as fn
            async def main():
                result, ctx = await fn.aact(action, context, backend, strategy=None)
        """)
        locs = find_fixable_calls(src, Path("test.py"))
        assert len(locs) == 1
        assert locs[0].call_style == "functional"

    def test_bare_name_without_import_not_detected(self):
        """Bare aact() without a matching import should NOT be detected."""
        src = _dedent("""\
            async def main():
                result, ctx = await aact(action, context, backend, strategy=None)
        """)
        locs = find_fixable_calls(src, Path("test.py"))
        assert len(locs) == 0

    def test_unknown_module_without_session_import_not_detected(self):
        """foo.aact() without any mellea session import should NOT be detected."""
        src = _dedent("""\
            async def main():
                result = await foo.aact(action, strategy=None)
        """)
        locs = find_fixable_calls(src, Path("test.py"))
        assert len(locs) == 0

    def test_unknown_module_with_session_import_detected(self):
        """foo.aact() IS detected when file imports mellea session types."""
        src = _dedent("""\
            from mellea import MelleaSession
            async def main():
                result = await foo.aact(action, strategy=None)
        """)
        locs = find_fixable_calls(src, Path("test.py"))
        assert len(locs) == 1
        assert locs[0].call_style == "session"

    def test_session_import_via_start_session(self):
        """from mellea.stdlib.session import start_session enables session detection."""
        src = _dedent("""\
            from mellea.stdlib.session import start_session
            async def main():
                result = await m.aact(action, strategy=None)
        """)
        locs = find_fixable_calls(src, Path("test.py"))
        assert len(locs) == 1
        assert locs[0].call_style == "session"


# ===================================================================
# Skip tests — calls that should NOT be detected
# ===================================================================


class TestSkipConditions:
    """Calls that should be left alone."""

    def test_skip_achat(self):
        """achat defaults await_result=True — never needs fixing."""
        src = _dedent("""\
            from mellea.stdlib.functional import achat
            async def main():
                msg, ctx = await achat("hello", context, backend)
        """)
        locs = find_fixable_calls(src, Path("test.py"))
        assert len(locs) == 0

    def test_skip_already_has_await_result_true(self):
        src = _dedent("""\
            from mellea.stdlib.functional import aact
            async def main():
                r, c = await aact(a, ctx, be, strategy=None, await_result=True)
        """)
        locs = find_fixable_calls(src, Path("test.py"))
        assert len(locs) == 0

    def test_skip_no_strategy_none_aact(self):
        """Default strategy is non-None — result is always computed."""
        src = _dedent("""\
            from mellea.stdlib.functional import aact
            async def main():
                r, c = await aact(a, ctx, be)
        """)
        locs = find_fixable_calls(src, Path("test.py"))
        assert len(locs) == 0

    def test_skip_no_strategy_none_ainstruct(self):
        src = _dedent("""\
            from mellea.stdlib.functional import ainstruct
            async def main():
                r, c = await ainstruct("desc", ctx, be)
        """)
        locs = find_fixable_calls(src, Path("test.py"))
        assert len(locs) == 0

    def test_skip_explicit_non_none_strategy(self):
        src = _dedent("""\
            from mellea.stdlib.functional import aact
            async def main():
                r, c = await aact(a, ctx, be, strategy=my_strategy)
        """)
        locs = find_fixable_calls(src, Path("test.py"))
        assert len(locs) == 0

    def test_skip_return_sampling_results_true(self):
        src = _dedent("""\
            from mellea.stdlib.functional import aact
            async def main():
                r = await aact(a, ctx, be, strategy=None, return_sampling_results=True)
        """)
        locs = find_fixable_calls(src, Path("test.py"))
        assert len(locs) == 0

    def test_skip_sync_function_call(self):
        """Non-awaited calls are not our target."""
        src = _dedent("""\
            from mellea.stdlib.functional import aact
            def main():
                r, c = aact(a, ctx, be, strategy=None)
        """)
        locs = find_fixable_calls(src, Path("test.py"))
        assert len(locs) == 0

    def test_aquery_skip_already_has_await_result(self):
        src = _dedent("""\
            from mellea.stdlib.functional import aquery
            async def main():
                r, c = await aquery(obj, "q", ctx, be, await_result=True)
        """)
        locs = find_fixable_calls(src, Path("test.py"))
        assert len(locs) == 0

    def test_skip_no_mellea_imports_at_all(self):
        """File with no mellea imports should have nothing detected."""
        src = _dedent("""\
            import something_else
            async def main():
                r = await session.aact(action, strategy=None)
                r2, c = await aact(a, ctx, be, strategy=None)
        """)
        locs = find_fixable_calls(src, Path("test.py"))
        assert len(locs) == 0


# ===================================================================
# Variable name extraction tests
# ===================================================================


class TestVariableExtraction:
    def test_tuple_unpack_functional(self):
        src = _dedent("""\
            from mellea.stdlib.functional import aact
            async def main():
                result, ctx = await aact(a, context, backend, strategy=None)
        """)
        locs = find_fixable_calls(src, Path("test.py"))
        assert locs[0].target_variable == "result"
        assert locs[0].context_variable == "ctx"

    def test_simple_assign_session(self):
        src = _dedent("""\
            from mellea import MelleaSession
            async def main():
                mot = await session.aact(action, strategy=None)
        """)
        locs = find_fixable_calls(src, Path("test.py"))
        assert locs[0].target_variable == "mot"
        assert locs[0].context_variable is None

    def test_different_variable_names(self):
        src = _dedent("""\
            from mellea.stdlib.functional import ainstruct
            async def main():
                output, new_ctx = await ainstruct("x", ctx, be, strategy=None)
        """)
        locs = find_fixable_calls(src, Path("test.py"))
        assert locs[0].target_variable == "output"

    def test_no_assignment(self):
        src = _dedent("""\
            from mellea import MelleaSession
            async def main():
                await session.aact(action, strategy=None)
        """)
        locs = find_fixable_calls(src, Path("test.py"))
        assert len(locs) == 1
        assert locs[0].target_variable is None

    def test_aquery_tuple_unpack(self):
        src = _dedent("""\
            from mellea.stdlib.functional import aquery
            async def main():
                answer, ctx = await aquery(obj, "q", context, backend)
        """)
        locs = find_fixable_calls(src, Path("test.py"))
        assert locs[0].target_variable == "answer"

    def test_aquery_simple_assign_session(self):
        src = _dedent("""\
            from mellea import MelleaSession
            async def main():
                answer = await session.aquery(obj, "q")
        """)
        locs = find_fixable_calls(src, Path("test.py"))
        assert locs[0].target_variable == "answer"
        assert locs[0].context_variable is None


# ===================================================================
# add-await-result fix tests
# ===================================================================


class TestAddAwaitResult:
    def test_single_line_fix(self, tmp_path):
        src = _dedent("""\
            from mellea.stdlib.functional import aact
            async def main():
                result, ctx = await aact(action, context, backend, strategy=None)
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        locs = fix_file(f, _FixMode.ADD_AWAIT_RESULT)
        fixed = f.read_text()
        assert "await_result=True" in fixed
        assert len(locs) == 1

    def test_multiline_fix(self, tmp_path):
        src = _dedent("""\
            from mellea.stdlib.functional import aact
            async def main():
                result, ctx = await aact(
                    action,
                    context,
                    backend,
                    strategy=None,
                )
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        fix_file(f, _FixMode.ADD_AWAIT_RESULT)
        fixed = f.read_text()
        assert "await_result=True" in fixed
        compile(fixed, str(f), "exec")

    def test_session_api_fix(self, tmp_path):
        src = _dedent("""\
            from mellea import MelleaSession
            async def main():
                result = await session.aact(action, strategy=None)
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        fix_file(f, _FixMode.ADD_AWAIT_RESULT)
        fixed = f.read_text()
        assert "await_result=True" in fixed

    def test_multiple_calls_fixed(self, tmp_path):
        src = _dedent("""\
            from mellea.stdlib.functional import aact, ainstruct
            async def main():
                r1, c1 = await aact(a, ctx, be, strategy=None)
                r2, c2 = await ainstruct("x", ctx, be, strategy=None)
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        locs = fix_file(f, _FixMode.ADD_AWAIT_RESULT)
        fixed = f.read_text()
        assert fixed.count("await_result=True") == 2
        assert len(locs) == 2
        compile(fixed, str(f), "exec")

    def test_preserves_other_code(self, tmp_path):
        src = _dedent("""\
            import asyncio
            from mellea.stdlib.functional import aact

            async def main():
                print("before")
                r, c = await aact(a, ctx, be, strategy=None)
                print("after")

            if __name__ == "__main__":
                asyncio.run(main())
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        fix_file(f, _FixMode.ADD_AWAIT_RESULT)
        fixed = f.read_text()
        assert 'print("before")' in fixed
        assert 'print("after")' in fixed
        assert "asyncio.run(main())" in fixed
        assert "await_result=True" in fixed
        compile(fixed, str(f), "exec")

    def test_aquery_fix(self, tmp_path):
        src = _dedent("""\
            from mellea.stdlib.functional import aquery
            async def main():
                r, c = await aquery(obj, "q", ctx, be)
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        fix_file(f, _FixMode.ADD_AWAIT_RESULT)
        fixed = f.read_text()
        assert "await_result=True" in fixed
        compile(fixed, str(f), "exec")

    def test_trailing_comma_single_line(self, tmp_path):
        src = _dedent("""\
            from mellea.stdlib.functional import aact
            async def main():
                r, c = await aact(a, ctx, be, strategy=None,)
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        fix_file(f, _FixMode.ADD_AWAIT_RESULT)
        fixed = f.read_text()
        assert "await_result=True" in fixed
        compile(fixed, str(f), "exec")

    def test_explicit_await_result_false_is_fixed(self, tmp_path):
        """If user explicitly set await_result=False, we should fix it."""
        src = _dedent("""\
            from mellea.stdlib.functional import aact
            async def main():
                r, c = await aact(a, ctx, be, strategy=None, await_result=False)
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        fix_file(f, _FixMode.ADD_AWAIT_RESULT)
        fixed = f.read_text()
        assert "await_result=True" in fixed
        assert fixed.count("await_result") == 1
        compile(fixed, str(f), "exec")

    def test_multiline_trailing_comma(self, tmp_path):
        src = _dedent("""\
            from mellea.stdlib.functional import aact
            async def main():
                result, ctx = await aact(
                    action,
                    context,
                    backend,
                    strategy=None,
                )
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        fix_file(f, _FixMode.ADD_AWAIT_RESULT)
        fixed = f.read_text()
        assert "await_result=True" in fixed
        compile(fixed, str(f), "exec")


# ===================================================================
# add-stream-loop fix tests
# ===================================================================


class TestAddStreamLoop:
    def test_session_api_stream_loop(self, tmp_path):
        src = _dedent("""\
            from mellea import MelleaSession
            async def main():
                result = await session.aact(action, strategy=None)
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        fix_file(f, _FixMode.ADD_STREAM_LOOP)
        fixed = f.read_text()
        assert "while not result.is_computed():" in fixed
        assert "await result.astream()" in fixed
        compile(fixed, str(f), "exec")

    def test_functional_api_stream_loop(self, tmp_path):
        src = _dedent("""\
            from mellea.stdlib.functional import aact
            async def main():
                result, ctx = await aact(action, context, backend, strategy=None)
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        fix_file(f, _FixMode.ADD_STREAM_LOOP)
        fixed = f.read_text()
        assert "while not result.is_computed():" in fixed
        assert "await result.astream()" in fixed
        compile(fixed, str(f), "exec")

    def test_correct_indentation(self, tmp_path):
        src = _dedent("""\
            from mellea.stdlib.session import start_session
            async def main():
                if True:
                    result = await session.aact(action, strategy=None)
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        fix_file(f, _FixMode.ADD_STREAM_LOOP)
        fixed = f.read_text()
        lines = fixed.splitlines()
        loop_line = next(line for line in lines if "while not" in line)
        assert loop_line.startswith("        ")  # 8 spaces (2 levels)
        compile(fixed, str(f), "exec")

    def test_skips_unassigned(self, tmp_path):
        src = _dedent("""\
            from mellea import MelleaSession
            async def main():
                await session.aact(action, strategy=None)
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        locs = fix_file(f, _FixMode.ADD_STREAM_LOOP)
        fixed = f.read_text()
        assert "while not" not in fixed
        assert len(locs) == 1

    def test_different_variable_names(self, tmp_path):
        src = _dedent("""\
            from mellea import MelleaSession
            async def main():
                output = await session.aact(action, strategy=None)
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        fix_file(f, _FixMode.ADD_STREAM_LOOP)
        fixed = f.read_text()
        assert "while not output.is_computed():" in fixed
        assert "await output.astream()" in fixed

    def test_aquery_stream_loop(self, tmp_path):
        src = _dedent("""\
            from mellea import MelleaSession
            async def main():
                answer = await session.aquery(obj, "q")
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        fix_file(f, _FixMode.ADD_STREAM_LOOP)
        fixed = f.read_text()
        assert "while not answer.is_computed():" in fixed
        compile(fixed, str(f), "exec")


# ===================================================================
# Dry run tests
# ===================================================================


class TestDryRun:
    def test_dry_run_no_file_modification(self, tmp_path):
        src = _dedent("""\
            from mellea.stdlib.functional import aact
            async def main():
                r, c = await aact(a, ctx, be, strategy=None)
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        locs = fix_file(f, _FixMode.ADD_AWAIT_RESULT, dry_run=True)
        assert len(locs) == 1
        assert f.read_text() == src

    def test_dry_run_correct_location_reporting(self, tmp_path):
        src = _dedent("""\
            from mellea.stdlib.functional import aact
            async def main():
                r, c = await aact(a, ctx, be, strategy=None)
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        locs = fix_file(f, _FixMode.ADD_AWAIT_RESULT, dry_run=True)
        assert locs[0].filepath == f
        assert locs[0].line == 3
        assert locs[0].function_name == "aact"


# ===================================================================
# Traversal tests
# ===================================================================


class TestTraversal:
    def test_single_file(self, tmp_path):
        src = _dedent("""\
            from mellea.stdlib.functional import aact
            async def main():
                r, c = await aact(a, ctx, be, strategy=None)
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        result = fix_path(f, _FixMode.ADD_AWAIT_RESULT, dry_run=True)
        assert isinstance(result, FixResult)
        assert result.total_fixes == 1
        assert result.files_affected == 1

    def test_recursive_directory(self, tmp_path):
        src = _dedent("""\
            from mellea.stdlib.functional import aact
            async def f():
                r, c = await aact(a, ctx, be, strategy=None)
        """)
        (tmp_path / "sub").mkdir()
        (tmp_path / "a.py").write_text(src)
        (tmp_path / "sub" / "b.py").write_text(src)
        result = fix_path(tmp_path, _FixMode.ADD_AWAIT_RESULT, dry_run=True)
        assert result.total_fixes == 2
        assert result.files_affected == 2

    def test_skips_non_python(self, tmp_path):
        (tmp_path / "data.txt").write_text("not python")
        (tmp_path / "config.json").write_text("{}")
        result = fix_path(tmp_path, _FixMode.ADD_AWAIT_RESULT, dry_run=True)
        assert result.total_fixes == 0

    def test_skips_pycache(self, tmp_path):
        src = _dedent("""\
            from mellea.stdlib.functional import aact
            async def f():
                r, c = await aact(a, ctx, be, strategy=None)
        """)
        cache_dir = tmp_path / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "mod.py").write_text(src)
        result = fix_path(tmp_path, _FixMode.ADD_AWAIT_RESULT, dry_run=True)
        assert result.total_fixes == 0

    def test_handles_syntax_errors(self, tmp_path):
        (tmp_path / "bad.py").write_text("def broken(:\n")
        src = _dedent("""\
            from mellea.stdlib.functional import aact
            async def f():
                r, c = await aact(a, ctx, be, strategy=None)
        """)
        (tmp_path / "good.py").write_text(src)
        result = fix_path(tmp_path, _FixMode.ADD_AWAIT_RESULT, dry_run=True)
        assert result.total_fixes == 1
        assert result.files_affected == 1

    def test_skips_venv(self, tmp_path):
        src = _dedent("""\
            from mellea.stdlib.functional import aact
            async def f():
                r, c = await aact(a, ctx, be, strategy=None)
        """)
        venv_dir = tmp_path / ".venv"
        venv_dir.mkdir()
        (venv_dir / "mod.py").write_text(src)
        result = fix_path(tmp_path, _FixMode.ADD_AWAIT_RESULT, dry_run=True)
        assert result.total_fixes == 0


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:
    def test_await_result_false_detected(self):
        """Explicit await_result=False with strategy=None should be detected."""
        src = _dedent("""\
            from mellea.stdlib.functional import aact
            async def main():
                r, c = await aact(a, ctx, be, strategy=None, await_result=False)
        """)
        locs = find_fixable_calls(src, Path("test.py"))
        assert len(locs) == 1

    def test_no_calls_returns_empty(self):
        src = _dedent("""\
            def hello():
                print("no async here")
        """)
        locs = find_fixable_calls(src, Path("test.py"))
        assert len(locs) == 0

    def test_empty_file(self):
        locs = find_fixable_calls("", Path("test.py"))
        assert len(locs) == 0

    def test_fix_result_aggregation(self, tmp_path):
        src1 = _dedent("""\
            from mellea.stdlib.functional import aact
            async def f():
                r, c = await aact(a, ctx, be, strategy=None)
        """)
        src2 = _dedent("""\
            from mellea.stdlib.functional import ainstruct, aquery
            async def g():
                r, c = await ainstruct("x", ctx, be, strategy=None)
                r2, c2 = await aquery(obj, "q", ctx, be)
        """)
        (tmp_path / "a.py").write_text(src1)
        (tmp_path / "b.py").write_text(src2)
        result = fix_path(tmp_path, _FixMode.ADD_AWAIT_RESULT, dry_run=True)
        assert result.total_fixes == 3
        assert result.files_affected == 2
        assert len(result.locations) == 3

    def test_add_await_result_idempotent_functional(self, tmp_path):
        """Running fix twice should not double-apply await_result=True."""
        src = _dedent("""\
            from mellea.stdlib.functional import aact
            async def main():
                r, c = await aact(a, ctx, be, strategy=None)
        """)
        expected = _dedent("""\
            from mellea.stdlib.functional import aact
            async def main():
                r, c = await aact(a, ctx, be, strategy=None, await_result=True)
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        fix_file(f, _FixMode.ADD_AWAIT_RESULT)
        assert f.read_text() == expected
        # Second run should be a no-op
        locs = fix_file(f, _FixMode.ADD_AWAIT_RESULT)
        assert f.read_text() == expected
        assert len(locs) == 0

    def test_add_await_result_idempotent_session(self, tmp_path):
        """Running fix twice on session calls should not double-apply."""
        src = _dedent("""\
            from mellea import MelleaSession
            async def main():
                r = await session.aact(action, strategy=None)
        """)
        expected = _dedent("""\
            from mellea import MelleaSession
            async def main():
                r = await session.aact(action, strategy=None, await_result=True)
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        fix_file(f, _FixMode.ADD_AWAIT_RESULT)
        assert f.read_text() == expected
        locs = fix_file(f, _FixMode.ADD_AWAIT_RESULT)
        assert f.read_text() == expected
        assert len(locs) == 0

    def test_add_await_result_idempotent_multiline(self, tmp_path):
        """Multiline calls should also be idempotent."""
        src = _dedent("""\
            from mellea.stdlib.functional import aact
            async def main():
                r, c = await aact(
                    action,
                    ctx,
                    be,
                    strategy=None,
                )
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        fix_file(f, _FixMode.ADD_AWAIT_RESULT)
        first_pass = f.read_text()
        assert first_pass.count("await_result=True") == 1
        locs = fix_file(f, _FixMode.ADD_AWAIT_RESULT)
        assert f.read_text() == first_pass
        assert len(locs) == 0

    def test_mixed_functional_and_session_imports(self):
        """File with both functional and session imports detects both styles."""
        src = _dedent("""\
            from mellea.stdlib.functional import aact
            from mellea import MelleaSession
            async def main():
                r1, c1 = await aact(a, ctx, be, strategy=None)
                r2 = await session.aact(action, strategy=None)
        """)
        locs = find_fixable_calls(src, Path("test.py"))
        assert len(locs) == 2
        styles = {loc.call_style for loc in locs}
        assert styles == {"functional", "session"}


# ===================================================================
# Exact output tests — full before/after file content matching
# ===================================================================


class TestExactOutputAddAwaitResult:
    """Verify the exact file content produced by add-await-result mode."""

    def test_single_line_bare_aact(self, tmp_path):
        src = _dedent("""\
            from mellea.stdlib.functional import aact
            async def main():
                result, ctx = await aact(action, context, backend, strategy=None)
        """)
        expected = _dedent("""\
            from mellea.stdlib.functional import aact
            async def main():
                result, ctx = await aact(action, context, backend, strategy=None, await_result=True)
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        fix_file(f, _FixMode.ADD_AWAIT_RESULT)
        assert f.read_text() == expected

    def test_single_line_session_aact(self, tmp_path):
        src = _dedent("""\
            from mellea import MelleaSession
            async def main():
                result = await session.aact(action, strategy=None)
        """)
        expected = _dedent("""\
            from mellea import MelleaSession
            async def main():
                result = await session.aact(action, strategy=None, await_result=True)
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        fix_file(f, _FixMode.ADD_AWAIT_RESULT)
        assert f.read_text() == expected

    def test_single_line_aquery(self, tmp_path):
        src = _dedent("""\
            from mellea.stdlib.functional import aquery
            async def main():
                r, c = await aquery(obj, "q", ctx, be)
        """)
        expected = _dedent("""\
            from mellea.stdlib.functional import aquery
            async def main():
                r, c = await aquery(obj, "q", ctx, be, await_result=True)
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        fix_file(f, _FixMode.ADD_AWAIT_RESULT)
        assert f.read_text() == expected

    def test_single_line_module_qualified(self, tmp_path):
        src = _dedent("""\
            import mellea.stdlib.functional as mfuncs
            async def main():
                r, c = await mfuncs.ainstruct("desc", ctx, be, strategy=None)
        """)
        expected = _dedent("""\
            import mellea.stdlib.functional as mfuncs
            async def main():
                r, c = await mfuncs.ainstruct("desc", ctx, be, strategy=None, await_result=True)
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        fix_file(f, _FixMode.ADD_AWAIT_RESULT)
        assert f.read_text() == expected

    def test_single_line_trailing_comma(self, tmp_path):
        src = _dedent("""\
            from mellea.stdlib.functional import aact
            async def main():
                r, c = await aact(a, ctx, be, strategy=None,)
        """)
        expected = _dedent("""\
            from mellea.stdlib.functional import aact
            async def main():
                r, c = await aact(a, ctx, be, strategy=None, await_result=True)
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        fix_file(f, _FixMode.ADD_AWAIT_RESULT)
        assert f.read_text() == expected

    def test_multiline_with_trailing_comma(self, tmp_path):
        src = _dedent("""\
            from mellea.stdlib.functional import aact
            async def main():
                result, ctx = await aact(
                    action,
                    context,
                    backend,
                    strategy=None,
                )
        """)
        expected = _dedent("""\
            from mellea.stdlib.functional import aact
            async def main():
                result, ctx = await aact(
                    action,
                    context,
                    backend,
                    strategy=None,
                    await_result=True,
                )
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        fix_file(f, _FixMode.ADD_AWAIT_RESULT)
        assert f.read_text() == expected

    def test_multiline_without_trailing_comma(self, tmp_path):
        src = _dedent("""\
            from mellea.stdlib.functional import aact
            async def main():
                result, ctx = await aact(
                    action,
                    context,
                    backend,
                    strategy=None
                )
        """)
        expected = _dedent("""\
            from mellea.stdlib.functional import aact
            async def main():
                result, ctx = await aact(
                    action,
                    context,
                    backend,
                    strategy=None,
                    await_result=True,
                )
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        fix_file(f, _FixMode.ADD_AWAIT_RESULT)
        assert f.read_text() == expected

    def test_replace_await_result_false(self, tmp_path):
        src = _dedent("""\
            from mellea.stdlib.functional import aact
            async def main():
                r, c = await aact(a, ctx, be, strategy=None, await_result=False)
        """)
        expected = _dedent("""\
            from mellea.stdlib.functional import aact
            async def main():
                r, c = await aact(a, ctx, be, strategy=None, await_result=True)
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        fix_file(f, _FixMode.ADD_AWAIT_RESULT)
        assert f.read_text() == expected

    def test_multiple_calls_in_file(self, tmp_path):
        src = _dedent("""\
            import asyncio
            from mellea.stdlib.functional import aact, ainstruct

            async def main():
                print("before")
                r1, c1 = await aact(a, ctx, be, strategy=None)
                print("middle")
                r2, c2 = await ainstruct("x", ctx, be, strategy=None)
                print("after")

            if __name__ == "__main__":
                asyncio.run(main())
        """)
        expected = _dedent("""\
            import asyncio
            from mellea.stdlib.functional import aact, ainstruct

            async def main():
                print("before")
                r1, c1 = await aact(a, ctx, be, strategy=None, await_result=True)
                print("middle")
                r2, c2 = await ainstruct("x", ctx, be, strategy=None, await_result=True)
                print("after")

            if __name__ == "__main__":
                asyncio.run(main())
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        fix_file(f, _FixMode.ADD_AWAIT_RESULT)
        assert f.read_text() == expected

    def test_mixed_fixable_and_skipped(self, tmp_path):
        src = _dedent("""\
            from mellea.stdlib.functional import aact, aquery, achat
            async def main():
                r1, c1 = await aact(a, ctx, be, strategy=None)
                r2, c2 = await aact(a, ctx, be)
                r3, c3 = await aact(a, ctx, be, strategy=None, await_result=True)
                msg, c4 = await achat("hi", ctx, be)
                r5, c5 = await aquery(obj, "q", ctx, be)
        """)
        expected = _dedent("""\
            from mellea.stdlib.functional import aact, aquery, achat
            async def main():
                r1, c1 = await aact(a, ctx, be, strategy=None, await_result=True)
                r2, c2 = await aact(a, ctx, be)
                r3, c3 = await aact(a, ctx, be, strategy=None, await_result=True)
                msg, c4 = await achat("hi", ctx, be)
                r5, c5 = await aquery(obj, "q", ctx, be, await_result=True)
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        fix_file(f, _FixMode.ADD_AWAIT_RESULT)
        assert f.read_text() == expected

    def test_session_aquery_no_args(self, tmp_path):
        src = _dedent("""\
            from mellea.stdlib.session import start_session
            async def run():
                answer = await session.aquery(obj, "what is this?")
        """)
        expected = _dedent("""\
            from mellea.stdlib.session import start_session
            async def run():
                answer = await session.aquery(obj, "what is this?", await_result=True)
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        fix_file(f, _FixMode.ADD_AWAIT_RESULT)
        assert f.read_text() == expected

    def test_no_assignment_bare_await(self, tmp_path):
        src = _dedent("""\
            from mellea import MelleaSession
            async def fire_and_forget():
                await session.aact(action, strategy=None)
        """)
        expected = _dedent("""\
            from mellea import MelleaSession
            async def fire_and_forget():
                await session.aact(action, strategy=None, await_result=True)
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        fix_file(f, _FixMode.ADD_AWAIT_RESULT)
        assert f.read_text() == expected

    def test_custom_alias_functional_module(self, tmp_path):
        """import mellea.stdlib.functional as f — custom alias."""
        src = _dedent("""\
            import mellea.stdlib.functional as f
            async def main():
                r, c = await f.aact(action, ctx, be, strategy=None)
        """)
        expected = _dedent("""\
            import mellea.stdlib.functional as f
            async def main():
                r, c = await f.aact(action, ctx, be, strategy=None, await_result=True)
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        fix_file(f, _FixMode.ADD_AWAIT_RESULT)
        assert f.read_text() == expected

    def test_from_stdlib_import_functional_alias(self, tmp_path):
        """from mellea.stdlib import functional as fn."""
        src = _dedent("""\
            from mellea.stdlib import functional as fn
            async def main():
                r, c = await fn.aact(action, ctx, be, strategy=None)
        """)
        expected = _dedent("""\
            from mellea.stdlib import functional as fn
            async def main():
                r, c = await fn.aact(action, ctx, be, strategy=None, await_result=True)
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        fix_file(f, _FixMode.ADD_AWAIT_RESULT)
        assert f.read_text() == expected


class TestExactOutputAddStreamLoop:
    """Verify the exact file content produced by add-stream-loop mode."""

    def test_session_simple_assign(self, tmp_path):
        src = _dedent("""\
            from mellea import MelleaSession
            async def main():
                result = await session.aact(action, strategy=None)
        """)
        expected = _dedent("""\
            from mellea import MelleaSession
            async def main():
                result = await session.aact(action, strategy=None)
                while not result.is_computed():
                    await result.astream()
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        fix_file(f, _FixMode.ADD_STREAM_LOOP)
        assert f.read_text() == expected

    def test_functional_tuple_unpack(self, tmp_path):
        src = _dedent("""\
            from mellea.stdlib.functional import aact
            async def main():
                result, ctx = await aact(action, context, backend, strategy=None)
        """)
        expected = _dedent("""\
            from mellea.stdlib.functional import aact
            async def main():
                result, ctx = await aact(action, context, backend, strategy=None)
                while not result.is_computed():
                    await result.astream()
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        fix_file(f, _FixMode.ADD_STREAM_LOOP)
        assert f.read_text() == expected

    def test_nested_indentation(self, tmp_path):
        src = _dedent("""\
            from mellea.stdlib.session import start_session
            async def main():
                if True:
                    result = await session.aact(action, strategy=None)
        """)
        expected = _dedent("""\
            from mellea.stdlib.session import start_session
            async def main():
                if True:
                    result = await session.aact(action, strategy=None)
                    while not result.is_computed():
                        await result.astream()
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        fix_file(f, _FixMode.ADD_STREAM_LOOP)
        assert f.read_text() == expected

    def test_no_assignment_no_loop_inserted(self, tmp_path):
        src = _dedent("""\
            from mellea import MelleaSession
            async def main():
                await session.aact(action, strategy=None)
        """)
        expected = _dedent("""\
            from mellea import MelleaSession
            async def main():
                await session.aact(action, strategy=None)
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        fix_file(f, _FixMode.ADD_STREAM_LOOP)
        assert f.read_text() == expected

    def test_multiple_calls_with_code_between(self, tmp_path):
        src = _dedent("""\
            from mellea import MelleaSession
            async def main():
                r1 = await session.aact(action1, strategy=None)
                print("processing")
                r2 = await session.aact(action2, strategy=None)
        """)
        expected = _dedent("""\
            from mellea import MelleaSession
            async def main():
                r1 = await session.aact(action1, strategy=None)
                while not r1.is_computed():
                    await r1.astream()
                print("processing")
                r2 = await session.aact(action2, strategy=None)
                while not r2.is_computed():
                    await r2.astream()
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        fix_file(f, _FixMode.ADD_STREAM_LOOP)
        assert f.read_text() == expected

    def test_aquery_session_stream_loop(self, tmp_path):
        src = _dedent("""\
            from mellea import MelleaSession
            async def main():
                answer = await session.aquery(obj, "question")
        """)
        expected = _dedent("""\
            from mellea import MelleaSession
            async def main():
                answer = await session.aquery(obj, "question")
                while not answer.is_computed():
                    await answer.astream()
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        fix_file(f, _FixMode.ADD_STREAM_LOOP)
        assert f.read_text() == expected

    def test_aquery_functional_stream_loop(self, tmp_path):
        src = _dedent("""\
            from mellea.stdlib.functional import aquery
            async def main():
                answer, ctx = await aquery(obj, "q", context, backend)
        """)
        expected = _dedent("""\
            from mellea.stdlib.functional import aquery
            async def main():
                answer, ctx = await aquery(obj, "q", context, backend)
                while not answer.is_computed():
                    await answer.astream()
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        fix_file(f, _FixMode.ADD_STREAM_LOOP)
        assert f.read_text() == expected


# ===================================================================
# Idempotency and cross-mode awareness tests
# ===================================================================


class TestStreamLoopIdempotency:
    """Running add-stream-loop twice should not insert a duplicate loop."""

    def test_idempotent_session(self, tmp_path):
        src = _dedent("""\
            from mellea import MelleaSession
            async def main():
                result = await session.aact(action, strategy=None)
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        locs1 = fix_file(f, _FixMode.ADD_STREAM_LOOP)
        assert len(locs1) == 1
        first_pass = f.read_text()
        assert first_pass.count("while not result.is_computed()") == 1

        # Second run: no new locations, file unchanged
        locs2 = fix_file(f, _FixMode.ADD_STREAM_LOOP)
        assert len(locs2) == 0
        assert f.read_text() == first_pass

    def test_idempotent_functional(self, tmp_path):
        src = _dedent("""\
            from mellea.stdlib.functional import aact
            async def main():
                result, ctx = await aact(action, context, backend, strategy=None)
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        fix_file(f, _FixMode.ADD_STREAM_LOOP)
        first_pass = f.read_text()
        assert first_pass.count("while not result.is_computed()") == 1

        locs2 = fix_file(f, _FixMode.ADD_STREAM_LOOP)
        assert len(locs2) == 0
        assert f.read_text() == first_pass

    def test_idempotent_multiple_calls(self, tmp_path):
        src = _dedent("""\
            from mellea import MelleaSession
            async def main():
                r1 = await session.aact(action1, strategy=None)
                r2 = await session.aact(action2, strategy=None)
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        fix_file(f, _FixMode.ADD_STREAM_LOOP)
        first_pass = f.read_text()
        assert first_pass.count("while not") == 2

        locs2 = fix_file(f, _FixMode.ADD_STREAM_LOOP)
        assert len(locs2) == 0
        assert f.read_text() == first_pass


class TestCrossModeAwareness:
    """Running one mode after the other should not double-fix."""

    def test_stream_loop_then_await_result(self, tmp_path):
        """After add-stream-loop, add-await-result should find 0 locations."""
        src = _dedent("""\
            from mellea import MelleaSession
            async def main():
                result = await session.aact(action, strategy=None)
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        fix_file(f, _FixMode.ADD_STREAM_LOOP)
        after_loop = f.read_text()

        locs = fix_file(f, _FixMode.ADD_AWAIT_RESULT)
        assert len(locs) == 0
        assert f.read_text() == after_loop

    def test_await_result_then_stream_loop(self, tmp_path):
        """After add-await-result, add-stream-loop should find 0 locations."""
        src = _dedent("""\
            from mellea.stdlib.functional import aact
            async def main():
                r, c = await aact(a, ctx, be, strategy=None)
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        fix_file(f, _FixMode.ADD_AWAIT_RESULT)
        after_await = f.read_text()
        assert "await_result=True" in after_await

        locs = fix_file(f, _FixMode.ADD_STREAM_LOOP)
        assert len(locs) == 0
        assert f.read_text() == after_await

    def test_cross_mode_session_aquery(self, tmp_path):
        """Cross-mode for session aquery (no strategy param)."""
        src = _dedent("""\
            from mellea import MelleaSession
            async def main():
                answer = await session.aquery(obj, "question")
        """)
        f = tmp_path / "test.py"
        f.write_text(src)
        fix_file(f, _FixMode.ADD_STREAM_LOOP)
        after_loop = f.read_text()

        locs = fix_file(f, _FixMode.ADD_AWAIT_RESULT)
        assert len(locs) == 0
        assert f.read_text() == after_loop


class TestUserWrittenPatternDetection:
    """User-written consumption patterns should be detected and skipped."""

    def test_existing_while_loop(self):
        """while not r.is_computed() already present → not detected."""
        src = _dedent("""\
            from mellea import MelleaSession
            async def main():
                r = await session.aact(action, strategy=None)
                while not r.is_computed():
                    await r.astream()
        """)
        locs = find_fixable_calls(src, Path("test.py"))
        assert len(locs) == 0

    def test_existing_await_avalue(self):
        """await r.avalue() already present → not detected."""
        src = _dedent("""\
            from mellea import MelleaSession
            async def main():
                r = await session.aact(action, strategy=None)
                await r.avalue()
        """)
        locs = find_fixable_calls(src, Path("test.py"))
        assert len(locs) == 0

    def test_existing_await_astream(self):
        """Bare await r.astream() already present → not detected."""
        src = _dedent("""\
            from mellea import MelleaSession
            async def main():
                r = await session.aact(action, strategy=None)
                await r.astream()
        """)
        locs = find_fixable_calls(src, Path("test.py"))
        assert len(locs) == 0

    def test_one_handled_one_not(self):
        """Only the unhandled call should be detected."""
        src = _dedent("""\
            from mellea import MelleaSession
            async def main():
                r1 = await session.aact(action1, strategy=None)
                while not r1.is_computed():
                    await r1.astream()
                r2 = await session.aact(action2, strategy=None)
        """)
        locs = find_fixable_calls(src, Path("test.py"))
        assert len(locs) == 1
        assert locs[0].target_variable == "r2"

    def test_avalue_inside_nested_if(self):
        """await r.avalue() inside a nested if block → still detected as consumption."""
        src = _dedent("""\
            from mellea import MelleaSession
            async def main():
                r = await session.aact(action, strategy=None)
                if some_condition:
                    await r.avalue()
        """)
        locs = find_fixable_calls(src, Path("test.py"))
        assert len(locs) == 0

    def test_astream_inside_try_block(self):
        """await r.astream() inside a try block → still detected as consumption."""
        src = _dedent("""\
            from mellea import MelleaSession
            async def main():
                r = await session.aact(action, strategy=None)
                try:
                    await r.astream()
                except Exception:
                    pass
        """)
        locs = find_fixable_calls(src, Path("test.py"))
        assert len(locs) == 0

    def test_assigned_avalue(self):
        """val = await r.avalue() — assigned form should also be detected."""
        src = _dedent("""\
            from mellea import MelleaSession
            async def main():
                r = await session.aact(action, strategy=None)
                val = await r.avalue()
        """)
        locs = find_fixable_calls(src, Path("test.py"))
        assert len(locs) == 0

    def test_from_stdlib_import_session_detected(self):
        """``from mellea.stdlib import session`` should enable session detection."""
        src = _dedent("""\
            from mellea.stdlib import session
            async def main():
                s = session.start_session()
                out = await s.aact(action, strategy=None)
        """)
        locs = find_fixable_calls(src, Path("test.py"))
        assert len(locs) == 1
        assert locs[0].function_name == "aact"
        assert locs[0].call_style == "session"

    def test_bare_import_mellea_detected(self):
        """``import mellea`` + session method call should be detected."""
        src = _dedent("""\
            import mellea
            m = mellea.start_session()
            async def main():
                out = await m.ainstruct("do stuff", ctx, be, strategy=None)
        """)
        locs = find_fixable_calls(src, Path("test.py"))
        assert len(locs) == 1
        assert locs[0].function_name == "ainstruct"
        assert locs[0].call_style == "session"

    def test_functional_with_existing_loop(self):
        """Functional style with existing loop → not detected."""
        src = _dedent("""\
            from mellea.stdlib.functional import aact
            async def main():
                result, ctx = await aact(action, context, backend, strategy=None)
                while not result.is_computed():
                    await result.astream()
        """)
        locs = find_fixable_calls(src, Path("test.py"))
        assert len(locs) == 0


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
