"""Tests for the ``m fix genslots`` CLI tool."""

import textwrap
from pathlib import Path

from cli.fix.genstub_fixer import (
    GenStubFixLocation,
    find_genslot_refs,
    fix_genslot_file,
    fix_genslot_path,
)


def _dedent(source: str) -> str:
    return textwrap.dedent(source).lstrip("\n")


def _desc_list(locs: list[GenStubFixLocation]) -> list[str]:
    return [loc.description for loc in locs]


# ===================================================================
# Detection — fully-qualified imports
# ===================================================================


def test_detects_from_genslot_import():
    src = _dedent("""\
        from mellea.stdlib.components.genslot import generative
    """)
    locs = find_genslot_refs(src, Path("test.py"))
    assert len(locs) == 1
    assert "genslot" in locs[0].description
    assert "genstub" in locs[0].description


def test_detects_import_genslot_bare():
    src = _dedent("""\
        import mellea.stdlib.components.genslot
    """)
    locs = find_genslot_refs(src, Path("test.py"))
    assert len(locs) == 1


def test_detects_import_genslot_as_alias():
    src = _dedent("""\
        import mellea.stdlib.components.genslot as gs
    """)
    locs = find_genslot_refs(src, Path("test.py"))
    assert len(locs) == 1


# ===================================================================
# Detection — parent-level imports
# ===================================================================


def test_detects_from_components_import_genslot():
    src = _dedent("""\
        from mellea.stdlib.components import genslot
    """)
    locs = find_genslot_refs(src, Path("test.py"))
    assert len(locs) == 1
    assert "import genslot" in locs[0].description


def test_detects_from_components_import_genslot_as():
    src = _dedent("""\
        from mellea.stdlib.components import genslot as gs
    """)
    locs = find_genslot_refs(src, Path("test.py"))
    assert len(locs) == 1


# ===================================================================
# Detection — relative imports
# ===================================================================


def test_detects_relative_import_dot_genslot():
    src = _dedent("""\
        from .genslot import PreconditionException
    """)
    locs = find_genslot_refs(src, Path("test.py"))
    assert len(locs) == 1
    assert ".genslot" in locs[0].description


def test_detects_relative_import_dotdot_genslot():
    src = _dedent("""\
        from ..components.genslot import generative
    """)
    locs = find_genslot_refs(src, Path("test.py"))
    assert len(locs) == 1


# ===================================================================
# Detection — old class names
# ===================================================================


def test_detects_generative_slot_class():
    src = _dedent("""\
        from mellea.stdlib.components.genstub import GenerativeSlot
    """)
    locs = find_genslot_refs(src, Path("test.py"))
    assert len(locs) == 1
    assert "GenerativeSlot" in locs[0].description
    assert "GenerativeStub" in locs[0].description


def test_detects_sync_generative_slot():
    src = _dedent("""\
        from mellea.stdlib.components.genstub import SyncGenerativeSlot
    """)
    locs = find_genslot_refs(src, Path("test.py"))
    assert len(locs) == 1
    assert "SyncGenerativeSlot" in locs[0].description


def test_detects_async_generative_slot():
    src = _dedent("""\
        from mellea.stdlib.components.genstub import AsyncGenerativeSlot
    """)
    locs = find_genslot_refs(src, Path("test.py"))
    assert len(locs) == 1
    assert "AsyncGenerativeSlot" in locs[0].description


def test_detects_module_and_class_on_same_line():
    src = _dedent("""\
        from mellea.stdlib.components.genslot import GenerativeSlot
    """)
    locs = find_genslot_refs(src, Path("test.py"))
    # Module path + class name = 2 detections on the same line.
    assert len(locs) == 2


def test_detects_class_in_type_annotation():
    src = _dedent("""\
        def foo(stub: GenerativeSlot) -> None:
            pass
    """)
    locs = find_genslot_refs(src, Path("test.py"))
    assert len(locs) == 1


def test_detects_multiple_classes_on_one_import():
    src = _dedent("""\
        from mellea.stdlib.components.genslot import GenerativeSlot, SyncGenerativeSlot
    """)
    locs = find_genslot_refs(src, Path("test.py"))
    # genslot module + GenerativeSlot + SyncGenerativeSlot = 3
    assert len(locs) == 3


# ===================================================================
# No false positives
# ===================================================================


def test_no_false_positive_on_new_names():
    src = _dedent("""\
        from mellea.stdlib.components.genstub import GenerativeStub
        from mellea.stdlib.components.genstub import SyncGenerativeStub
        from mellea.stdlib.components.genstub import AsyncGenerativeStub
    """)
    locs = find_genslot_refs(src, Path("test.py"))
    assert len(locs) == 0


def test_no_false_positive_on_substring():
    src = _dedent("""\
        # MyGenerativeSlotHelper should not match (no word boundary)
        x = "GenerativeSlotting"
    """)
    locs = find_genslot_refs(src, Path("test.py"))
    assert len(locs) == 0


def test_no_false_positive_genslot_in_unrelated_path():
    src = _dedent("""\
        from mypackage.genslot import something
    """)
    locs = find_genslot_refs(src, Path("test.py"))
    # The relative import regex matches `.genslot` but this is not a dotted
    # relative import — `from mypackage.genslot` has no leading dot so only
    # the `\bgenslot\b` in the relative regex can fire if the prefix matches.
    # Actually `from mypackage.genslot` does NOT start with `from .` so neither
    # the fully-qualified nor relative pattern fires.
    assert len(locs) == 0


# ===================================================================
# Transformation
# ===================================================================


def test_rewrites_module_path(tmp_path: Path):
    src = "from mellea.stdlib.components.genslot import generative\n"
    f = tmp_path / "example.py"
    f.write_text(src)

    fix_genslot_file(f)

    assert "genstub" in f.read_text()
    assert "genslot" not in f.read_text()


def test_rewrites_class_names(tmp_path: Path):
    src = _dedent("""\
        from mellea.stdlib.components.genstub import GenerativeSlot, SyncGenerativeSlot
        x: GenerativeSlot = get_stub()
    """)
    f = tmp_path / "example.py"
    f.write_text(src)

    fix_genslot_file(f)

    result = f.read_text()
    assert "GenerativeStub" in result
    assert "SyncGenerativeStub" in result
    assert "GenerativeSlot" not in result


def test_rewrites_async_class(tmp_path: Path):
    src = "isinstance(x, AsyncGenerativeSlot)\n"
    f = tmp_path / "example.py"
    f.write_text(src)

    fix_genslot_file(f)

    assert "AsyncGenerativeStub" in f.read_text()
    assert "AsyncGenerativeSlot" not in f.read_text()


def test_rewrites_combined_module_and_class(tmp_path: Path):
    src = "from mellea.stdlib.components.genslot import GenerativeSlot\n"
    f = tmp_path / "example.py"
    f.write_text(src)

    fix_genslot_file(f)

    assert (
        f.read_text() == "from mellea.stdlib.components.genstub import GenerativeStub\n"
    )


def test_rewrites_from_components_import_genslot(tmp_path: Path):
    src = "from mellea.stdlib.components import genslot\n"
    f = tmp_path / "example.py"
    f.write_text(src)

    fix_genslot_file(f)

    assert f.read_text() == "from mellea.stdlib.components import genstub\n"


def test_rewrites_from_components_import_genslot_as(tmp_path: Path):
    src = "from mellea.stdlib.components import genslot as gs\n"
    f = tmp_path / "example.py"
    f.write_text(src)

    fix_genslot_file(f)

    assert f.read_text() == "from mellea.stdlib.components import genstub as gs\n"


def test_rewrites_relative_import(tmp_path: Path):
    src = "from .genslot import PreconditionException\n"
    f = tmp_path / "example.py"
    f.write_text(src)

    fix_genslot_file(f)

    assert f.read_text() == "from .genstub import PreconditionException\n"


def test_rewrites_relative_dotdot_import(tmp_path: Path):
    src = "from ..components.genslot import generative\n"
    f = tmp_path / "example.py"
    f.write_text(src)

    fix_genslot_file(f)

    assert f.read_text() == "from ..components.genstub import generative\n"


def test_dry_run_does_not_modify(tmp_path: Path):
    src = "from mellea.stdlib.components.genslot import GenerativeSlot\n"
    f = tmp_path / "example.py"
    f.write_text(src)

    locs = fix_genslot_file(f, dry_run=True)

    assert len(locs) > 0
    assert f.read_text() == src


def test_idempotent(tmp_path: Path):
    src = "from mellea.stdlib.components.genslot import GenerativeSlot\n"
    f = tmp_path / "example.py"
    f.write_text(src)

    fix_genslot_file(f)
    first_pass = f.read_text()

    locs = fix_genslot_file(f)
    assert locs == []
    assert f.read_text() == first_pass


# ===================================================================
# Directory traversal
# ===================================================================


def test_single_file(tmp_path: Path):
    f = tmp_path / "example.py"
    f.write_text("from mellea.stdlib.components.genslot import generative\n")

    result = fix_genslot_path(f)

    assert result.total_fixes == 1
    assert result.files_affected == 1


def test_directory_recursive(tmp_path: Path):
    (tmp_path / "a.py").write_text(
        "from mellea.stdlib.components.genslot import generative\n"
    )
    sub = tmp_path / "pkg"
    sub.mkdir()
    (sub / "b.py").write_text("x: GenerativeSlot = foo()\n")

    result = fix_genslot_path(tmp_path)

    assert result.files_affected == 2
    assert result.total_fixes == 2


def test_skips_venv(tmp_path: Path):
    venv = tmp_path / ".venv" / "lib"
    venv.mkdir(parents=True)
    (venv / "bad.py").write_text(
        "from mellea.stdlib.components.genslot import generative\n"
    )

    result = fix_genslot_path(tmp_path, dry_run=True)

    assert result.total_fixes == 0


def test_no_matches_returns_zero(tmp_path: Path):
    (tmp_path / "clean.py").write_text("import os\n")

    result = fix_genslot_path(tmp_path, dry_run=True)

    assert result.total_fixes == 0
    assert result.files_affected == 0
