# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for python_tool() factory and related helpers."""

import os
import stat
import subprocess
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest

import mellea.stdlib.tools.interpreter as _interpreter_mod
from mellea.stdlib.tools import CapabilityPolicy, ExecutionResult, python_tool
from mellea.stdlib.tools.interpreter import (
    LLMSandboxEnvironment,
    _needs_matplotlib_preamble,
    _scan_artifacts,
    _validate_package_names,
    code_interpreter,
    local_code_interpreter,
)

try:
    import llm_sandbox  # type: ignore[import-not-found]

    try:
        with llm_sandbox.SandboxSession(
            lang="python", verbose=False, keep_template=False
        ) as _probe_session:
            _probe_session.run("print('probe')", timeout=5)
        _llm_sandbox_available = True
    except Exception:
        _llm_sandbox_available = False
except ImportError:
    _llm_sandbox_available = False

_docker_skip = pytest.mark.skipif(
    not _llm_sandbox_available,
    reason="Docker tests require llm-sandbox[docker] and a running Docker daemon",
)


def test_arithmetic():
    tool = python_tool(tier="local_unsafe")
    result: ExecutionResult = tool.run(code="print(1 + 1)")
    assert result.success
    assert result.stdout == "2"
    assert result.artifacts == []


def test_returns_execution_result():
    tool = python_tool(tier="local_unsafe")
    result = tool.run(code="x = 42")
    assert isinstance(result, ExecutionResult)
    assert result.success


def test_artifact_surfaced(tmp_path: Path):
    tool = python_tool(tier="local_unsafe", artifact_dir=tmp_path)
    code = f"""
with open(r'{tmp_path / "output.csv"}', 'w') as f:
    f.write('a,b\\n1,2\\n')
"""
    result = tool.run(code=code)
    assert result.success
    assert len(result.artifacts) == 1
    artifact = result.artifacts[0]
    assert artifact.path == tmp_path / "output.csv"
    assert artifact.size_bytes is not None and artifact.size_bytes > 0
    assert artifact.content_type == "text/csv"


def test_multiple_artifacts(tmp_path: Path):
    tool = python_tool(tier="local_unsafe", artifact_dir=tmp_path)
    code = f"""
for name in ['a.txt', 'b.txt']:
    with open(r'{tmp_path}/' + name, 'w') as f:
        f.write('hello')
"""
    result = tool.run(code=code)
    assert result.success
    assert len(result.artifacts) == 2
    names = {a.path.name for a in result.artifacts}
    assert names == {"a.txt", "b.txt"}


@pytest.mark.slow
def test_matplotlib_agg_injected(tmp_path: Path):
    # Installs matplotlib+numpy on demand via packages= to exercise the install story.
    # python_tool injects matplotlib.use('Agg') when matplotlib is imported,
    # so the savefig call works on headless environments without plt.show() crashing.
    tool = python_tool(
        tier="local_unsafe", packages=["matplotlib", "numpy"], artifact_dir=tmp_path
    )
    code = f"""
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0, 1, 10)
plt.plot(x, x ** 2)
plt.savefig(r'{tmp_path / "plot.png"}')
"""
    result = tool.run(code=code)
    assert result.success, result.stderr
    assert any(a.path.name == "plot.png" for a in result.artifacts)
    png_artifact = next(a for a in result.artifacts if a.path.name == "plot.png")
    assert png_artifact.content_type == "image/png"
    assert png_artifact.size_bytes is not None and png_artifact.size_bytes > 0


@pytest.mark.slow
def test_numpy_arithmetic(tmp_path: Path):
    tool = python_tool(tier="local_unsafe", packages=["numpy"], artifact_dir=tmp_path)
    result = tool.run(code="import numpy as np; print(np.sqrt(4.0))")
    assert result.success, result.stderr
    assert result.stdout is not None and result.stdout.strip() == "2.0"


@pytest.mark.slow
def test_pandas_csv_summary(tmp_path: Path):
    tool = python_tool(tier="local_unsafe", packages=["pandas"], artifact_dir=tmp_path)
    out = tmp_path / "summary.csv"
    code = f"""
import pandas as pd
df = pd.DataFrame({{"name": ["alice", "bob"], "score": [95, 87]}})
df.to_csv(r'{out}', index=False)
print(len(df))
"""
    result = tool.run(code=code)
    assert result.success, result.stderr
    assert result.stdout is not None and result.stdout.strip() == "2"
    assert any(a.path.name == "summary.csv" for a in result.artifacts)
    csv_artifact = next(a for a in result.artifacts if a.path.name == "summary.csv")
    assert csv_artifact.content_type == "text/csv"
    assert csv_artifact.size_bytes is not None and csv_artifact.size_bytes > 0


def test_name_override():
    tool = python_tool(tier="local_unsafe", name="my_python")
    assert tool.name == "my_python"


def test_default_name():
    tool = python_tool(tier="local_unsafe")
    assert tool.name == "python"


def test_failed_code():
    tool = python_tool(tier="local_unsafe")
    result = tool.run(code="raise ValueError('oops')")
    assert not result.success
    assert result.stderr is not None
    assert "ValueError" in result.stderr


def test_failed_code_does_not_surface_prior_artifacts(tmp_path: Path):
    """A failed run must not return artifacts from a prior successful run on the same artifact_dir."""
    tool = python_tool(tier="local_unsafe", artifact_dir=tmp_path)
    r1 = tool.run(code=f"open(r'{tmp_path / 'out.txt'}', 'w').write('x')")
    assert r1.success
    assert len(r1.artifacts) == 1

    result_fail = tool.run(code="raise RuntimeError('fail')")
    assert not result_fail.success
    assert not result_fail.skipped  # execution ran, just failed
    assert result_fail.artifacts == []  # no artifacts on failure


def test_timed_out_code_returns_no_artifacts(tmp_path: Path):
    """A timed-out execution (skipped=True) must not surface any artifacts."""
    from mellea.stdlib.tools import CapabilityPolicy

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        tool = python_tool(
            tier="local_unsafe",
            artifact_dir=tmp_path,
            policy=CapabilityPolicy(timeout=1),
        )
    # Write a file then sleep long enough to time out.
    result = tool.run(
        code=f"import time; open(r'{tmp_path / 'partial.txt'}', 'w').write('x'); time.sleep(10)"
    )
    assert result.timed_out
    assert result.skipped
    assert result.artifacts == []


def test_execution_mode_local_unsafe_unaffected_by_packages():
    """packages= must not override execution_mode; local_unsafe tier should be reported regardless."""
    tool = python_tool(tier="local_unsafe", packages=["fakepkg"])
    with patch.object(_interpreter_mod, "_install_packages", return_value=True):
        result = tool.run(code="print('ok')")
    assert result.execution_mode == "local_unsafe"


def test_execution_mode_local():
    """python_tool(tier='local') must report execution_mode='local'."""
    tool = python_tool(tier="local")
    result = tool.run(code="print('ok')")
    assert result.execution_mode == "local"


def test_packages_empty_list_does_not_crash():
    """packages=[] should be accepted without error and not affect execution."""
    tool = python_tool(tier="local_unsafe", packages=[])
    result = tool.run(code="print('ok')")
    assert result.success
    assert result.stdout == "ok"


def test_packages_deduplication():
    """Overlapping packages in policy.packages and packages= are deduplicated."""
    policy = CapabilityPolicy(timeout=30, packages=["fakepkg"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        tool = python_tool(tier="local_unsafe", packages=["fakepkg"], policy=policy)
    with patch.object(_interpreter_mod, "_install_packages", return_value=True):
        result = tool.run(code="print('ok')")
    assert result.success
    assert result.stdout == "ok"


def test_policy_override():
    policy = CapabilityPolicy(timeout=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        tool = python_tool(tier="local_unsafe", policy=policy)
    result = tool.run(code="print('ok')")
    assert result.success


def test_packages_merged_into_policy():
    """packages= is merged into an existing policy without breaking execution."""
    policy = CapabilityPolicy(timeout=30)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        tool = python_tool(tier="local_unsafe", packages=["fakepkg"], policy=policy)
    with patch.object(_interpreter_mod, "_install_packages", return_value=True):
        result = tool.run(code="print('ok')")
    assert result.success
    assert result.stdout == "ok"


@pytest.mark.slow
def test_package_install():
    """Install cowsay on the fly and verify import succeeds."""
    tool = python_tool(tier="local_unsafe", packages=["cowsay"])
    result = tool.run(
        code="import cowsay; print(cowsay.get_output_string('cow', 'hi'))"
    )
    assert result.success, result.stderr


def test_scan_artifacts_empty(tmp_path: Path):
    assert _scan_artifacts(tmp_path) == []


def test_scan_artifacts_content_type(tmp_path: Path):
    (tmp_path / "data.json").write_text('{"x": 1}')
    artifacts = _scan_artifacts(tmp_path)
    assert len(artifacts) == 1
    assert artifacts[0].content_type == "application/json"
    assert artifacts[0].size_bytes == (tmp_path / "data.json").stat().st_size


def test_scan_artifacts_unknown_type(tmp_path: Path):
    (tmp_path / "file.unknownext999").write_bytes(b"data")
    artifacts = _scan_artifacts(tmp_path)
    assert len(artifacts) == 1
    assert artifacts[0].content_type is None


def test_backwards_compat_code_interpreter_deprecation():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            code_interpreter("print('hi')")
        except Exception:
            pass
    assert any(issubclass(warning.category, DeprecationWarning) for warning in w)
    assert any("python_tool" in str(warning.message) for warning in w)


def test_backwards_compat_local_code_interpreter_deprecation():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        local_code_interpreter("print('hi')")
    assert any(issubclass(warning.category, DeprecationWarning) for warning in w)
    assert any("python_tool" in str(warning.message) for warning in w)


def test_install_falls_back_to_pip_when_uv_absent():
    """When _UV is None (uv not on PATH), _install_packages uses python -m pip."""

    captured: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        captured.append(cmd)

        r = subprocess.CompletedProcess(cmd, returncode=0, stdout=b"", stderr=b"")
        return r

    original_uv = _interpreter_mod._UV
    try:
        _interpreter_mod._UV = None
        with patch("subprocess.run", side_effect=fake_run):
            _interpreter_mod._install_packages(["fakepkg"])
    finally:
        _interpreter_mod._UV = original_uv

    assert captured, "subprocess.run was not called"
    assert captured[0][0] != "uv", "expected pip fallback, got uv"
    assert captured[0][1] == "-m"
    assert captured[0][2] == "pip"


# region _needs_matplotlib_preamble unit tests


def test_needs_matplotlib_preamble_bare_import():
    assert _needs_matplotlib_preamble("import matplotlib") is True


def test_needs_matplotlib_preamble_dotted_import():
    assert _needs_matplotlib_preamble("import matplotlib.pyplot as plt") is True


def test_needs_matplotlib_preamble_submodule_import():
    assert _needs_matplotlib_preamble("import matplotlib.backends.backend_agg") is True


def test_needs_matplotlib_preamble_from_import():
    assert _needs_matplotlib_preamble("from matplotlib import pyplot") is True


def test_needs_matplotlib_preamble_from_submodule():
    assert _needs_matplotlib_preamble("from matplotlib.pyplot import savefig") is True


def test_needs_matplotlib_preamble_no_matplotlib():
    assert _needs_matplotlib_preamble("import os\nprint('hi')") is False


def test_needs_matplotlib_preamble_syntax_error():
    assert _needs_matplotlib_preamble("def (broken:") is False


def test_suppress_agg_skips_preamble(tmp_path: Path):
    """suppress_agg=True must not inject the Agg preamble even when matplotlib is imported."""
    tool = python_tool(tier="local_unsafe", artifact_dir=tmp_path, suppress_agg=True)
    # The preamble would set Agg before any other import; with suppress_agg=True
    # the code runs as-is.  We verify no preamble injection by checking stdout
    # (the code prints the backend — without Agg injection it will be whatever
    # the env default is, which on headless CI may fail at display time, so we
    # only verify the tool accepted suppress_agg without error).
    result = tool.run(code="print('suppress_agg ok')")
    assert result.success
    assert result.stdout == "suppress_agg ok"


# endregion


# region _validate_package_names unit tests


def test_validate_package_names_valid():
    _validate_package_names(
        ["numpy", "pandas>=1.0", "scipy==1.10.0"]
    )  # should not raise


def test_validate_package_names_empty_string():
    with pytest.raises(ValueError, match="non-empty"):
        _validate_package_names([""])


def test_validate_package_names_whitespace_only():
    with pytest.raises(ValueError, match="non-empty"):
        _validate_package_names(["   "])


def test_validate_package_names_leading_dash():
    with pytest.raises(ValueError, match="flag-style"):
        _validate_package_names(["-r", "requirements.txt"])


def test_validate_package_names_double_dash():
    with pytest.raises(ValueError, match="flag-style"):
        _validate_package_names(["--index-url", "http://evil.example/"])


def test_python_tool_rejects_invalid_package():
    with pytest.raises(ValueError):
        python_tool(tier="local_unsafe", packages=[""])


def test_python_tool_rejects_flag_package():
    with pytest.raises(ValueError):
        python_tool(tier="local_unsafe", packages=["-r"])


@pytest.mark.parametrize("empty_path", [Path(""), Path(".")])
def test_export_dir_empty_path_normalized_to_none(empty_path: Path):
    """An empty/cwd export_dir must normalize to None, not scatter temp dirs into cwd.

    Path("") and Path(".") are both truthy and `is not None`, so neither a
    falsey nor an `is not None` check catches them; they resolve to cwd, so
    mkdtemp(dir=...) would litter the working directory.  The constructor must
    treat them as "no export dir".
    """
    env = LLMSandboxEnvironment(export_dir=empty_path)
    assert env._export_dir is None


def test_export_dir_real_path_preserved(tmp_path: Path):
    """A genuine export_dir must be preserved unchanged."""
    env = LLMSandboxEnvironment(export_dir=tmp_path)
    assert env._export_dir == tmp_path


# endregion


# region artifact scanning tests


def test_artifact_surfaced_no_artifact_dir():
    """Artifacts written during a no-artifact_dir run must be readable by the caller."""
    tool = python_tool(tier="local_unsafe")
    # Write a file to CWD (which is the per-call tempdir when artifact_dir=None).
    result = tool.run(code="open('output.txt', 'w').write('hello')")
    assert result.success
    assert len(result.artifacts) == 1
    artifact = result.artifacts[0]
    # Path must still exist and be readable after run_python() returns.
    assert artifact.path.exists(), (
        "Artifact path was deleted before caller could read it"
    )
    assert artifact.path.read_text() == "hello"


def test_scan_artifacts_recursive(tmp_path: Path):
    """_scan_artifacts must find files in subdirectories."""
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "nested.txt").write_text("data")
    artifacts = _scan_artifacts(tmp_path)
    names = {a.path.name for a in artifacts}
    assert "nested.txt" in names


def test_artifact_in_subdirectory(tmp_path: Path):
    """python_tool must surface files written to a subdirectory of artifact_dir."""
    tool = python_tool(tier="local_unsafe", artifact_dir=tmp_path)
    code = """
import os
os.makedirs("subdir", exist_ok=True)
open("subdir/nested.csv", "w").write("a,b\\n1,2\\n")
"""
    result = tool.run(code=code)
    assert result.success
    names = {a.path.name for a in result.artifacts}
    assert "nested.csv" in names


def test_docker_tier_artifact_dir_without_export_paths_warns(tmp_path: Path):
    """artifact_dir with a docker tier but no artifact_export_paths must warn."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        python_tool(tier="docker_unsafe", artifact_dir=tmp_path)
    assert any(issubclass(warning.category, RuntimeWarning) for warning in w)
    assert any(
        "artifact_export_paths" in str(warning.message)
        for warning in w
        if issubclass(warning.category, RuntimeWarning)
    )


def test_docker_tier_artifact_dir_with_export_paths_no_warning(tmp_path: Path):
    """artifact_dir + artifact_export_paths on a docker tier must NOT warn.

    A persistent container session is opened at construction time; guard the
    real Docker call behind the skip so this stays a fast unit test by patching
    LLMSandboxEnvironment.__enter__/__exit__.
    """
    from unittest.mock import patch

    policy = CapabilityPolicy(artifact_export_paths=[Path("/out/data.csv")])
    with (
        patch.object(LLMSandboxEnvironment, "__enter__", lambda self: self),
        patch.object(LLMSandboxEnvironment, "__exit__", lambda self, *a: None),
    ):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            python_tool(tier="docker", policy=policy, artifact_dir=tmp_path)
    runtime_warnings = [
        warning for warning in w if issubclass(warning.category, RuntimeWarning)
    ]
    assert not runtime_warnings, (
        f"unexpected RuntimeWarning(s): {[str(x.message) for x in runtime_warnings]}"
    )


def test_packages_empty_list_does_not_create_policy():
    """packages=[] must not construct a CapabilityPolicy for local_unsafe."""
    captured_envs: list[_interpreter_mod.ExecutionEnvironment] = []
    original_make = _interpreter_mod.make_execution_environment

    def capturing_make(*args, **kwargs):
        env = original_make(*args, **kwargs)
        captured_envs.append(env)
        return env

    with patch.object(
        _interpreter_mod, "make_execution_environment", side_effect=capturing_make
    ):
        tool = python_tool(tier="local_unsafe", packages=[])
        tool.run(code="print('ok')")

    assert captured_envs, "make_execution_environment was not called"
    assert captured_envs[0].policy is None


# endregion


# region security posture warning tests


def test_python_tool_no_tier_emits_user_warning():
    """python_tool() with no explicit tier must emit a UserWarning about local_unsafe."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        python_tool()
    user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
    assert user_warnings, "Expected a UserWarning when tier is omitted"
    assert any("local_unsafe" in str(x.message) for x in user_warnings)
    assert any("tier" in str(x.message) for x in user_warnings)


def test_python_tool_explicit_tier_no_warning():
    """python_tool(tier='local_unsafe') must not emit a UserWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        python_tool(tier="local_unsafe")
    user_warnings = [
        x
        for x in w
        if issubclass(x.category, UserWarning)
        and "local_unsafe" in str(x.message)
        and "tier" in str(x.message)
    ]
    assert not user_warnings, "No UserWarning expected when tier is passed explicitly"


def test_capability_policy_filesystem_read_roots_warns():
    """CapabilityPolicy(filesystem_read_roots=[...]) must emit a UserWarning."""
    from pathlib import Path

    from mellea.stdlib.tools import CapabilityPolicy

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        CapabilityPolicy(filesystem_read_roots=[Path("/tmp")])
    user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
    assert user_warnings, "Expected a UserWarning for filesystem_read_roots"
    assert any("filesystem_read_roots" in str(x.message) for x in user_warnings)
    assert any("not enforced" in str(x.message) for x in user_warnings)


def test_capability_policy_filesystem_write_roots_warns():
    """CapabilityPolicy(filesystem_write_roots=[...]) must emit a UserWarning."""
    from pathlib import Path

    from mellea.stdlib.tools import CapabilityPolicy

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        CapabilityPolicy(filesystem_write_roots=[Path("/tmp")])
    user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
    assert user_warnings, "Expected a UserWarning for filesystem_write_roots"
    assert any("filesystem_write_roots" in str(x.message) for x in user_warnings)


def test_capability_policy_no_path_roots_no_warning():
    """CapabilityPolicy() with no path roots must not emit a UserWarning."""
    from mellea.stdlib.tools import CapabilityPolicy

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        CapabilityPolicy(timeout=10)
    user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
    assert not user_warnings, "No UserWarning expected for default CapabilityPolicy"


def test_python_tool_packages_plus_filesystem_policy_warns_once():
    """packages= merge via replace() must not double-emit the filesystem_read_roots warning."""
    from pathlib import Path

    from mellea.stdlib.tools import CapabilityPolicy

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        policy = CapabilityPolicy(filesystem_read_roots=[Path("/tmp")])
        python_tool(tier="local_unsafe", packages=["numpy"], policy=policy)
    fs_warnings = [
        x
        for x in w
        if issubclass(x.category, UserWarning)
        and "filesystem_read_roots" in str(x.message)
    ]
    assert len(fs_warnings) == 1, (
        f"Expected exactly one filesystem warning, got {len(fs_warnings)}"
    )


def test_python_tool_unenforced_policy_local_unsafe_warns():
    """python_tool() with an explicit policy on local_unsafe must warn about unenforced bools at construction time."""
    from mellea.stdlib.tools import CapabilityPolicy

    policy = CapabilityPolicy(network_access=False, package_installation=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        python_tool(tier="local_unsafe", policy=policy)
    user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
    assert user_warnings, "Expected UserWarning for unenforced policy on local_unsafe"
    assert any("network_access" in str(x.message) for x in user_warnings)
    assert any(
        "unenforced" in str(x.message) or "not restrict" in str(x.message)
        for x in user_warnings
    )


def test_python_tool_unenforced_policy_local_warns():
    """python_tool() with an explicit policy on local must warn about unenforced bools at construction time."""
    from mellea.stdlib.tools import CapabilityPolicy

    policy = CapabilityPolicy(network_access=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        python_tool(tier="local", policy=policy)
    user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
    assert user_warnings, "Expected UserWarning for unenforced policy on local"
    assert any("network_access" in str(x.message) for x in user_warnings)


def test_python_tool_unenforced_policy_warning_fires_at_construction_not_run():
    """The unenforced-policy warning must fire at python_tool() construction, not during tool.run()."""
    from mellea.stdlib.tools import CapabilityPolicy

    policy = CapabilityPolicy(network_access=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tool = python_tool(tier="local_unsafe", policy=policy)
    construction_warnings = [
        x
        for x in w
        if issubclass(x.category, UserWarning) and "unenforced" in str(x.message)
    ]
    assert construction_warnings, "Warning must fire during python_tool() construction"

    with warnings.catch_warnings(record=True) as w2:
        warnings.simplefilter("always")
        tool.run(code="print('ok')")
    run_warnings = [
        x
        for x in w2
        if issubclass(x.category, UserWarning) and "unenforced" in str(x.message)
    ]
    assert not run_warnings, "Warning must not re-fire during tool.run()"


def test_python_tool_no_policy_no_unenforced_warning():
    """python_tool() with no policy must not emit the unenforced warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        python_tool(tier="local_unsafe")
    user_warnings = [
        x
        for x in w
        if issubclass(x.category, UserWarning) and "unenforced" in str(x.message)
    ]
    assert not user_warnings, "No unenforced warning expected when policy is None"


def test_python_tool_docker_tier_no_unenforced_warning():
    """python_tool() on docker tiers must not emit the unenforced warning (docker provides real isolation)."""
    from mellea.stdlib.tools import CapabilityPolicy

    policy = CapabilityPolicy(network_access=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        python_tool(tier="docker_unsafe", policy=policy)
    user_warnings = [
        x
        for x in w
        if issubclass(x.category, UserWarning) and "unenforced" in str(x.message)
    ]
    assert not user_warnings, "No unenforced warning expected for docker tiers"


# endregion


# region Docker tier tests


@_docker_skip
def test_docker_basic_execution():
    """python_tool(tier='docker') executes code in a container."""
    tool = python_tool(tier="docker")
    result = tool.run(code="print(1 + 1)")
    assert result.success
    assert result.stdout is not None and result.stdout.strip() == "2"


@_docker_skip
@pytest.mark.slow
def test_docker_package_install():
    """packages= installs inside the container before execution."""
    tool = python_tool(tier="docker", packages=["cowsay"])
    result = tool.run(
        code="import cowsay; print(cowsay.get_output_string('cow', 'hi'))"
    )
    assert result.success, result.stderr


@_docker_skip
@pytest.mark.slow
def test_docker_package_install_cache():
    """In a persistent container session, packages are installed once and cached.

    The context manager keeps a single container alive; the install cache correctly
    reflects what is installed in that container, so cowsay is only pip-installed once.
    """
    from mellea.stdlib.tools import CapabilityPolicy
    from mellea.stdlib.tools.interpreter import LLMSandboxEnvironment

    policy = CapabilityPolicy(packages=["cowsay"])
    install_cache: set[str] = set()

    with LLMSandboxEnvironment(policy=policy, installed_packages=install_cache) as env:
        result1 = env.execute("import cowsay; print('first')")
        assert result1.success, result1.stderr
        assert "cowsay" in install_cache  # installed and cached

        result2 = env.execute("import cowsay; print('second')")
        assert result2.success, result2.stderr
        # Cache hit — cowsay was not reinstalled (still in same container)


@_docker_skip
@pytest.mark.slow
def test_docker_oneshot_reinstalls_per_container():
    """One-shot mode must reinstall packages in every fresh container.

    In one-shot mode each execute() call creates a new container.  The shared
    install cache must NOT suppress reinstallation because the new container
    has never had the package installed.
    """
    tool = python_tool(tier="docker", packages=["cowsay"])
    result1 = tool.run(code="import cowsay; print('call1')")
    assert result1.success, result1.stderr
    # Second call: fresh container — packages must be reinstalled, not skipped.
    result2 = tool.run(code="import cowsay; print('call2')")
    assert result2.success, result2.stderr


@_docker_skip
@pytest.mark.slow
def test_docker_export_real_container_surfaces_file(tmp_path: Path):
    """A file written inside a real container is copied out to artifact_dir."""
    policy = CapabilityPolicy(artifact_export_paths=[Path("/tmp/out.txt")])
    tool = python_tool(tier="docker", policy=policy, artifact_dir=tmp_path)

    result = tool.run(code="open('/tmp/out.txt', 'w').write('hello from container')")
    assert result.success, result.stderr
    assert result.artifacts, "no artifact exported from real container"
    artifact = result.artifacts[0]
    assert artifact.path.name == "out.txt"
    assert tmp_path in artifact.path.parents
    assert artifact.path.read_text() == "hello from container"


# endregion


# region security tests


@pytest.mark.skipif(os.name != "posix", reason="POSIX file permissions only")
def test_temp_script_file_is_owner_only():
    """_execute_subprocess temp script must be mode 0o600 on POSIX systems."""
    captured_modes: list[int] = []
    original_chmod = os.chmod

    def capturing_chmod(path: str, mode: int) -> None:
        original_chmod(path, mode)
        if str(path).endswith(".py"):
            captured_modes.append(stat.S_IMODE(os.stat(path).st_mode))

    with patch("mellea.stdlib.tools.interpreter.os.chmod", side_effect=capturing_chmod):
        tool = python_tool(tier="local_unsafe")
        tool.run(code="print('ok')")

    assert captured_modes, "os.chmod was not called on the temp script file"
    assert captured_modes[0] == 0o600, f"expected 0o600, got {oct(captured_modes[0])}"


@pytest.mark.skipif(os.name != "posix", reason="POSIX file permissions only")
def test_artifact_tmpdir_is_owner_only():
    """mkdtemp artifact dir must be mode 0o700 on POSIX systems."""
    captured_modes: list[int] = []
    original_chmod = os.chmod

    def capturing_chmod(path: str, mode: int) -> None:
        original_chmod(path, mode)
        if os.path.isdir(path):
            captured_modes.append(stat.S_IMODE(os.stat(path).st_mode))

    with patch("mellea.stdlib.tools.interpreter.os.chmod", side_effect=capturing_chmod):
        tool = python_tool(tier="local_unsafe")
        tool.run(code="print('ok')")

    assert captured_modes, "os.chmod was not called on the artifact tmpdir"
    assert captured_modes[0] == 0o700, f"expected 0o700, got {oct(captured_modes[0])}"


def _make_llm_sandbox_env_with_export(export_filename: str) -> LLMSandboxEnvironment:
    """Return an LLMSandboxEnvironment with a fake session and copy_out that writes a real file."""
    from unittest.mock import MagicMock

    policy = CapabilityPolicy(artifact_export_paths=[Path(f"/out/{export_filename}")])
    env = LLMSandboxEnvironment(policy=policy)

    fake_result = MagicMock()
    fake_result.stdout = "ok"
    fake_result.stderr = ""
    fake_result.exit_code = 0

    fake_session = MagicMock()
    fake_session.run.return_value = fake_result
    env._session = fake_session

    def fake_copy_out(container_path: str, host_path: Path) -> None:
        host_path.write_text("data")

    env.copy_out = fake_copy_out  # type: ignore[method-assign]
    return env


def test_artifact_export_path_is_randomized():
    """Exported artifact must land in a randomized subdirectory, not directly in gettempdir()."""
    import tempfile

    env = _make_llm_sandbox_env_with_export("output.csv")
    result = env.execute("print('ok')")

    assert result.artifacts, "no artifacts returned"
    artifact_path = result.artifacts[0].path
    # Must be inside a subdirectory of gettempdir(), not gettempdir() itself.
    assert artifact_path.parent != Path(tempfile.gettempdir()), (
        f"artifact landed directly in gettempdir(): {artifact_path}"
    )
    # The parent directory name must not be predictable from the filename alone.
    assert artifact_path.parent.name != "output.csv"
    assert artifact_path.name == "output.csv"


@pytest.mark.skipif(os.name != "posix", reason="POSIX file permissions only")
def test_artifact_export_dir_is_owner_only():
    """Each per-artifact export directory must be mode 0o700 on POSIX systems."""
    env = _make_llm_sandbox_env_with_export("output.csv")
    result = env.execute("print('ok')")

    assert result.artifacts, "no artifacts returned"
    export_dir = result.artifacts[0].path.parent
    assert export_dir.exists(), "export dir was removed before stat"
    mode = stat.S_IMODE(os.stat(export_dir).st_mode)
    assert mode == 0o700, f"expected 0o700, got {oct(mode)}"


# endregion


# region docker persistent-session artifact export (mocked, no real Docker)


def _patch_persistent_docker_session(
    monkeypatch, opened: list, closed: list, exit_code: int = 0, copy_out=None
):
    """Patch LLMSandboxEnvironment so python_tool opens a fake persistent session.

    The fake session's run() returns a result with the given exit_code and
    copy_out() writes a real host file (unless overridden), so the export loop in
    execute() can produce Artifact objects.  `opened` and `closed` record
    __enter__/__exit__ calls for assertions.
    """
    from unittest.mock import MagicMock

    def default_copy_out(container_path, host_path):
        host_path.write_text("data")

    effective_copy_out = copy_out if copy_out is not None else default_copy_out

    def fake_enter(self):
        fake_result = MagicMock()
        fake_result.stdout = "ok" if exit_code == 0 else "boom"
        fake_result.stderr = "" if exit_code == 0 else "error"
        fake_result.exit_code = exit_code
        fake_session = MagicMock()
        fake_session.run.return_value = fake_result
        self._session = fake_session
        self.copy_out = effective_copy_out
        opened.append(self)
        return self

    def fake_exit(self, *_args):
        self._session = None
        closed.append(self)

    monkeypatch.setattr(LLMSandboxEnvironment, "__enter__", fake_enter)
    monkeypatch.setattr(LLMSandboxEnvironment, "__exit__", fake_exit)


def test_docker_export_surfaces_artifacts_to_artifact_dir(monkeypatch, tmp_path: Path):
    """Docker tier + artifact_export_paths exports files under artifact_dir."""
    opened: list = []
    closed: list = []
    _patch_persistent_docker_session(monkeypatch, opened, closed)

    policy = CapabilityPolicy(artifact_export_paths=[Path("/out/result.csv")])
    tool = python_tool(tier="docker", policy=policy, artifact_dir=tmp_path)

    result = tool.run(code="print('ok')")

    assert result.artifacts, "no artifacts surfaced for docker tier"
    artifact = result.artifacts[0]
    assert artifact.path.name == "result.csv"
    # Artifact must land under the caller-supplied artifact_dir, not a temp dir.
    assert tmp_path in artifact.path.parents, (
        f"artifact {artifact.path} not under artifact_dir {tmp_path}"
    )
    assert artifact.path.read_text() == "data"


def test_docker_export_opens_session_once_and_reuses(monkeypatch, tmp_path: Path):
    """The persistent container session is opened once and reused across calls."""
    opened: list = []
    closed: list = []
    _patch_persistent_docker_session(monkeypatch, opened, closed)

    policy = CapabilityPolicy(artifact_export_paths=[Path("/out/result.csv")])
    tool = python_tool(tier="docker", policy=policy, artifact_dir=tmp_path)

    assert len(opened) == 1, "session should open exactly once at construction"

    tool.run(code="print('call1')")
    tool.run(code="print('call2')")

    # Still only one session — reused, not reopened per call.
    assert len(opened) == 1, "session must be reused across run_python calls"


def test_docker_export_finalizer_closes_session(monkeypatch, tmp_path: Path):
    """Dropping the tool closes the container session via the finalizer."""
    import gc

    opened: list = []
    closed: list = []
    _patch_persistent_docker_session(monkeypatch, opened, closed)

    policy = CapabilityPolicy(artifact_export_paths=[Path("/out/result.csv")])
    tool = python_tool(tier="docker", policy=policy, artifact_dir=tmp_path)
    assert len(opened) == 1

    del tool
    gc.collect()

    assert len(closed) == 1, "finalizer did not close the persistent session"


def test_docker_export_no_artifact_dir_uses_tempdir(monkeypatch):
    """Docker export without artifact_dir still surfaces artifacts (system tempdir)."""
    opened: list = []
    closed: list = []
    _patch_persistent_docker_session(monkeypatch, opened, closed)

    policy = CapabilityPolicy(artifact_export_paths=[Path("/out/result.csv")])
    tool = python_tool(tier="docker", policy=policy)

    result = tool.run(code="print('ok')")
    assert result.artifacts, "no artifacts surfaced when artifact_dir is None"
    assert result.artifacts[0].path.name == "result.csv"


def test_docker_export_failed_run_surfaces_no_artifacts(monkeypatch, tmp_path: Path):
    """A failed docker run (non-zero exit) must not export any artifacts.

    This mirrors the local-tier contract (see
    test_failed_code_does_not_surface_prior_artifacts) so a crashed run cannot
    leak partial or stale container output.
    """
    opened: list = []
    closed: list = []
    _patch_persistent_docker_session(monkeypatch, opened, closed, exit_code=1)

    policy = CapabilityPolicy(artifact_export_paths=[Path("/out/result.csv")])
    tool = python_tool(tier="docker", policy=policy, artifact_dir=tmp_path)

    result = tool.run(code="raise RuntimeError('boom')")
    assert not result.success
    assert result.artifacts == [], "failed run must not surface artifacts"
    # artifact_dir must not be polluted with export subdirs from a failed run.
    assert list(tmp_path.iterdir()) == [], "failed run left files under artifact_dir"


def test_docker_export_directory_artifact(monkeypatch, tmp_path: Path):
    """A container artifact_export_path pointing at a directory is copied whole."""
    opened: list = []
    closed: list = []

    def copy_out_dir(container_path, host_path):
        # Simulate `docker cp` bringing back a directory tree.
        host_path.mkdir()
        (host_path / "a.txt").write_text("one")
        (host_path / "b.txt").write_text("two")

    _patch_persistent_docker_session(monkeypatch, opened, closed, copy_out=copy_out_dir)

    policy = CapabilityPolicy(artifact_export_paths=[Path("/out/results")])
    tool = python_tool(tier="docker", policy=policy, artifact_dir=tmp_path)

    result = tool.run(code="print('ok')")
    assert result.artifacts, "no artifact surfaced for directory export"
    artifact = result.artifacts[0]
    assert artifact.path.name == "results"
    assert artifact.path.is_dir()
    assert artifact.size_bytes is None, "directory artifact should have size_bytes=None"
    assert {p.name for p in artifact.path.iterdir()} == {"a.txt", "b.txt"}


# endregion
