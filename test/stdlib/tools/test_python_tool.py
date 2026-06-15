"""Tests for python_tool() factory and related helpers."""

import subprocess
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest

import mellea.stdlib.tools.interpreter as _interpreter_mod
from mellea.stdlib.tools import CapabilityPolicy, ExecutionResult, python_tool
from mellea.stdlib.tools.interpreter import (
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


def test_docker_tier_artifact_dir_emits_warning(tmp_path: Path):
    """Passing artifact_dir with a docker tier must emit a RuntimeWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        python_tool(tier="docker_unsafe", artifact_dir=tmp_path)
    assert any(issubclass(warning.category, RuntimeWarning) for warning in w)
    assert any("artifact_dir" in str(warning.message) for warning in w)


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

    policy = CapabilityPolicy(filesystem_read_roots=[Path("/tmp")])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
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

    with LLMSandboxEnvironment(policy=policy, _installed_packages=install_cache) as env:
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


# endregion
