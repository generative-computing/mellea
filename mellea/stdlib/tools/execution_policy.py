# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capability policy, artifact model, and compatibility matrix for code execution environments.

Four execution tiers are available, selectable by intent rather than by class name:

- `"local_unsafe"`  — subprocess in the current Python env, no policy applied.
- `"local"`         — subprocess in the current Python env, policy declared and partially enforced.
- `"docker_unsafe"` — Docker-isolated execution via llm-sandbox, no policy applied.
- `"docker"`        — Docker-isolated execution via llm-sandbox, policy declared and partially enforced.

`CapabilityPolicy` declares what a code execution environment is *allowed* to do.
Enforcement is honest: each capability has a companion `ENFORCED_*` class attribute
indicating whether the declared value is actively enforced at runtime or is informational
only.

`Artifact` represents a file produced by execution and exported from the environment.

`COMPATIBILITY_MATRIX` records which capabilities each tier supports.
"""

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal

ExecutionTier = Literal["static", "local_unsafe", "local", "docker_unsafe", "docker"]


@dataclass
class Artifact:
    """A file produced by code execution and exported from the execution environment.

    Args:
        path (Path): Absolute path on the host where the artifact was written.
        size_bytes (int | None): File size in bytes, or `None` if unknown.
        content_type (str | None): MIME type or informal label (e.g. `"text/csv"`,
            `"image/png"`), or `None` if undetermined.
    """

    path: Path
    size_bytes: int | None = None
    content_type: str | None = None


@dataclass
class CapabilityPolicy:
    """Declared capabilities and resource limits for a code execution environment.

    The enforcement gap — the difference between what is *declared* and what is
    *actively enforced at runtime* — is made explicit through per-field
    `ENFORCED_*` class attributes.  Callers and UX layers can read these to
    decide whether to prompt the user ("allow once / allow always") or display
    a warning.

    .. warning::
        **Most fields are declarative only and are not enforced at runtime.**
        Setting `network_access=False`, `package_installation=False`,
        `subprocess_execution=False`, or `env_var_access=False` does *not*
        block those capabilities in the current implementation — the declared
        value is informational only.  Check the `ENFORCED_*` class attributes
        to see what is actively enforced.  For stronger isolation use a
        `"docker"` execution tier.

        `filesystem_read_roots` and `filesystem_write_roots` emit a
        :class:`UserWarning` at construction time when set to a non-`None`
        value, because those restrictions are also unenforced.

    Args:
        filesystem_read_roots (list[Path] | None): Host paths the environment may
            read.  `None` means unrestricted.  **Declared only — not enforced.**
            A :class:`UserWarning` is emitted when this is set to a non-`None`
            value.
        filesystem_write_roots (list[Path] | None): Host paths the environment may
            write.  `None` means unrestricted.  **Declared only — not enforced.**
            A :class:`UserWarning` is emitted when this is set to a non-`None`
            value.
        network_access (bool): Whether outbound network connections are allowed.
            Defaults to `False`.  **Declared only — not enforced.**
        package_installation (bool): Whether the environment may install packages.
            **Declared only — not enforced.**
        subprocess_execution (bool): Whether spawning child processes is allowed.
            **Declared only — not enforced.**
        env_var_access (bool): Whether environment variables are readable.
            **Declared only — not enforced.**
        timeout (int): Wall-clock seconds before execution is killed.  Enforced.
        stdout_max_bytes (int | None): Truncate stdout to this byte count; `None`
            disables truncation.  Enforced.
        stderr_max_bytes (int | None): Truncate stderr to this byte count; `None`
            disables truncation.  Enforced.
        artifact_export_paths (list[Path]): Paths inside the container/environment
            to copy out after execution as `Artifact` objects.  Enforced.
        packages (list[str]): Python packages to install (via `pip install`) before
            execution.  Enforced — the runtime installs packages prior to executing
            user code and aborts with a skipped `ExecutionResult` if installation
            fails.  Failed packages are not retried on subsequent calls (clear
            `_failed_packages` on the environment to force a retry).  Local tiers
            use `uv pip install` / `python -m pip`; Docker tiers run
            `pip install` inside the container.  Defaults to `[]`.
    """

    filesystem_read_roots: list[Path] | None = None
    filesystem_write_roots: list[Path] | None = None
    network_access: bool = False
    package_installation: bool = False
    subprocess_execution: bool = False
    env_var_access: bool = True
    timeout: int = 30
    stdout_max_bytes: int | None = None
    stderr_max_bytes: int | None = None
    artifact_export_paths: list[Path] = field(default_factory=list)
    packages: list[str] = field(default_factory=list)

    # True  = this policy value is actively enforced by the runtime.
    # False = this policy value is declarative only (honest gap).
    ENFORCED_filesystem_read_roots: ClassVar[bool] = False
    ENFORCED_filesystem_write_roots: ClassVar[bool] = False
    ENFORCED_network_access: ClassVar[bool] = False
    ENFORCED_package_installation: ClassVar[bool] = False
    ENFORCED_subprocess_execution: ClassVar[bool] = False
    ENFORCED_env_var_access: ClassVar[bool] = False
    ENFORCED_timeout: ClassVar[bool] = True
    ENFORCED_stdout_max_bytes: ClassVar[bool] = True
    ENFORCED_stderr_max_bytes: ClassVar[bool] = True
    ENFORCED_artifact_export_paths: ClassVar[bool] = True
    ENFORCED_packages: ClassVar[bool] = True

    def __post_init__(self) -> None:
        """Warn when unenforced path-restriction fields are explicitly set.

        `filesystem_read_roots` and `filesystem_write_roots` are the only
        unenforced fields whose default value (`None`) means *unrestricted*,
        so a non-`None` value unambiguously signals an intent to restrict
        access that the runtime cannot enforce.  Emit a `UserWarning` in that
        case so the false-confidence scenario is caught at construction time.

        Boolean unenforced fields (`network_access`, `package_installation`,
        `subprocess_execution`, `env_var_access`) default to their most
        restrictive value, so they cannot be distinguished from an explicit
        restriction — callers relying on those are covered by the docstring
        `.. warning::` blocks rather than a runtime warning.
        """
        triggered: list[str] = []
        if self.filesystem_read_roots is not None:
            triggered.append(
                "  'filesystem_read_roots': declared but not enforced at runtime"
            )
        if self.filesystem_write_roots is not None:
            triggered.append(
                "  'filesystem_write_roots': declared but not enforced at runtime"
            )
        if triggered:
            field_list = "\n".join(triggered)
            warnings.warn(
                "CapabilityPolicy was constructed with path-restriction fields that are "
                "declared but not enforced at runtime — the declared paths will not "
                "restrict actual filesystem access:\n"
                f"{field_list}\n"
                "These fields are informational only. "
                "Check CapabilityPolicy.ENFORCED_* class attributes to see what is "
                "actively enforced. Use a 'docker' tier for stronger isolation.",
                UserWarning,
                stacklevel=3,
            )

    def unenforced_capabilities(self) -> list[str]:
        """Return capability names that are declared but not enforced at runtime.

        Returns:
            list[str]: Field names whose declared values are informational only.
        """
        return [
            name.removeprefix("ENFORCED_")
            for name, val in _iter_enforced_flags(type(self))
            if val is False
        ]

    def enforced_capabilities(self) -> list[str]:
        """Return capability names that are actively enforced at runtime.

        Returns:
            list[str]: Field names whose declared values are honoured by the runtime.
        """
        return [
            name.removeprefix("ENFORCED_")
            for name, val in _iter_enforced_flags(type(self))
            if val is True
        ]


def _iter_enforced_flags(cls: type) -> list[tuple[str, bool]]:
    """Collect ENFORCED_* class attributes across the MRO, innermost class wins."""
    seen: dict[str, bool] = {}
    # Reverse so base-class values are overwritten by subclass values.
    for klass in reversed(cls.__mro__):
        for name, val in vars(klass).items():
            if name.startswith("ENFORCED_") and isinstance(val, bool):
                seen[name] = val
    return list(seen.items())


# Canonical default policies for each execution tier.
# "unsafe" tiers carry no policy — pass None to make_execution_environment.

LOCAL_POLICY = CapabilityPolicy(
    network_access=False,
    package_installation=False,
    subprocess_execution=True,
    env_var_access=True,
    timeout=30,
)

DOCKER_POLICY = CapabilityPolicy(
    network_access=False,
    package_installation=True,
    subprocess_execution=True,
    env_var_access=True,
    timeout=60,
)


COMPATIBILITY_MATRIX: dict[str, dict[str, bool]] = {
    "static": {
        "execute_code": False,
        "timeout_enforcement": False,
        "import_allowlist": True,
        "policy_applied": False,
        "copy_in": False,
        "copy_out": False,
        "package_installation": False,
        "docker_isolation": False,
    },
    "local_unsafe": {
        "execute_code": True,
        "timeout_enforcement": True,
        "import_allowlist": True,
        "policy_applied": False,
        "copy_in": False,
        "copy_out": False,
        "package_installation": True,
        "docker_isolation": False,
    },
    "local": {
        "execute_code": True,
        "timeout_enforcement": True,
        "import_allowlist": True,
        "policy_applied": True,
        "copy_in": False,
        "copy_out": False,
        "package_installation": True,
        "docker_isolation": False,
    },
    "docker_unsafe": {
        "execute_code": True,
        "timeout_enforcement": True,
        "import_allowlist": True,
        "policy_applied": False,
        "copy_in": True,
        "copy_out": True,
        "package_installation": True,
        "docker_isolation": True,
    },
    "docker": {
        "execute_code": True,
        "timeout_enforcement": True,
        "import_allowlist": True,
        "policy_applied": True,
        "copy_in": True,
        "copy_out": True,
        "package_installation": True,
        "docker_isolation": True,
    },
}
