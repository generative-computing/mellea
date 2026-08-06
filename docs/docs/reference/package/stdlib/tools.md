---
id: tools
title: "mellea.stdlib.tools"
sidebar_label: "tools"
sidebar_position: 11
description: "Implementations of tools."
# diataxis: reference
---

Source: [`mellea/stdlib/tools/__init__.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/tools/__init__.py) at commit `a535fc6345a0`.

Implementations of tools.

Declared exports (`__all__`): `COMPATIBILITY_MATRIX`, `DOCKER_POLICY`, `LOCAL_POLICY`, `Artifact`, `BashEnvironment`, `CapabilityPolicy`, `ExecutionEnvironment`, `ExecutionResult`, `ExecutionTier`, `LLMSandboxEnvironment`, `StaticAnalysisEnvironment`, `StaticBashEnvironment`, `UnsafeEnvironment`, `bash_executor`, `code_interpreter`, `local_code_interpreter`, `make_execution_environment`, `python_tool`

---

## Module `mellea.stdlib.tools.execution_policy`

Source: [`mellea/stdlib/tools/execution_policy.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/tools/execution_policy.py) at commit `a535fc6345a0`.

Capability policy, artifact model, and compatibility matrix for code execution environments.

### `Artifact`

*class* — [line 32](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/tools/execution_policy.py#L32) 

A file produced by code execution and exported from the execution environment.

### `CapabilityPolicy`

*class* — [line 48](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/tools/execution_policy.py#L48) 

Declared capabilities and resource limits for a code execution environment.

Methods (defined on this class; inherited members not listed):

- `unenforced_capabilities() -> list[str]` — [line 167](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/tools/execution_policy.py#L167)
  Return capability names that are declared but not enforced at runtime.
- `enforced_capabilities() -> list[str]` — [line 179](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/tools/execution_policy.py#L179)
  Return capability names that are actively enforced at runtime.

---

## Module `mellea.stdlib.tools.interpreter`

Source: [`mellea/stdlib/tools/interpreter.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/tools/interpreter.py) at commit `a535fc6345a0`.

Code interpreter tool and execution environments for agentic workflows.

### `ExecutionResult`

*class* — [line 82](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/tools/interpreter.py#L82) 

Result of code execution.

Methods (defined on this class; inherited members not listed):

- `to_validationresult_reason() -> str` — [line 127](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/tools/interpreter.py#L127)
  Map an ExecutionResult to a ValidationResult reason string.

### `ExecutionEnvironment`

*class* — [line 151](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/tools/interpreter.py#L151) (`ABC`)

Abstract environment for executing Python code.

Constructor: `ExecutionEnvironment(allowed_imports: list[str] | None = None, policy: CapabilityPolicy | None = None, working_directory: str | None = None)`

Methods (defined on this class; inherited members not listed):

- `execute(code: str, timeout: int | None = None) -> ExecutionResult` — [line 176](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/tools/interpreter.py#L176)
  Execute the given code and return the result.
- `copy_in(host_path: Path, container_path: str) -> None` — [line 197](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/tools/interpreter.py#L197)
  Copy a file from the host into the execution environment.
- `copy_out(container_path: str, host_path: Path) -> None` — [line 212](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/tools/interpreter.py#L212)
  Copy a file from the execution environment to the host.

### `StaticAnalysisEnvironment`

*class* — [line 228](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/tools/interpreter.py#L228) (`ExecutionEnvironment`)

Safe environment that validates but does not execute code.

Methods (defined on this class; inherited members not listed):

- `execute(code: str, timeout: int | None = None) -> ExecutionResult` — [line 231](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/tools/interpreter.py#L231)
  Validate code syntax and imports without executing.

### `UnsafeEnvironment`

*class* — [line 280](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/tools/interpreter.py#L280) (`ExecutionEnvironment`)

Environment that executes code directly via subprocess.

Constructor: `UnsafeEnvironment(allowed_imports: list[str] | None = None, policy: CapabilityPolicy | None = None, working_directory: str | None = None, installed_packages: set[str] | None = None, failed_packages: set[str] | None = None, tier: str | None = None) -> None`

Methods (defined on this class; inherited members not listed):

- `execute(code: str, timeout: int | None = None) -> ExecutionResult` — [line 330](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/tools/interpreter.py#L330)
  Execute code with subprocess after checking imports.

### `LLMSandboxEnvironment`

*class* — [line 447](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/tools/interpreter.py#L447) (`ExecutionEnvironment`)

Docker-isolated execution environment via `llm-sandbox`.

Constructor: `LLMSandboxEnvironment(allowed_imports: list[str] | None = None, policy: CapabilityPolicy | None = None, working_directory: str | None = None, installed_packages: set[str] | None = None, failed_packages: set[str] | None = None, tier: str | None = None, export_dir: Path | None = None)`

Methods (defined on this class; inherited members not listed):

- `copy_in(host_path: Path, container_path: str) -> None` — [line 556](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/tools/interpreter.py#L556)
  Copy a file from the host into the running Docker container via `docker cp`.
- `copy_out(container_path: str, host_path: Path) -> None` — [line 575](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/tools/interpreter.py#L575)
  Copy a file from the running Docker container to the host via `docker cp`.
- `execute(code: str, timeout: int | None = None) -> ExecutionResult` — [line 667](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/tools/interpreter.py#L667)
  Execute code in a Docker container.

### `make_execution_environment()`

*function* — [line 878](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/tools/interpreter.py#L878)

`make_execution_environment(tier: ExecutionTier, policy: CapabilityPolicy | None = None, allowed_imports: list[str] | None = None, working_directory: str | None = None, _install_cache: set[str] | None = None, _failed_cache: set[str] | None = None, export_dir: Path | None = None) -> ExecutionEnvironment`

Create an :class:`ExecutionEnvironment` for the given tier.

### `python_tool()`

*function* — [line 1106](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/tools/interpreter.py#L1106)

`python_tool(tier: ExecutionTier | None = None, packages: list[str] | None = None, artifact_dir: Path | None = None, policy: CapabilityPolicy | None = None, allowed_imports: list[str] | None = None, name: str = 'python', suppress_agg: bool = False) -> MelleaTool`

Create a configurable Python execution tool that returns structured artifacts.

### `code_interpreter()`

*function* — [line 1405](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/tools/interpreter.py#L1405)

`code_interpreter(code: str) -> ExecutionResult`

Execute Python code in a Docker sandbox (docker_unsafe tier).

### `local_code_interpreter()`

*function* — [line 1431](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/tools/interpreter.py#L1431)

`local_code_interpreter(code: str) -> ExecutionResult`

Execute Python code in the current process environment (local_unsafe tier).

---

## Module `mellea.stdlib.tools.mcp`

Source: [`mellea/stdlib/tools/mcp.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/tools/mcp.py) at commit `a535fc6345a0`.

MCP tool discovery and MelleaTool wrapping.

Declared exports (`__all__`): `MCPToolSpec`, `discover_mcp_tools`, `http_connection`, `sse_connection`, `stdio_connection`

### `MCPToolSpec`

*class* — [line 54](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/tools/mcp.py#L54) 

Metadata for a single tool from an MCP server.

Constructor: `MCPToolSpec(name: str, description: str, input_schema: dict[str, Any], connection: dict[str, Any]) -> None`

Methods (defined on this class; inherited members not listed):

- `as_mellea_tool() -> MelleaTool` — [line 82](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/tools/mcp.py#L82)
  Create a callable `MelleaTool` from this spec.

### `discover_mcp_tools()`

*async function* — [line 112](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/tools/mcp.py#L112)

`discover_mcp_tools(connection: dict[str, Any]) -> list[MCPToolSpec]`

Discover all tools on an MCP server and return their metadata.

### `http_connection()`

*function* — [line 140](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/tools/mcp.py#L140)

`http_connection(url: str, *, api_key: str | None = None, headers: dict[str, str] | None = None, connect_timeout: float = 30.0, read_timeout: float = 300.0) -> dict[str, Any]`

Build a Streamable HTTP connection config.

### `sse_connection()`

*function* — [line 174](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/tools/mcp.py#L174)

`sse_connection(url: str, *, api_key: str | None = None, headers: dict[str, str] | None = None, connect_timeout: float = 30.0, read_timeout: float = 300.0) -> dict[str, Any]`

Build an SSE connection config.

### `stdio_connection()`

*function* — [line 208](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/tools/mcp.py#L208)

`stdio_connection(command: str, *, args: list[str] | None = None, env: dict[str, str] | None = None, timeout: float = 300.0) -> dict[str, Any]`

Build a stdio connection config.

---

## Module `mellea.stdlib.tools.shell`

Source: [`mellea/stdlib/tools/shell.py`](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/tools/shell.py) at commit `a535fc6345a0`.

Bash shell command execution tool and execution environments for agentic workflows.

### `BashEnvironment`

*class* — [line 683](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/tools/shell.py#L683) (`ABC`)

Abstract environment for executing bash commands.

Constructor: `BashEnvironment(allowed_paths: list[str] | None = None, working_dir: str | None = None, timeout: int = 60)`

Methods (defined on this class; inherited members not listed):

- `execute(command: str) -> ExecutionResult` — [line 828](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/tools/shell.py#L828)
  Execute the given bash command and return the result.

### `StaticBashEnvironment`

*class* — [line 840](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/tools/shell.py#L840) (`BashEnvironment`)

Safe environment that validates but does not execute bash commands.

Methods (defined on this class; inherited members not listed):

- `execute(command: str) -> ExecutionResult` — [line 850](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/tools/shell.py#L850)
  Parse and validate command without executing.

### `bash_executor()`

*function* — [line 970](https://github.com/generative-computing/mellea/blob/a535fc6345a00885bfc0f5ac70089df6b0fdf4f5/mellea/stdlib/tools/shell.py#L970)

`bash_executor(command: str, working_dir: str | None = None, allowed_paths: list[str] | None = None) -> ExecutionResult`

Execute a bash command with denylist safety checks.

---

*Generated from source by `docs/scripts/generate_package_map.py` — do not edit by hand; regenerate instead. Symbols and annotations reflect the pinned commit exactly; where type annotations are absent in source, this page says so rather than guessing.*
