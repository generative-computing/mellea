"""Bee AI Platform integration for Mellea.

This module provides integration with the Bee AI Platform (BAIP) to enable GUI-based
chat interfaces and trace visualization for Mellea programs.
"""

import datetime
import json
import os
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Union

# Gracefully handle missing BeeAI framework dependencies
try:
    from mellea.backends.beeai import BeeAIBackend

    BEEAI_AVAILABLE = True
except ImportError:
    BEEAI_AVAILABLE = False
    # Create a placeholder class for when BeeAI is not available
    from mellea.backends.formatter import FormatterBackend

    BeeAIBackend = FormatterBackend  # type: ignore

from mellea.backends import BaseModelSubclass
from mellea.backends.types import ModelOption
from mellea.helpers.fancy_logger import FancyLogger
from mellea.stdlib.base import CBlock, Component, Context, GenerateLog, ModelOutputThunk


class BeeAITrace:
    """Represents a trace entry for BeeAI Platform visualization."""

    def __init__(
        self,
        trace_id: str,
        name: str,
        start_time: datetime.datetime,
        end_time: datetime.datetime | None = None,
        input_data: dict[str, Any] | None = None,
        output_data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        parent_trace_id: str | None = None,
    ):
        self.trace_id = trace_id
        self.name = name
        self.start_time = start_time
        self.end_time = end_time
        self.input_data = input_data or {}
        self.output_data = output_data or {}
        self.metadata = metadata or {}
        self.parent_trace_id = parent_trace_id
        self.child_traces: list[BeeAITrace] = []

    def finish(self, output_data: dict[str, Any] | None = None):
        """Mark the trace as finished with optional output data."""
        self.end_time = datetime.datetime.now()
        if output_data:
            self.output_data.update(output_data)

    def add_child(self, child_trace: "BeeAITrace"):
        """Add a child trace."""
        child_trace.parent_trace_id = self.trace_id
        self.child_traces.append(child_trace)

    def to_dict(self) -> dict[str, Any]:
        """Convert trace to dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "metadata": self.metadata,
            "parent_trace_id": self.parent_trace_id,
            "child_traces": [child.to_dict() for child in self.child_traces],
            "duration_ms": (
                (self.end_time - self.start_time).total_seconds() * 1000
                if self.end_time and self.start_time
                else None
            ),
        }


class BeeAITraceContext:
    """Context manager for BeeAI traces."""

    def __init__(self):
        self.traces: list[BeeAITrace] = []
        self.current_trace: BeeAITrace | None = None
        self.trace_stack: list[BeeAITrace] = []

    def start_trace(
        self,
        name: str,
        trace_id: str | None = None,
        input_data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BeeAITrace:
        """Start a new trace."""
        import uuid

        if not trace_id:
            trace_id = str(uuid.uuid4())

        trace = BeeAITrace(
            trace_id=trace_id,
            name=name,
            start_time=datetime.datetime.now(),
            input_data=input_data,
            metadata=metadata,
        )

        if self.current_trace:
            self.current_trace.add_child(trace)
        else:
            self.traces.append(trace)

        self.trace_stack.append(trace)
        self.current_trace = trace
        return trace

    def end_trace(self, output_data: dict[str, Any] | None = None):
        """End the current trace."""
        if self.current_trace:
            self.current_trace.finish(output_data)
            self.trace_stack.pop()
            self.current_trace = self.trace_stack[-1] if self.trace_stack else None

    def get_traces(self) -> list[dict[str, Any]]:
        """Get all traces as dictionaries."""
        return [trace.to_dict() for trace in self.traces]

    def clear(self):
        """Clear all traces."""
        self.traces.clear()
        self.current_trace = None
        self.trace_stack.clear()


class BeeAIPlatformBackend(BeeAIBackend):
    """BeeAI backend with platform integration and trace support."""

    def __init__(
        self,
        model_id: str,
        formatter,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        provider: str | None = None,
        model_options: dict[str, Any] | None = None,
        trace_granularity: str = "generate",  # "none", "generate", "component", "all"
        enable_traces: bool = True,
        trace_output_dir: str | None = None,
    ):
        """Initialize the BeeAI Platform backend.

        Args:
            model_id: The model identifier to use.
            formatter: The formatter to use for converting components to messages.
            api_key: API key for authentication (if required).
            base_url: Base URL for the BeeAI service (if different from default).
            provider: The provider to use (e.g., 'openai', 'anthropic', 'local').
            model_options: Default model options for this backend.
            trace_granularity: Level of tracing detail ("none", "generate", "component", "all").
            enable_traces: Whether to enable trace collection.
            trace_output_dir: Directory to save trace files (default: temp directory).
        """
        # Check if BeeAI framework is available
        if not BEEAI_AVAILABLE:
            FancyLogger.get_logger().warning(
                "BeeAI framework not available. Install with: pip install 'mellea[beeai]'"
            )
            # Fall back to basic FormatterBackend functionality
            super(FormatterBackend, self).__init__(
                model_id=model_id, model_options=model_options or {}
            )
            self.formatter = formatter
        else:
            super().__init__(
                model_id=model_id,
                formatter=formatter,
                api_key=api_key,
                base_url=base_url,
                provider=provider,
                model_options=model_options,
            )

        self.trace_granularity = trace_granularity
        self.enable_traces = enable_traces and trace_granularity != "none"
        self.trace_context = BeeAITraceContext()

        # Set up trace output directory
        if trace_output_dir:
            self.trace_output_dir = Path(trace_output_dir)
        else:
            self.trace_output_dir = Path(tempfile.gettempdir()) / "mellea_traces"
        self.trace_output_dir.mkdir(exist_ok=True)

        FancyLogger.get_logger().info(
            f"BeeAI Platform backend initialized with trace granularity: {trace_granularity}"
        )

    def _should_trace(self, level: str) -> bool:
        """Check if tracing should be enabled for the given level."""
        if not self.enable_traces:
            return False

        if self.trace_granularity == "all":
            return True
        elif self.trace_granularity == "component" and level in [
            "component",
            "generate",
        ]:
            return True
        elif self.trace_granularity == "generate" and level == "generate":
            return True

        return False

    def generate_from_context(
        self,
        action: Component | CBlock,
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict[str, Any] | None = None,
        generate_logs: list[GenerateLog] | None = None,
        tool_calls: bool = False,
    ) -> ModelOutputThunk:
        """Generate a response with trace support."""

        # Start trace if enabled
        trace = None
        if self._should_trace("generate"):
            rendered = (
                ctx.render_for_generation()
                if hasattr(ctx, "render_for_generation")
                else None
            )
            input_data = {
                "action": str(action),
                "context_length": len(rendered) if rendered is not None else 0,
                "model_options": model_options or {},
                "format": format.__name__ if format else None,
                "tool_calls_enabled": tool_calls,
            }

            trace = self.trace_context.start_trace(
                name=f"generate_{self.model_id}",
                input_data=input_data,
                metadata={
                    "backend": f"beeai_platform::{self.model_id}",
                    "provider": getattr(self, "provider", "unknown"),
                    "timestamp": datetime.datetime.now().isoformat(),
                },
            )

        try:
            # Call parent implementation if BeeAI is available
            if BEEAI_AVAILABLE:
                result = super().generate_from_context(
                    action=action,
                    ctx=ctx,
                    format=format,
                    model_options=model_options,
                    generate_logs=generate_logs,
                    tool_calls=tool_calls,
                )
            else:
                # Fallback implementation when BeeAI is not available
                FancyLogger.get_logger().warning(
                    "BeeAI framework not available. Using mock implementation."
                )
                result = ModelOutputThunk(
                    value=f"Mock response for: {action} (BeeAI framework required for actual generation)"
                )

            # End trace with results
            if trace:
                output_data = {
                    "result_value": str(result.value) if result.value else None,
                    "result_length": len(str(result.value)) if result.value else 0,
                    "parsed_repr": str(result.parsed_repr)
                    if result.parsed_repr
                    else None,
                    "tool_calls": list(result.tool_calls.keys())
                    if result.tool_calls
                    else [],
                }
                self.trace_context.end_trace(output_data)

            return result

        except Exception as e:
            # End trace with error
            if trace:
                self.trace_context.end_trace(
                    {"error": str(e), "error_type": type(e).__name__}
                )
            raise

    def _generate_from_raw(
        self,
        actions: list[Component | CBlock],
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict[str, Any] | None = None,
        generate_logs: list[GenerateLog] | None = None,
    ) -> list[ModelOutputThunk]:
        """Generate responses from raw actions with optional BeeAI backend."""
        if BEEAI_AVAILABLE:
            return super()._generate_from_raw(
                actions=actions,
                format=format,
                model_options=model_options,
                generate_logs=generate_logs,
            )
        else:
            # Fallback implementation
            FancyLogger.get_logger().warning(
                "BeeAI framework not available. Using mock implementation."
            )
            return [
                ModelOutputThunk(
                    value=f"Mock response for: {action} (BeeAI framework required)"
                )
                for action in actions
            ]

    def save_traces(self, filename: str | None = None) -> str:
        """Save current traces to a file."""
        if not filename:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mellea_traces_{timestamp}.json"

        filepath = self.trace_output_dir / filename

        traces_data = {
            "version": "1.0",
            "timestamp": datetime.datetime.now().isoformat(),
            "backend": f"beeai_platform::{self.model_id}",
            "provider": getattr(self, "provider", "unknown"),
            "trace_granularity": self.trace_granularity,
            "traces": self.trace_context.get_traces(),
        }

        with open(filepath, "w") as f:
            json.dump(traces_data, f, indent=2, default=str)

        FancyLogger.get_logger().info(f"Traces saved to: {filepath}")
        return str(filepath)

    def clear_traces(self):
        """Clear all collected traces."""
        self.trace_context.clear()

    def get_trace_summary(self) -> dict[str, Any]:
        """Get a summary of collected traces."""
        traces = self.trace_context.get_traces()

        return {
            "total_traces": len(traces),
            "trace_granularity": self.trace_granularity,
            "traces_enabled": self.enable_traces,
            "output_directory": str(self.trace_output_dir),
        }


def create_beeai_agent_manifest(
    mellea_program: str,
    agent_name: str,
    description: str,
    version: str = "1.0.0",
    output_dir: str | None = None,
) -> str:
    """Create a BeeAI agent manifest for a Mellea program.

    Args:
        mellea_program: Path to the Mellea program file.
        agent_name: Name of the agent.
        description: Description of the agent.
        version: Version of the agent.
        output_dir: Directory to save the manifest (default: same as program).

    Returns:
        Path to the created manifest file.
    """
    program_path = Path(mellea_program)
    if not program_path.exists():
        raise FileNotFoundError(f"Mellea program not found: {mellea_program}")

    if output_dir:
        manifest_dir = Path(output_dir)
    else:
        manifest_dir = program_path.parent

    manifest_dir.mkdir(exist_ok=True)

    manifest = {
        "name": agent_name,
        "description": description,
        "version": version,
        "type": "mellea_agent",
        "runtime": {
            "type": "python",
            "entry_point": str(program_path.name),
            "requirements": ["mellea"],
        },
        "endpoints": {
            "chat": {
                "path": "/chat",
                "method": "POST",
                "description": "Chat with the Mellea agent",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"},
                        "session_id": {"type": "string", "optional": True},
                        "model_options": {"type": "object", "optional": True},
                    },
                    "required": ["message"],
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "response": {"type": "string"},
                        "session_id": {"type": "string"},
                        "traces": {"type": "array", "optional": True},
                    },
                },
            }
        },
        "capabilities": ["chat", "traces", "session_management", "mellea"],
        "metadata": {
            "framework": "mellea",
            "created_at": datetime.datetime.now().isoformat(),
            "mellea_program": str(program_path.absolute()),
        },
    }

    manifest_path = manifest_dir / f"{agent_name}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    FancyLogger.get_logger().info(f"BeeAI agent manifest created: {manifest_path}")
    return str(manifest_path)


def start_beeai_platform(
    port: int = 8080, host: str = "localhost", background: bool = False
) -> subprocess.Popen | None:
    """Start a local BeeAI platform instance.

    Args:
        port: Port to run the platform on.
        host: Host to bind to.
        background: Whether to run in background.

    Returns:
        Process object if running in background, None otherwise.
    """
    try:
        # Check if beeai CLI is available
        subprocess.run(["beeai", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        FancyLogger.get_logger().error(
            "BeeAI CLI not found. Install with: uv tool install beeai-cli"
        )
        raise RuntimeError("BeeAI CLI not available")

    cmd = ["beeai", "platform", "start", "--host", host, "--port", str(port)]

    FancyLogger.get_logger().info(f"Starting BeeAI platform on {host}:{port}")

    if background:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        FancyLogger.get_logger().info(
            f"BeeAI platform started in background (PID: {process.pid})"
        )
        return process
    else:
        subprocess.run(cmd)
        return None
