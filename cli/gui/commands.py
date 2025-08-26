"""CLI commands for BeeAI Platform GUI integration."""

import os
import time
import signal
import sys
from pathlib import Path
from typing import Optional

import typer
from mellea.helpers.fancy_logger import FancyLogger
from mellea.backends.beeai_platform import (
    start_beeai_platform,
    create_beeai_agent_manifest,
    BeeAIPlatformBackend,
)

gui_app = typer.Typer(name="gui", help="BeeAI Platform GUI commands")


@gui_app.command()
def chat(
    script_path: Optional[str] = typer.Argument(
        None,
        help="Path to the Mellea program to serve (optional)"
    ),
    port: int = typer.Option(8080, help="Port to run the BeeAI platform on"),
    host: str = typer.Option("localhost", help="Host to bind to"),
    auto_manifest: bool = typer.Option(
        True,
        help="Automatically create agent manifest for the script"
    ),
    trace_granularity: str = typer.Option(
        "generate",
        help="Trace granularity level (none, generate, component, all)"
    ),
):
    """Start a local BeeAI Platform instance with chat interface.
    
    This command spins up a local BeeAI platform instance that provides a web-based
    chat interface for interacting with Mellea programs. If a script path is provided,
    it will automatically create an agent manifest and configure the platform.
    """
    
    logger = FancyLogger.get_logger()
    
    # Validate trace granularity
    valid_granularities = ["none", "generate", "component", "all"]
    if trace_granularity not in valid_granularities:
        typer.echo(
            f"Error: Invalid trace granularity '{trace_granularity}'. "
            f"Must be one of: {', '.join(valid_granularities)}"
        )
        raise typer.Exit(1)
    
    try:
        # Check if BeeAI CLI is available
        import subprocess
        subprocess.run(["beeai", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        typer.echo("❌ BeeAI CLI not found.")
        typer.echo("📦 Install with: uv tool install beeai-cli")
        typer.echo("📖 See: https://docs.beeai.dev for more information")
        raise typer.Exit(1)
    
    # Create agent manifest if script is provided
    if script_path:
        script_path_obj = Path(script_path)
        if not script_path_obj.exists():
            typer.echo(f"❌ Script not found: {script_path}")
            raise typer.Exit(1)
        
        if auto_manifest:
            try:
                agent_name = script_path_obj.stem
                manifest_path = create_beeai_agent_manifest(
                    mellea_program=script_path,
                    agent_name=agent_name,
                    description=f"Mellea agent: {agent_name}",
                    version="1.0.0",
                )
                typer.echo(f"✅ Created agent manifest: {manifest_path}")
            except Exception as e:
                logger.warning(f"Failed to create agent manifest: {e}")
                typer.echo(f"⚠️  Warning: Could not create agent manifest: {e}")
    
    # Display startup information
    typer.echo("🚀 Starting BeeAI Platform...")
    typer.echo(f"🌐 Host: {host}")
    typer.echo(f"🔌 Port: {port}")
    typer.echo(f"📊 Trace granularity: {trace_granularity}")
    
    if script_path:
        typer.echo(f"📜 Mellea script: {script_path}")
    
    typer.echo(f"🖥️  Platform will be available at: http://{host}:{port}")
    typer.echo("🎯 Web UI will be available at: http://{host}:{port}/ui")
    typer.echo("\n💡 Tip: Use Ctrl+C to stop the platform")
    typer.echo("📖 Documentation: https://docs.beeai.dev")
    typer.echo("\n" + "="*50)
    
    # Set up signal handler for graceful shutdown
    def signal_handler(signum, frame):
        typer.echo("\n\n🛑 Shutting down BeeAI Platform...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start the BeeAI platform
        start_beeai_platform(port=port, host=host, background=False)
    except KeyboardInterrupt:
        typer.echo("\n🛑 BeeAI Platform stopped by user")
    except Exception as e:
        logger.error(f"Failed to start BeeAI platform: {e}")
        typer.echo(f"❌ Error: {e}")
        raise typer.Exit(1)


@gui_app.command()
def status():
    """Check the status of BeeAI Platform installation and configuration."""
    
    typer.echo("🔍 Checking BeeAI Platform status...\n")
    
    # Check BeeAI CLI installation
    try:
        import subprocess
        result = subprocess.run(
            ["beeai", "--version"], 
            check=True, 
            capture_output=True, 
            text=True
        )
        typer.echo(f"✅ BeeAI CLI installed: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        typer.echo("❌ BeeAI CLI not found")
        typer.echo("📦 Install with: uv tool install beeai-cli")
        return
    
    # Check if platform is running
    try:
        import requests
        response = requests.get("http://localhost:8080/health", timeout=2)
        if response.status_code == 200:
            typer.echo("✅ BeeAI Platform is running on localhost:8080")
        else:
            typer.echo("⚠️  BeeAI Platform responded with non-200 status")
    except requests.RequestException:
        typer.echo("❌ BeeAI Platform is not running on localhost:8080")
    
    # Check Mellea BeeAI backend
    try:
        from mellea.backends.beeai import BeeAIBackend
        typer.echo("✅ Mellea BeeAI backend available")
    except ImportError as e:
        typer.echo(f"❌ Mellea BeeAI backend not available: {e}")
    
    # Check BeeAI Platform backend
    try:
        from mellea.backends.beeai_platform import BeeAIPlatformBackend
        typer.echo("✅ Mellea BeeAI Platform backend available")
    except ImportError as e:
        typer.echo(f"❌ Mellea BeeAI Platform backend not available: {e}")
    
    typer.echo("\n📖 For more information, visit: https://docs.beeai.dev")


@gui_app.command()
def manifest(
    script_path: str = typer.Argument(..., help="Path to the Mellea program"),
    agent_name: Optional[str] = typer.Option(
        None, 
        help="Name for the agent (default: script filename)"
    ),
    description: Optional[str] = typer.Option(
        None,
        help="Description of the agent"
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        help="Output directory for the manifest"
    ),
    version: str = typer.Option("1.0.0", help="Version of the agent"),
):
    """Create a BeeAI agent manifest for a Mellea program.
    
    This command generates a manifest file that describes how to run a Mellea
    program as a BeeAI agent, enabling it to be discovered and used within
    the BeeAI Platform ecosystem.
    """
    
    script_path_obj = Path(script_path)
    if not script_path_obj.exists():
        typer.echo(f"❌ Script not found: {script_path}")
        raise typer.Exit(1)
    
    # Use defaults if not provided
    if not agent_name:
        agent_name = script_path_obj.stem
    
    if not description:
        description = f"Mellea agent based on {script_path_obj.name}"
    
    try:
        manifest_path = create_beeai_agent_manifest(
            mellea_program=script_path,
            agent_name=agent_name,
            description=description,
            version=version,
            output_dir=output_dir,
        )
        
        typer.echo(f"✅ Agent manifest created successfully!")
        typer.echo(f"📄 Manifest file: {manifest_path}")
        typer.echo(f"🤖 Agent name: {agent_name}")
        typer.echo(f"📝 Description: {description}")
        typer.echo(f"🏷️  Version: {version}")
        
        typer.echo("\n💡 Next steps:")
        typer.echo("1. Start BeeAI Platform: m gui chat")
        typer.echo(f"2. Register your agent with the manifest file")
        typer.echo("3. Access the web UI to interact with your agent")
        
    except Exception as e:
        typer.echo(f"❌ Failed to create manifest: {e}")
        raise typer.Exit(1)


@gui_app.command()
def install():
    """Install BeeAI CLI if not already installed."""
    
    typer.echo("📦 Installing BeeAI CLI...")
    
    try:
        import subprocess
        
        # Check if already installed
        try:
            result = subprocess.run(
                ["beeai", "--version"], 
                check=True, 
                capture_output=True, 
                text=True
            )
            typer.echo(f"✅ BeeAI CLI already installed: {result.stdout.strip()}")
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # Try to install using uv
        try:
            subprocess.run(["uv", "--version"], check=True, capture_output=True)
            typer.echo("📦 Installing BeeAI CLI using uv...")
            subprocess.run(["uv", "tool", "install", "beeai-cli"], check=True)
            typer.echo("✅ BeeAI CLI installed successfully!")
        except (subprocess.CalledProcessError, FileNotFoundError):
            typer.echo("❌ uv not found. Please install uv first:")
            typer.echo("   curl -LsSf https://astral.sh/uv/install.sh | sh")
            typer.echo("   Or visit: https://docs.astral.sh/uv/getting-started/installation/")
            raise typer.Exit(1)
        
        # Verify installation
        result = subprocess.run(
            ["beeai", "--version"], 
            check=True, 
            capture_output=True, 
            text=True
        )
        typer.echo(f"🎉 Installation verified: {result.stdout.strip()}")
        
        typer.echo("\n💡 Next steps:")
        typer.echo("1. Start BeeAI Platform: m gui chat")
        typer.echo("2. Configure LLM provider: beeai env setup")
        typer.echo("3. Access web UI when platform starts")
        
    except subprocess.CalledProcessError as e:
        typer.echo(f"❌ Installation failed: {e}")
        typer.echo("📖 Manual installation instructions: https://docs.beeai.dev/getting-started/installation/")
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"❌ Unexpected error: {e}")
        raise typer.Exit(1)
