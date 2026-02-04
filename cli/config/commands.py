"""CLI commands for Mellea configuration management."""

from pathlib import Path

import typer
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from mellea.config import find_config_file, init_project_config, load_config

config_app = typer.Typer(name="config", help="Manage Mellea configuration files")
console = Console()


@config_app.command("init")
def init_project(
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite existing config file"
    ),
) -> None:
    """Create a project configuration file at ./mellea.toml."""
    try:
        config_path = init_project_config(force=force)
        console.print(f"[green]✓[/green] Created project config at: {config_path}")
        console.print("\nEdit this file to set your backend, model, and other options.")
        console.print(
            "Run [cyan]m config show[/cyan] to view the current configuration."
        )
    except FileExistsError as e:
        console.print(f"[red]✗[/red] {e}")
        console.print("Use [cyan]--force[/cyan] to overwrite the existing file.")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]✗[/red] Error creating config: {e}")
        raise typer.Exit(1)


@config_app.command("show")
def show_config() -> None:
    """Display the current effective configuration."""
    try:
        config, config_path = load_config()

        # Display config source
        if config_path:
            console.print(f"[bold]Configuration loaded from:[/bold] {config_path}\n")
        else:
            console.print(
                "[yellow]No configuration file found. Using defaults.[/yellow]\n"
            )

        # Create a table for the configuration
        table = Table(
            title="Effective Configuration", show_header=True, header_style="bold cyan"
        )
        table.add_column("Setting", style="dim")
        table.add_column("Value")

        # Backend settings
        table.add_row(
            "Backend Name", config.backend.name or "[dim](default: ollama)[/dim]"
        )
        table.add_row(
            "Model ID",
            config.backend.model_id or "[dim](default: granite-4-micro:3b)[/dim]",
        )

        # Model options
        if config.backend.model_options:
            for key, value in config.backend.model_options.items():
                table.add_row(f"  {key}", str(value))

        # Backend kwargs
        if config.backend.kwargs:
            for key, value in config.backend.kwargs.items():
                table.add_row(f"  backend.{key}", str(value))

        # Credentials (masked)
        if config.credentials.openai_api_key:
            table.add_row("OpenAI API Key", "[dim]***configured***[/dim]")
        if config.credentials.watsonx_api_key:
            table.add_row("Watsonx API Key", "[dim]***configured***[/dim]")
        if config.credentials.watsonx_project_id:
            table.add_row("Watsonx Project ID", config.credentials.watsonx_project_id)
        if config.credentials.watsonx_url:
            table.add_row("Watsonx URL", config.credentials.watsonx_url)

        # General settings
        table.add_row(
            "Context Type", config.context_type or "[dim](default: simple)[/dim]"
        )
        table.add_row("Log Level", config.log_level or "[dim](default: INFO)[/dim]")

        console.print(table)

        console.print(
            "\n[dim]Explicit parameters in code override config file values.[/dim]"
        )

    except Exception as e:
        console.print(f"[red]✗[/red] Error loading config: {e}")
        raise typer.Exit(1)


@config_app.command("path")
def show_path() -> None:
    """Show the path to the currently loaded configuration file."""
    try:
        config_path = find_config_file()

        if config_path:
            console.print(f"[green]✓[/green] Using config file: {config_path}")

            # Show the file content
            console.print("\n[bold]File contents:[/bold]")
            with open(config_path) as f:
                content = f.read()
            syntax = Syntax(content, "toml", theme="monokai", line_numbers=True)
            console.print(syntax)
        else:
            console.print("[yellow]No configuration file found.[/yellow]")
            console.print("\nSearched: ./mellea.toml (current dir and parents)")
            console.print(
                "\nRun [cyan]m config init[/cyan] to create a project config."
            )
    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}")
        raise typer.Exit(1)


@config_app.command("where")
def show_locations() -> None:
    """Show configuration file location."""
    project_config_path = Path.cwd() / "mellea.toml"

    console.print("[bold]Configuration file location:[/bold]\n")

    # Project config
    console.print(f"[cyan]Project config:[/cyan] {project_config_path}")
    if project_config_path.exists():
        console.print("  [green]✓ exists[/green]")
    else:
        console.print("  [dim]✗ not found[/dim]")
        console.print("  Run [cyan]m config init[/cyan] to create")

    console.print()

    # Currently loaded (might be in parent dir)
    current = find_config_file()
    if current:
        console.print(f"[bold green]Currently loaded:[/bold green] {current}")
        if current != project_config_path:
            console.print("  [dim](found in parent directory)[/dim]")
    else:
        console.print(
            "[yellow]No config file currently loaded (using defaults)[/yellow]"
        )
