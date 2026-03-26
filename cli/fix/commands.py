"""CLI command for `m fix async`."""

from pathlib import Path

import typer

from cli.fix import _FixMode


def fix_async(
    path: str = typer.Argument(..., help="File or directory to scan"),
    mode: _FixMode = typer.Option(
        _FixMode.ADD_AWAIT_RESULT, "--mode", "-m", help="Fix strategy to apply"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report locations without modifying files"
    ),
):
    """Fix async calls (aact, ainstruct, aquery) for the await_result default change.

    Args:
        path: File or directory to scan.
        mode: Fix strategy to apply.
        dry_run: If ``True``, report locations without modifying files.

    Raises:
        typer.Exit: If *path* does not exist.

    \b
    Modes:
      add-await-result  (default) Adds await_result=True to each call so it
                        blocks until the result is ready. Use this if you don't
                        need to stream partial results.
      add-stream-loop   Inserts a `while not r.is_computed(): await r.astream()`
                        loop after each call. This only works if you passed a
                        streaming model option (e.g. stream=True) to the call;
                        otherwise the loop will finish immediately.

    \b
    Best practices:
      - Run with --dry-run first to review what will be changed.
      - Only run a given mode once per file. The tool detects prior fixes and
        skips calls that already have await_result=True or a stream loop, but
        it is safest to treat it as a one-shot migration.
      - Do not run both modes on the same file. If a stream loop is already
        present, add-await-result will skip that call (and vice versa).

    \b
    Detection notes:
      - Most import styles are detected: `import mellea`,
        `from mellea import MelleaSession`,
        `from mellea.stdlib.functional import aact`, module aliases, etc.
      - Calls that are already followed by `await r.avalue()`,
        `await r.astream()`, or a `while not r.is_computed()` loop are
        automatically skipped, even when nested inside if/try/for blocks.
    """
    from cli.fix.fixer import fix_path

    target = Path(path)
    if not target.exists():
        typer.echo(f"Error: {path} does not exist", err=True)
        raise typer.Exit(code=1)

    result = fix_path(target, mode, dry_run=dry_run)

    if result.total_fixes == 0:
        typer.echo("No fixable calls found.")
        return

    action = "Found" if dry_run else "Fixed"
    typer.echo(
        f"{action} {result.total_fixes} call(s) in {result.files_affected} file(s):"
    )
    for loc in result.locations:
        typer.echo(
            f"  {loc.filepath}:{loc.line} - {loc.function_name}() [{loc.call_style}]"
        )
