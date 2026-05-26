"""CLI for fixing async calls after top-level ainstruct, aquery, and aact contract change.."""

from enum import StrEnum

import typer

fix_app = typer.Typer(name="fix", help="Fix code for API changes.")


class _FixMode(StrEnum):
    """Types of fixes that can be applied."""

    ADD_AWAIT_RESULT = "add-await-result"
    ADD_STREAM_LOOP = "add-stream-loop"


from cli.fix.commands import fix_async, fix_genslots  # noqa: E402

fix_app.command("async")(fix_async)
fix_app.command("genslots")(fix_genslots)
