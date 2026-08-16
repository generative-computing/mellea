# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI for applying automated code migrations after Mellea API changes."""

import typer

from cli.fix.commands import fix_genslots

fix_app = typer.Typer(name="fix", help="Fix code for API changes.")

fix_app.command("genslots")(fix_genslots)
