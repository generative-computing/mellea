# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `m serve` CLI command definition (cli/serve/commands.py)."""

import re
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from cli.m import cli

runner = CliRunner()

# Rich colorizes CLI error output (e.g. option names) with ANSI codes when a
# color-forcing env var like FORCE_COLOR is set, so strip them before
# asserting on the message text.
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(output: str) -> str:
    """Strip ANSI escape codes from CLI runner output."""
    return _ANSI_ESCAPE.sub("", output)


class TestResponseTtlValidation:
    """`--response-ttl` must be a positive integer."""

    @pytest.mark.parametrize("ttl", ["0", "-60"])
    def test_non_positive_ttl_rejected(self, ttl):
        """A zero or negative ttl is rejected with a usage error and never starts the server."""
        with patch("cli.serve.app.run_server") as mock_run_server:
            result = runner.invoke(cli, ["serve", "my_app.py", "--response-ttl", ttl])

        assert result.exit_code == 2
        assert "--response-ttl" in _plain(result.output)
        mock_run_server.assert_not_called()

    @pytest.mark.parametrize("ttl", [1, 3600])
    def test_positive_ttl_accepted(self, ttl):
        """A positive ttl, including the minimum of 1, is passed to run_server unchanged."""
        with patch("cli.serve.app.run_server") as mock_run_server:
            result = runner.invoke(
                cli, ["serve", "my_app.py", "--response-ttl", str(ttl)]
            )

        assert result.exit_code == 0
        mock_run_server.assert_called_once()
        assert mock_run_server.call_args.kwargs["response_ttl"] == ttl

    def test_default_ttl_accepted(self):
        """Omitting --response-ttl keeps the 30-minute default."""
        with patch("cli.serve.app.run_server") as mock_run_server:
            result = runner.invoke(cli, ["serve", "my_app.py"])

        assert result.exit_code == 0
        mock_run_server.assert_called_once()
        assert mock_run_server.call_args.kwargs["response_ttl"] == 1800
