# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unsupported variadic tool signatures must fail before model invocation."""

import pytest

from mellea.backends.tools import MelleaTool, convert_function_to_ollama_tool


@pytest.mark.parametrize("signature", ["args", "kwargs"])
def test_variadic_tool_signature_rejected(signature):
    def positional(*args: int) -> int:
        """Count positional inputs."""
        return len(args)

    def keyword(**kwargs: int) -> int:
        """Count keyword inputs."""
        return len(kwargs)

    func = positional if signature == "args" else keyword
    with pytest.raises(ValueError, match=r"Variadic parameters.*not supported"):
        convert_function_to_ollama_tool(func)
    with pytest.raises(ValueError, match=r"Variadic parameters.*not supported"):
        MelleaTool.from_callable(func)
