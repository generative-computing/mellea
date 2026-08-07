# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for postponed annotation (PEP 563) resolution in tool schemas.

Covers `convert_function_to_ollama_tool`'s handling of `from __future__
import annotations`: non-builtin parameter types must resolve to real type
objects for Pydantic schema building, while a return annotation that is
unresolvable at call time (e.g. `TYPE_CHECKING`-only imports) must not break
the conversion, since the return annotation is never consumed by the
produced schema.
"""

import pydantic
import pytest

from mellea.backends.tools import convert_function_to_ollama_tool
from test.backends._pep563_samples import (
    Address,
    Period,
    send_letter,
    tc_only_return_builtin_param,
    tc_return_custom_param,
    unresolvable_param,
)


def test_convert_function_to_ollama_tool_resolves_postponed_annotations():
    # Regression test: under `from __future__ import annotations`, a
    # non-builtin parameter type's annotation is a string rather than the
    # real type object, which Pydantic cannot resolve when building the
    # dynamic schema model - raising PydanticUserError instead of producing
    # a tool schema.

    # Guard the precondition: if the sample module ever drops its
    # `from __future__ import annotations`, this test would otherwise keep
    # passing without exercising postponed annotations at all.
    assert send_letter.__annotations__["to"] == "Address"

    tool = convert_function_to_ollama_tool(send_letter)
    assert tool.function is not None
    assert tool.function.parameters is not None

    props = tool.function.parameters.model_dump(exclude_none=True)["properties"]
    assert props["to"]["type"] == "object"
    assert props["to"]["title"] == Address.__name__
    assert props["to"]["properties"]["city"]["type"] == "string"


def test_convert_function_to_ollama_tool_tc_only_return():
    """TYPE_CHECKING-only return + builtin params must produce schema."""
    tool = convert_function_to_ollama_tool(tc_only_return_builtin_param)
    assert tool.function is not None
    assert tool.function.parameters is not None
    props = tool.function.parameters.model_dump(exclude_none=True)["properties"]
    assert "query" in props


def test_convert_function_to_ollama_tool_tc_return_custom_param():
    """TYPE_CHECKING return + custom param must resolve param.

    This is the case that separates the try/except-around-`eval_str=True`
    fallback (which discards parameter resolution entirely on any
    failure) from per-parameter resolution: the return annotation is
    unresolvable, but the custom parameter type still must resolve.
    """
    tool = convert_function_to_ollama_tool(tc_return_custom_param)
    assert tool.function is not None
    assert tool.function.parameters is not None
    props = tool.function.parameters.model_dump(exclude_none=True)["properties"]
    assert props["period"]["type"] == "object"
    assert props["period"]["title"] == Period.__name__


def test_convert_function_to_ollama_tool_unresolvable_param():
    """Genuinely unresolvable parameter must still raise PydanticUserError.

    Confirms the documented degradation: unresolvable param annotations
    surface as `PydanticUserError`, not `NameError`.
    """
    with pytest.raises(pydantic.PydanticUserError):
        convert_function_to_ollama_tool(unresolvable_param)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
