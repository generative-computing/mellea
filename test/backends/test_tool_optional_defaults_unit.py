# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for tool schema serialization of optional/defaulted parameters.

Regression tests for the bug where parameters with default values were losing
their optional status (incorrectly listed as required) and their default values
during conversion to the OpenAI-compatible tool schema. The conversion is shared
by every backend via `convert_tools_to_json`, so a fix here applies to all of
them (OpenAI, LiteLLM, HF, Watsonx, Ollama).

See: https://github.com/generative-computing/mellea/issues/1569
"""

from mellea.backends.tools import (
    MelleaTool,
    convert_function_to_ollama_tool,
    convert_tools_to_json,
)


def _params(func, name=None):
    """Return the serialized `parameters` block for a callable."""
    schema = convert_function_to_ollama_tool(func, name or func.__name__).model_dump(
        exclude_none=True
    )
    return schema["function"]["parameters"]


def test_defaulted_param_not_required():
    """A parameter with a default value must not appear in `required`."""

    def get_weather(location: str, units: str = "celsius") -> dict:
        """Get weather.

        Args:
            location: City name
            units: Temperature units
        """
        return {}

    params = _params(get_weather)

    assert "location" in params["required"]
    assert "units" not in params["required"], (
        "Parameter with a default value should not be required"
    )


def test_defaulted_param_preserves_default_value():
    """Default values must be carried into the serialized schema."""

    def get_weather(location: str, units: str = "celsius", days: int = 1) -> dict:
        """Get weather.

        Args:
            location: City name
            units: Temperature units
            days: Number of days
        """
        return {}

    props = _params(get_weather)["properties"]

    assert props["units"].get("default") == "celsius"
    assert props["days"].get("default") == 1


def test_defaulted_param_correct_type():
    """Defaulted simple params keep a scalar type, not an anyOf structure."""

    def get_weather(location: str, units: str = "celsius", days: int = 1) -> dict:
        """Get weather.

        Args:
            location: City name
            units: Temperature units
            days: Number of days
        """
        return {}

    props = _params(get_weather)["properties"]

    assert props["units"]["type"] == "string"
    assert "anyOf" not in props["units"]
    assert props["days"]["type"] == "integer"
    assert "anyOf" not in props["days"]


def test_required_param_has_no_default():
    """A parameter with no default must be required and carry no default key."""

    def get_weather(location: str, units: str = "celsius") -> dict:
        """Get weather.

        Args:
            location: City name
            units: Temperature units
        """
        return {}

    params = _params(get_weather)

    assert "location" in params["required"]
    assert "default" not in params["properties"]["location"]


def test_optional_typed_with_default_value():
    """`x: str | None = "hi"` is optional, keeps its default, and is a scalar."""

    def process(x: str, y: str | None = "hi") -> str:
        """Process text.

        Args:
            x: Required text
            y: Optional text
        """
        return f"{x} {y}"

    params = _params(process)

    assert "y" not in params["required"]
    y_prop = params["properties"]["y"]
    assert y_prop["type"] == "string"
    assert "anyOf" not in y_prop
    assert y_prop.get("default") == "hi"


def test_falsy_defaults_are_preserved():
    """Falsy defaults (0, "", False) must still be emitted and be optional."""

    def configure(count: int = 0, label: str = "", enabled: bool = False) -> dict:
        """Configure.

        Args:
            count: A count
            label: A label
            enabled: A flag
        """
        return {}

    params = _params(configure)
    props = params["properties"]

    assert params.get("required", []) == []
    assert props["count"].get("default") == 0
    assert props["label"].get("default") == ""
    assert props["enabled"].get("default") is False


def test_no_default_params_unchanged():
    """Regression: an all-required signature keeps every param required."""

    def add(a: int, b: int) -> int:
        """Add two numbers.

        Args:
            a: First number
            b: Second number
        """
        return a + b

    params = _params(add)

    assert set(params["required"]) == {"a", "b"}
    assert "default" not in params["properties"]["a"]
    assert "default" not in params["properties"]["b"]


def test_union_type_serialized_as_list():
    """A multi-type union emits a JSON Schema type list, not a comma-joined string."""

    def pick(value: str | int) -> dict:
        """Pick.

        Args:
            value: A string or integer
        """
        return {}

    props = _params(pick)["properties"]

    assert props["value"]["type"] == ["integer", "string"]


def test_single_type_serialized_as_scalar():
    """A single-type param keeps a scalar type string (not a one-element list)."""

    def echo(value: str) -> dict:
        """Echo.

        Args:
            value: A string
        """
        return {}

    props = _params(echo)["properties"]

    assert props["value"]["type"] == "string"


def test_optional_union_drops_null_and_lists_remaining():
    """`str | int | None` is optional and serializes the non-null types as a list."""

    def pick(value: str | int | None = None) -> dict:
        """Pick.

        Args:
            value: A string, integer, or nothing
        """
        return {}

    params = _params(pick)

    assert "value" not in params.get("required", [])
    assert params["properties"]["value"]["type"] == ["integer", "string"]


def test_complex_param_default_preserved():
    """A complex (Pydantic model) parameter's default survives $ref inlining."""
    from pydantic import BaseModel

    class Config(BaseModel):
        verbose: bool = False

    def run(cfg: Config = Config()) -> dict:
        """Run.

        Args:
            cfg: Configuration object
        """
        return {}

    props = _params(run)["properties"]

    assert props["cfg"].get("default") == {"verbose": False}
    assert "cfg" not in _params(run).get("required", [])


def test_mixed_signature_via_convert_tools_to_json():
    """End-to-end through the path every backend uses (`convert_tools_to_json`)."""

    def get_weather(location: str, units: str = "celsius", days: int = 1) -> dict:
        """Get weather.

        Args:
            location: City name
            units: Temperature units
            days: Number of days
        """
        return {}

    tool = MelleaTool.from_callable(get_weather)
    serialized = convert_tools_to_json({"get_weather": tool})

    params = serialized[0]["function"]["parameters"]
    props = params["properties"]

    assert params["required"] == ["location"]
    assert props["units"].get("default") == "celsius"
    assert props["days"].get("default") == 1
