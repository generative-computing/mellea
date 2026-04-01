"""Unit tests for TokenMetricsPlugin."""

from unittest.mock import patch

import pytest

pytest.importorskip("cpex", reason="cpex not installed — install mellea[hooks]")

from mellea.core.base import ModelOutputThunk
from mellea.plugins.hooks.generation import GenerationPostCallPayload
from mellea.telemetry.metrics_plugins import TokenMetricsPlugin


@pytest.fixture
def plugin():
    return TokenMetricsPlugin()


def _make_payload(usage=None, model="test-model", provider="test-provider"):
    """Create a GenerationPostCallPayload with a ModelOutputThunk."""
    mot = ModelOutputThunk(value="hello")
    mot.usage = usage
    mot.model = model
    mot.provider = provider
    return GenerationPostCallPayload(model_output=mot)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "usage,expected_input,expected_output",
    [
        ({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}, 10, 5),
        ({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, 0, 0),
    ],
    ids=["normal-usage", "zero-tokens"],
)
async def test_record_token_metrics_with_usage(
    plugin, usage, expected_input, expected_output
):
    """Test that metrics are recorded when usage is present."""
    payload = _make_payload(usage=usage)

    with patch("mellea.telemetry.metrics.record_token_usage_metrics") as mock_record:
        await plugin.record_token_metrics(payload, {})

        mock_record.assert_called_once_with(
            input_tokens=expected_input,
            output_tokens=expected_output,
            model="test-model",
            provider="test-provider",
        )


@pytest.mark.asyncio
async def test_record_token_metrics_no_usage(plugin):
    """Test that metrics are not recorded when usage is None."""
    payload = _make_payload(usage=None)

    with patch("mellea.telemetry.metrics.record_token_usage_metrics") as mock_record:
        await plugin.record_token_metrics(payload, {})

        mock_record.assert_not_called()


@pytest.mark.asyncio
async def test_record_token_metrics_missing_model_provider(plugin):
    """Test fallback to 'unknown' when model/provider are None."""
    payload = _make_payload(
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        model=None,
        provider=None,
    )

    with patch("mellea.telemetry.metrics.record_token_usage_metrics") as mock_record:
        await plugin.record_token_metrics(payload, {})

        mock_record.assert_called_once_with(
            input_tokens=10, output_tokens=5, model="unknown", provider="unknown"
        )
