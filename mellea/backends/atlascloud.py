# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Atlas Cloud OpenAI-compatible backend helpers."""

from __future__ import annotations

import os
from typing import Final

from mellea.backends.openai import OpenAIBackend

ATLASCLOUD_DEFAULT_BASE_URL: Final = "https://api.atlascloud.ai/v1"
ATLASCLOUD_DEFAULT_MODEL: Final = "qwen/qwen3.5-flash"
ATLASCLOUD_REASONING_MODEL: Final = "deepseek-ai/deepseek-v4-pro"


class AtlasCloudBackend(OpenAIBackend):
    """OpenAI-compatible backend preconfigured for Atlas Cloud.

    Args:
        model_id: Atlas Cloud model identifier.
        base_url: Atlas Cloud OpenAI-compatible endpoint. Defaults to
            `ATLASCLOUD_API_BASE`, `ATLASCLOUD_BASE_URL`, or
            `https://api.atlascloud.ai/v1`.
        api_key: Atlas Cloud API key. Defaults to `ATLASCLOUD_API_KEY`.
        kwargs: Additional keyword arguments forwarded to `OpenAIBackend`.

    Raises:
        ValueError: If no API key is passed and `ATLASCLOUD_API_KEY` is unset.
    """

    def __init__(
        self,
        model_id: str = ATLASCLOUD_DEFAULT_MODEL,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        **kwargs,
    ) -> None:
        """Initialize an Atlas Cloud backend."""
        resolved_api_key = api_key or os.getenv("ATLASCLOUD_API_KEY")
        if resolved_api_key is None:
            raise ValueError(
                "ATLASCLOUD_API_KEY or api_key is required but not set. Please either:\n"
                "  1. Set the environment variable: export ATLASCLOUD_API_KEY='your-key-here'\n"
                "  2. Pass it as a parameter: AtlasCloudBackend(api_key='your-key-here')"
            )

        resolved_base_url = (
            base_url
            or os.getenv("ATLASCLOUD_API_BASE")
            or os.getenv("ATLASCLOUD_BASE_URL")
            or ATLASCLOUD_DEFAULT_BASE_URL
        )

        super().__init__(
            model_id=model_id,
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            **kwargs,
        )
        self._provider = "atlascloud"


def create_atlascloud_openai_backend(
    model_id: str = ATLASCLOUD_DEFAULT_MODEL,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    **kwargs,
) -> AtlasCloudBackend:
    """Return an OpenAI-compatible backend configured for Atlas Cloud.

    Args:
        model_id: Atlas Cloud model identifier.
        base_url: Optional endpoint override. Defaults to `ATLASCLOUD_API_BASE`,
            `ATLASCLOUD_BASE_URL`, or `https://api.atlascloud.ai/v1`.
        api_key: Optional API key override. Defaults to `ATLASCLOUD_API_KEY`.
        kwargs: Additional keyword arguments forwarded to `AtlasCloudBackend`.

    Returns:
        AtlasCloudBackend: A backend configured with Atlas Cloud credentials and
        endpoint defaults.

    Raises:
        ValueError: If no API key is passed and `ATLASCLOUD_API_KEY` is unset.
    """
    return AtlasCloudBackend(
        model_id=model_id, base_url=base_url, api_key=api_key, **kwargs
    )
