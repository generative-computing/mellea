# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A first-class OrcaRouter backend built on the OpenAI-compatible client.

OrcaRouter (https://www.orcarouter.ai) exposes a single OpenAI-compatible
endpoint (`https://api.orcarouter.ai/v1`) in front of many hosted models, with
adaptive routing, automatic failover, and gateway-level security. This backend
is a thin, provider-branded subclass of `OpenAIBackend` that pins the base URL
and API key environment variable, so callers get the OrcaRouter stack as a named
provider instead of an anonymous custom base URL.
"""

from __future__ import annotations

import os

from ..formatters import ChatFormatter, TemplateFormatter
from . import model_ids
from .openai import OpenAIBackend

#: Default OrcaRouter endpoint. Mirrors how ``OpenAIBackend`` defaults to the
#: OpenAI API; OrcaRouter is OpenAI-compatible, so the chat completions schema
#: is identical.
ORCAROUTER_BASE_URL = "https://api.orcarouter.ai/v1"


class OrcaRouterBackend(OpenAIBackend):
    """An OpenAI-compatible backend for OrcaRouter's hosted model gateway.

    OrcaRouter serves many models behind a single OpenAI-compatible endpoint and
    adds adaptive routing, automatic failover, observability, and gateway-level
    security on top. This backend points at that endpoint by default, so the
    OrcaRouter stack can be used as a named provider instead of an anonymous
    custom base URL.

    Args:
        model_id (str | ModelIdentifier): OpenAI-compatible model identifier.
            Defaults to `model_ids.ORCAROUTER_AUTO`.
        formatter (ChatFormatter | None): Formatter for rendering components.
            Defaults to `TemplateFormatter`.
        base_url (str | None): Base URL for the API endpoint; defaults to the
            OrcaRouter endpoint (`https://api.orcarouter.ai/v1`) if not set.
        model_options (dict | None): Default model options for generation requests.
        api_key (str | None): API key; falls back to `ORCAROUTER_API_KEY` env var.
        kwargs: Additional keyword arguments forwarded to the OpenAI client.

    Raises:
        ValueError: If neither `api_key` nor `ORCAROUTER_API_KEY` is set.
        TypeError: If `model_id` is neither a `str` nor a `ModelIdentifier`.
    """

    def __init__(
        self,
        model_id: str | model_ids.ModelIdentifier = model_ids.ORCAROUTER_AUTO,
        formatter: ChatFormatter | None = None,
        base_url: str | None = None,
        model_options: dict | None = None,
        *,
        default_to_constraint_checking_alora: bool = True,
        load_embedded_adapters: bool = False,
        adapter_source: str | None = None,
        api_key: str | None = None,
        default_extra_body: dict | None = None,
        **kwargs,
    ):
        """Initialize an OrcaRouter backend with the given model and credentials."""
        if api_key is None:
            api_key = os.getenv("ORCAROUTER_API_KEY")
        if api_key is None:
            raise ValueError(
                "ORCAROUTER_API_KEY or api_key is required but not set. Please either:\n"
                "  1. Set the environment variable: export ORCAROUTER_API_KEY='your-key-here'\n"
                "  2. Pass it as a parameter: OrcaRouterBackend(api_key='your-key-here')"
            )

        super().__init__(
            model_id=model_id,
            formatter=(
                formatter
                if formatter is not None
                else TemplateFormatter(model_id=model_id)
            ),
            base_url=base_url
            or os.getenv("ORCAROUTER_BASE_URL")
            or ORCAROUTER_BASE_URL,
            model_options=model_options,
            default_to_constraint_checking_alora=default_to_constraint_checking_alora,
            load_embedded_adapters=load_embedded_adapters,
            adapter_source=adapter_source,
            api_key=api_key,
            default_extra_body=default_extra_body,
            **kwargs,
        )
        self._provider: str = "orcarouter"
