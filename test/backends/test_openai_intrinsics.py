"""E2E tests for intrinsics on the OpenAI backend with a Granite Switch model.

Starts a vLLM server hosting a Granite Switch model, creates an OpenAIBackend with
``embedded_adapters=True``, and runs each intrinsic that has matching test data
through the full generation path.
"""

import json
import os
import pathlib
import signal
import subprocess
import time

import pytest
import requests

from test.predicates import require_gpu

# ---------------------------------------------------------------------------
# Module-level markers
# ---------------------------------------------------------------------------
pytestmark = [
    pytest.mark.openai,
    pytest.mark.e2e,
    pytest.mark.vllm,
    require_gpu(min_vram_gb=12),
    pytest.mark.skipif(
        int(os.environ.get("CICD", 0)) == 1,
        reason="Skipping OpenAI intrinsics tests in CI",
    ),
]

# ---------------------------------------------------------------------------
# Imports (after markers so collection-time skips fire first)
# ---------------------------------------------------------------------------
from mellea.backends.model_ids import IBM_GRANITE_SWITCH_4_1_3B
from mellea.backends.openai import OpenAIBackend
from mellea.formatters import TemplateFormatter
from mellea.stdlib import functional as mfuncs
from mellea.stdlib.components import Intrinsic, Message
from mellea.stdlib.components.docs.document import Document
from mellea.stdlib.components.intrinsic import rag
from mellea.stdlib.context import ChatContext
from test.formatters.granite.test_intrinsics_formatters import (
    _YAML_JSON_COMBOS_WITH_MODEL,
    YamlJsonCombo,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SWITCH_MODEL_ID = os.environ.get(
    "GRANITE_SWITCH_MODEL_ID", IBM_GRANITE_SWITCH_4_1_3B.hf_model_name
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def backend():
    """OpenAI backend with embedded adapters auto-loaded from the switch model."""
    base_url = (
        os.environ.get("VLLM_SWITCH_TEST_BASE_URL", "http://127.0.0.1:8000") + "/v1"
    )
    return OpenAIBackend(
        model_id=SWITCH_MODEL_ID,
        formatter=TemplateFormatter(model_id=SWITCH_MODEL_ID),
        base_url=base_url,
        api_key="EMPTY",
        load_embedded_adapters=True,
    )


def _registered_intrinsic_names(backend: OpenAIBackend) -> set[str]:
    """Return the set of intrinsic names that have registered adapters."""
    names = set()
    for adapter in backend._added_adapters.values():
        names.add(adapter.intrinsic_name)
    return names


def _get_matching_combos(backend: OpenAIBackend) -> dict[str, YamlJsonCombo]:
    """Filter test combos to those whose task matches a registered adapter."""
    registered = _registered_intrinsic_names(backend)
    return {
        k: v for k, v in _YAML_JSON_COMBOS_WITH_MODEL.items() if v.task in registered
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_chat_context(input_json: dict) -> ChatContext:
    """Build a ChatContext from the raw input JSON used by intrinsics tests.

    Parses messages and attaches any documents from ``extra_body.documents``
    to the last user message.
    """
    ctx = ChatContext()
    messages_data = input_json.get("messages", [])
    docs_data = input_json.get("extra_body", {}).get("documents", [])

    documents = [
        Document(text=d["text"], title=d.get("title"), doc_id=d.get("doc_id"))
        for d in docs_data
    ]

    for i, msg in enumerate(messages_data):
        role = msg["role"]
        content = msg["content"]
        # Attach documents to the last message
        is_last = i == len(messages_data) - 1
        if is_last and documents:
            ctx = ctx.add(Message(role, content, documents=documents))
        else:
            ctx = ctx.add(Message(role, content))

    return ctx


# ---------------------------------------------------------------------------
# Parametrized test
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module", params=list(_YAML_JSON_COMBOS_WITH_MODEL.keys()))
def intrinsic_combo(request, backend) -> YamlJsonCombo:
    """Yield each test combo that matches a registered adapter on the backend."""
    combo_name = request.param
    combo = _YAML_JSON_COMBOS_WITH_MODEL[combo_name]

    registered = _registered_intrinsic_names(backend)
    if combo.task not in registered:
        pytest.skip(
            f"Intrinsic '{combo.task}' has no registered adapter on this backend "
            f"(available: {registered})"
        )

    return combo._resolve_yaml()


@pytest.mark.qualitative
def test_intrinsic_generation(intrinsic_combo: YamlJsonCombo, backend: OpenAIBackend):
    """Run an intrinsic through the OpenAI backend and validate the result."""
    cfg = intrinsic_combo

    # Load input
    with open(cfg.inputs_file, encoding="utf-8") as f:
        input_json = json.load(f)

    # Load optional intrinsic kwargs
    intrinsic_kwargs = {}
    if cfg.arguments_file:
        with open(cfg.arguments_file, encoding="utf-8") as f:
            intrinsic_kwargs = json.load(f)

    # Build context and intrinsic action
    ctx = _build_chat_context(input_json)
    assert cfg.task is not None
    intrinsic = Intrinsic(cfg.task, intrinsic_kwargs=intrinsic_kwargs)

    # Run the full generation path
    result, _new_ctx = mfuncs.act(intrinsic, ctx, backend, strategy=None)

    # Validate that we got a non-empty result
    assert result.value is not None, f"Intrinsic '{cfg.task}' returned None"
    assert len(result.value) > 0, f"Intrinsic '{cfg.task}' returned empty string"

    # Validate that the result is parseable JSON
    try:
        json.loads(result.value)
    except json.JSONDecodeError:
        pytest.fail(
            f"Intrinsic '{cfg.task}' did not return valid JSON: {result.value[:200]}"
        )


# ---------------------------------------------------------------------------
# call_intrinsic tests — exercise the high-level convenience wrappers
# ---------------------------------------------------------------------------
_RAG_TEST_DATA = (
    pathlib.Path(__file__).parent.parent
    / "stdlib"
    / "components"
    / "intrinsic"
    / "testdata"
    / "input_json"
)


@pytest.fixture(scope="module")
def call_intrinsic_backend(vllm_switch_process):
    """OpenAI backend with embedded_adapters=False so call_intrinsic loads them dynamically."""
    base_url = (
        os.environ.get("VLLM_SWITCH_TEST_BASE_URL", "http://127.0.0.1:8000") + "/v1"
    )
    return OpenAIBackend(
        model_id=SWITCH_MODEL_ID,
        formatter=TemplateFormatter(model_id=SWITCH_MODEL_ID),
        base_url=base_url,
        api_key="EMPTY",
        load_embedded_adapters=False,
    )


def _read_rag_input(file_name: str) -> tuple[ChatContext, str, list[Document]]:
    """Load RAG test data and convert to Mellea types."""
    with open(_RAG_TEST_DATA / file_name, encoding="utf-8") as f:
        data = json.load(f)

    context = ChatContext()
    for m in data["messages"][:-1]:
        context = context.add(Message(m["role"], m["content"]))

    last_turn = data["messages"][-1]["content"]

    documents = [
        Document(text=d["text"], doc_id=d.get("doc_id"))
        for d in data.get("extra_body", {}).get("documents", [])
    ]
    return context, last_turn, documents


@pytest.mark.qualitative
def test_call_intrinsic_answerability(call_intrinsic_backend):
    """call_intrinsic path: check_answerability returns a score between 0 and 1."""
    context, question, documents = _read_rag_input("answerability.json")
    result = rag.check_answerability(
        question, documents, context, call_intrinsic_backend
    )
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


@pytest.mark.qualitative
def test_call_intrinsic_context_relevance(call_intrinsic_backend):
    """call_intrinsic path: check_context_relevance returns a score between 0 and 1."""
    context, question, documents = _read_rag_input("context_relevance.json")
    result = rag.check_context_relevance(
        question, documents[0], context, call_intrinsic_backend
    )
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])