"""E2E tests for intrinsics on the OpenAI backend with a Granite Switch model.

Starts a vLLM server hosting a Granite Switch model, creates an OpenAIBackend with
``embedded_adapters=True``, and runs each intrinsic that has matching test data
through the full generation path.
"""

import json
import os
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
from mellea.backends.openai import OpenAIBackend
from mellea.formatters import TemplateFormatter
from mellea.formatters.granite.intrinsics import json_util
from mellea.stdlib import functional as mfuncs
from mellea.stdlib.components import Intrinsic, Message
from mellea.stdlib.components.docs.document import Document
from mellea.stdlib.context import ChatContext
from test.formatters.granite.test_intrinsics_formatters import (
    _TEST_DATA_DIR,
    _YAML_JSON_COMBOS_WITH_MODEL,
    YamlJsonCombo,
    _round_floats,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SWITCH_MODEL_ID = os.environ.get(
    "GRANITE_SWITCH_MODEL_ID", "GrizleeBer/gs-test-2"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def vllm_switch_process():
    """Module-scoped vLLM process serving a Granite Switch model.

    If ``VLLM_SWITCH_TEST_BASE_URL`` is set the server is assumed to be running
    externally and no subprocess is started.
    """
    if os.environ.get("VLLM_SWITCH_TEST_BASE_URL"):
        # Verify the external server is serving the expected model.
        base = os.environ["VLLM_SWITCH_TEST_BASE_URL"]
        try:
            resp = requests.get(f"{base}/v1/models", timeout=5)
            resp.raise_for_status()
            served = {m["id"] for m in resp.json().get("data", [])}
            if SWITCH_MODEL_ID not in served:
                pytest.skip(
                    f"External vLLM server at {base} is not serving "
                    f"'{SWITCH_MODEL_ID}' (serving: {served})",
                    allow_module_level=True,
                )
        except requests.RequestException as exc:
            pytest.skip(
                f"Cannot reach external vLLM server at {base}: {exc}",
                allow_module_level=True,
            )
        yield None
        return

    # Require CUDA — vLLM does not support MPS
    try:
        subprocess.run(["nvidia-smi", "-L"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip(
            "No CUDA GPU detected — skipping vLLM OpenAI intrinsics tests",
            allow_module_level=True,
        )

    vllm_venv = os.environ.get("VLLM_VENV_PATH", ".vllm-venv")
    vllm_python = os.path.join(vllm_venv, "bin", "python")
    if not os.path.isfile(vllm_python):
        subprocess.run(["uv", "venv", vllm_venv, "--python", "3.11"], check=True)
        subprocess.run(
            ["uv", "pip", "install", "--python", vllm_python, "vllm"], check=True
        )

    process = None
    try:
        process = subprocess.Popen(
            [
                vllm_python,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                SWITCH_MODEL_ID,
                "--served-model-name",
                SWITCH_MODEL_ID,
                "--dtype",
                "bfloat16",
                "--enable-prefix-caching",
                "--gpu-memory-utilization",
                "0.4",
                "--max-num-seqs",
                "256",
                "--max-model-len",
                "4096",
            ],
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        url = "http://127.0.0.1:8000/ping"
        timeout = 600
        start_time = time.time()

        while True:
            if process.poll() is not None:
                output = process.stdout.read() if process.stdout else ""
                raise RuntimeError(
                    f"vLLM server exited before startup (code {process.returncode}).\n"
                    f"--- vLLM output ---\n{output}\n--- end ---"
                )
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Timed out waiting for vLLM health check at {url}")

        yield process

    except Exception as e:
        output = ""
        if process is not None and process.stdout:
            try:
                output = process.stdout.read()
            except Exception:
                pass
        skip_msg = (
            f"vLLM process not available: {e}\n"
            f"--- vLLM output ---\n{output}\n--- end ---"
        )
        print(skip_msg)
        pytest.skip(skip_msg, allow_module_level=True)

    finally:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)
            except Exception:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except Exception:
                    pass
                process.wait()


@pytest.fixture(scope="module")
def backend(vllm_switch_process):
    """OpenAI backend with embedded adapters auto-loaded from the switch model."""
    base_url = (
        os.environ.get("VLLM_SWITCH_TEST_BASE_URL", "http://127.0.0.1:8000") + "/v1"
    )
    return OpenAIBackend(
        model_id=SWITCH_MODEL_ID,
        formatter=TemplateFormatter(model_id=SWITCH_MODEL_ID),
        base_url=base_url,
        api_key="EMPTY",
        embedded_adapters=True,
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
    intrinsic = Intrinsic(cfg.task, intrinsic_kwargs=intrinsic_kwargs)

    # Run the full generation path
    result, _new_ctx = mfuncs.act(intrinsic, ctx, backend, strategy=None)

    # Validate that we got a result
    assert result.value is not None, f"Intrinsic '{cfg.task}' returned None"
    assert len(result.value) > 0, f"Intrinsic '{cfg.task}' returned empty string"

    # Parse the result JSON
    try:
        result_json = json.loads(result.value)
    except json.JSONDecodeError:
        pytest.fail(
            f"Intrinsic '{cfg.task}' did not return valid JSON: {result.value[:200]}"
        )

    # Compare against expected output
    expected_file = _TEST_DATA_DIR / f"test_run_transformers/{cfg.short_name}.json"
    if not expected_file.exists():
        # No expected file for this combo — just validate it's valid JSON
        return

    from mellea.formatters.granite import ChatCompletionResponse

    with open(expected_file, encoding="utf-8") as f:
        expected = ChatCompletionResponse.model_validate_json(f.read())

    # Round floats for approximate comparison
    result_rounded = _round_floats(
        json_util.parse_inline_json(result_json)
        if isinstance(result_json, dict)
        else result_json,
        num_digits=2,
    )
    expected_content = expected.choices[0].message.content
    expected_json = json.loads(expected_content) if expected_content else {}
    expected_rounded = _round_floats(
        json_util.parse_inline_json(expected_json)
        if isinstance(expected_json, dict)
        else expected_json,
        num_digits=2,
    )

    if result_rounded != expected_rounded:
        # Fall back to approximate comparison
        assert result_rounded == pytest.approx(expected_rounded, abs=0.1)
