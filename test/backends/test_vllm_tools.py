import os

import pytest

from mellea.backends.tools import MelleaTool

# Skip entire module in CI since the single test is qualitative
pytestmark = [
    pytest.mark.vllm,
    pytest.mark.llm,
    pytest.mark.requires_gpu,
    pytest.mark.requires_heavy_ram,
    pytest.mark.skipif(
        int(os.environ.get("CICD", 0)) == 1,
        reason="Skipping vLLM tools tests in CI - qualitative test",
    ),
]

# Try to import vLLM backend - skip all tests if not available
try:
    import mellea.backends.model_ids as model_ids
    from mellea import MelleaSession
    from mellea.backends import ModelOption
    from mellea.backends.vllm import LocalVLLMBackend
    from mellea.stdlib.context import ChatContext
except ImportError as e:
    pytest.skip(
        f"vLLM backend not available: {e}. Install with: pip install mellea[vllm]",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def backend():
    """Shared vllm backend for all tests in this module."""
    if os.environ.get("VLLM_USE_V1", -1) != "0":
        pytest.skip("skipping vllm tests; tests require `export VLLM_USE_V1=0`")

    backend = LocalVLLMBackend(
        model_id=model_ids.MISTRALAI_MISTRAL_0_3_7B,
        model_options={
            # made smaller for a testing environment with smaller gpus.
            # such an environment could possibly be running other gpu applications, including slack
            "gpu_memory_utilization": 0.8,
            "max_model_len": 8192,
            "max_num_seqs": 8,
        },
    )
    return backend


@pytest.fixture(scope="function")
def session(backend):
    """Fresh HuggingFace session for each test."""
    session = MelleaSession(backend, ctx=ChatContext())
    yield session
    session.reset()


@pytest.mark.qualitative
def test_tool(session):
    tool_call_history = []

    def get_temperature(location: str) -> int:
        """Returns today's temperature of the given city in Celsius.

        Args:
            location: a city name.
        """
        tool_call_history.append(location)
        return 21

    output = session.instruct(
        "What is today's temperature in Boston? Answer in Celsius. Reply the number only.",
        model_options={
            ModelOption.TOOLS: [MelleaTool.from_callable(get_temperature)],
            ModelOption.MAX_NEW_TOKENS: 1000,
        },
        tool_calls=True,
    )

    assert output.tool_calls is not None

    result = output.tool_calls["get_temperature"].call_func()
    print(result)

    assert len(tool_call_history) > 0
    assert tool_call_history[0].lower() == "boston"
    assert 21 == result


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
