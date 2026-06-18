"""Integration tests verifying that model IDs in model_ids.py resolve to real models.

HuggingFace tests check the Hub API (no model download required).
Ollama tests require a running Ollama server and are skipped otherwise.
"""

import inspect

import pytest

import mellea.backends.model_ids as model_ids
from mellea.backends.model_ids import ModelIdentifier

# Collect all ModelIdentifier constants defined at module level.
_ALL_IDS: list[tuple[str, ModelIdentifier]] = [
    (name, obj)
    for name, obj in inspect.getmembers(model_ids)
    if isinstance(obj, ModelIdentifier)
]

_HF_IDS = [(name, obj.hf_model_name) for name, obj in _ALL_IDS if obj.hf_model_name]
_OLLAMA_IDS = [
    (name, obj.ollama_name)
    for name, obj in _ALL_IDS
    if obj.ollama_name  # excludes None and ""
]


@pytest.mark.integration
@pytest.mark.parametrize("const_name,hf_name", _HF_IDS, ids=[n for n, _ in _HF_IDS])
def test_hf_model_names_exist(const_name: str, hf_name: str) -> None:
    """Every hf_model_name in model_ids.py must resolve to a real HuggingFace repo."""
    pytest.importorskip("huggingface_hub", reason="huggingface_hub not installed")
    from huggingface_hub import model_info
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    try:
        model_info(hf_name, token=False)
    except GatedRepoError:
        # Gated repos exist but require auth — that's fine.
        pass
    except RepositoryNotFoundError:
        pytest.fail(
            f"{const_name}.hf_model_name={hf_name!r} does not exist on HuggingFace Hub. "
            "Update or remove the model ID in mellea/backends/model_ids.py."
        )


@pytest.mark.integration
@pytest.mark.ollama
@pytest.mark.parametrize(
    "const_name,ollama_name", _OLLAMA_IDS, ids=[n for n, _ in _OLLAMA_IDS]
)
def test_ollama_model_names_exist(const_name: str, ollama_name: str) -> None:
    """Every ollama_name in model_ids.py must be pullable from the Ollama library."""
    import ollama as ollama_sdk

    client = ollama_sdk.Client()
    try:
        # show() works for locally-present models; for absent ones we check
        # the registry via a dry-run pull request that checks availability.
        client.show(ollama_name)
    except ollama_sdk.ResponseError as e:
        if "not found" in str(e).lower() or "404" in str(e):
            # Model not cached locally; verify it exists in the Ollama library.
            try:
                # Pull with stream=True but break immediately after the first
                # status update — we only need to confirm the model is known.
                for _update in client.pull(ollama_name, stream=True):
                    break
            except ollama_sdk.ResponseError as pull_err:
                if "not found" in str(pull_err).lower() or "404" in str(pull_err):
                    pytest.fail(
                        f"{const_name}.ollama_name={ollama_name!r} does not exist in the "
                        "Ollama library. Update or remove the model ID in "
                        "mellea/backends/model_ids.py."
                    )
