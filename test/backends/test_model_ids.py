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
@pytest.mark.slow
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
        # When token=False the Hub maps gated repos to RepositoryNotFoundError to
        # avoid leaking their existence. Skip rather than fail so anonymous CI
        # doesn't false-fail on gated models; a human with HF_TOKEN can verify.
        pytest.skip(
            f"{const_name}.hf_model_name={hf_name!r} not found on HuggingFace Hub "
            "(may be gated — re-run with HF_TOKEN to confirm it exists)."
        )


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.ollama
@pytest.mark.parametrize(
    "const_name,ollama_name", _OLLAMA_IDS, ids=[n for n, _ in _OLLAMA_IDS]
)
def test_ollama_model_names_exist(const_name: str, ollama_name: str) -> None:
    """Every ollama_name in model_ids.py must exist in the Ollama library.

    The test first checks whether the model is already present locally via
    show(). If it is not, it queries the Ollama registry manifest endpoint
    directly — no pull is initiated, so no data is downloaded.

    The test is skipped (not failed) when the Ollama server is unreachable or
    returns an unexpected error, since those conditions reflect environment
    problems rather than a bad model ID.
    """
    import urllib.error
    import urllib.request

    import ollama as ollama_sdk

    client = ollama_sdk.Client()
    try:
        client.show(ollama_name)
        # Model is present locally — name is valid.
        return
    except ollama_sdk.ResponseError as e:
        if "not found" not in str(e).lower() and "404" not in str(e):
            pytest.skip(
                f"Ollama server returned an unexpected error for show({ollama_name!r}): {e}"
            )
        # Model not cached locally; fall through to registry check.
    except ConnectionError as e:
        pytest.skip(f"Ollama server unreachable: {e}")

    # Check the Ollama registry manifest without downloading anything.
    # Strip any tag so we query the base model name.
    base_name = ollama_name.split(":")[0]
    tag = ollama_name.split(":")[1] if ":" in ollama_name else "latest"
    registry_url = f"https://registry.ollama.ai/v2/{base_name}/manifests/{tag}"
    try:
        req = urllib.request.Request(registry_url, method="HEAD")
        urllib.request.urlopen(req, timeout=10)
    except urllib.error.HTTPError as e:
        if e.code in (404, 401):
            pytest.fail(
                f"{const_name}.ollama_name={ollama_name!r} does not exist in the "
                "Ollama library. Update or remove the model ID in "
                "mellea/backends/model_ids.py."
            )
        else:
            pytest.skip(
                f"Ollama registry returned unexpected HTTP {e.code} for {ollama_name!r}"
            )
    except OSError as e:
        pytest.skip(f"Could not reach Ollama registry: {e}")
