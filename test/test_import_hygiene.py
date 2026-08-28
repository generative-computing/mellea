# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guards that `import mellea` stays cheap.

Several heavy third-party packages are reachable from mellea but are only needed
by narrow features (majority-voting sampling, pricing metrics, individual
backends). They are imported lazily at first use. These tests fail if any of them
is pulled back to module scope, which would silently re-add seconds to every
`import mellea`.
"""

import importlib.util
import json
import subprocess
import sys

import pytest

# Packages that must not be loaded by a bare `import mellea`, and the feature
# that legitimately needs each one.
FORBIDDEN_ON_IMPORT = {
    "nltk": "granite citation parsing / rouge_score",
    "scipy": "transitive via nltk",
    "sklearn": "transitive via nltk",
    "pandas": "transitive via litellm",
    "litellm": "pricing metrics + LiteLLM backend",
    "openai": "transitive via litellm; OpenAI backend",
    "rouge_score": "MBRDRougeLStrategy",
    "math_verify": "MajorityVotingStrategyForMath",
    "torch": "HuggingFace backend",
    "transformers": "HuggingFace backend",
    # Provider SDKs. Concrete backends are imported explicitly by user code from
    # `mellea.backends.<provider>`, so none of their clients should load here —
    # `import mellea` must not commit the caller to a provider.
    "ollama": "Ollama backend",
    "ibm_watsonx_ai": "Watsonx backend",
    "boto3": "Bedrock backend",
    "docling": "RichDocument",
    "matplotlib": "plotting requirements",
}


def _modules_loaded_by(statement: str) -> set[str]:
    """Return the set of top-level modules in sys.modules after running `statement`.

    Runs in a subprocess so the parent test session's already-imported modules
    do not pollute the result.

    Args:
        statement: Python source executed before sys.modules is sampled.

    Returns:
        Top-level module names (text before the first dot) present in
        `sys.modules` after the statement runs.
    """
    code = (
        f"{statement}\n"
        "import sys, json\n"
        "print(json.dumps(sorted({m.split('.')[0] for m in sys.modules})))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    return set(json.loads(proc.stdout.strip().splitlines()[-1]))


@pytest.fixture(scope="module")
def modules_after_import_mellea() -> set[str]:
    """Top-level modules loaded by a bare `import mellea`."""
    return _modules_loaded_by("import mellea")


@pytest.mark.parametrize(
    ("package", "reason"), sorted((k, v) for k, v in FORBIDDEN_ON_IMPORT.items())
)
def test_heavy_package_not_imported_by_mellea(
    package: str, reason: str, modules_after_import_mellea: set[str]
) -> None:
    """Heavy optional-feature packages stay unimported after `import mellea`."""
    if importlib.util.find_spec(package) is None:
        pytest.skip(f"{package} is not installed in this environment")
    assert package not in modules_after_import_mellea, (
        f"`import mellea` pulled in {package!r} (needed only for: {reason}). "
        "Move the import inside the function or method that uses it, or guard "
        "it with TYPE_CHECKING if it is only needed for annotations."
    )


def test_majority_voting_strategies_still_constructible() -> None:
    """The lazily-imported sampling strategies work when actually used."""
    from mellea.stdlib.sampling import MajorityVotingStrategyForMath, MBRDRougeLStrategy

    assert MBRDRougeLStrategy().compare_strings("a cat sat", "a cat sat") == 1.0
    assert MajorityVotingStrategyForMath().compare_strings("$1+1$", "$2$") == 1.0
