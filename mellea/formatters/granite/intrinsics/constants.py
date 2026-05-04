# SPDX-License-Identifier: Apache-2.0

"""Constants relating to of input and output processing for RAG-related intrinsics."""

YAML_REQUIRED_FIELDS = ["model", "response_format", "transformations"]
"""Fields that must be present in every intrinsic's YAML configuration file."""

YAML_OPTIONAL_FIELDS = [
    "logprobs_workaround",
    "docs_as_message",
    "instruction",
    "name",
    "parameters",
    "sentence_boundaries",
]
"""Fields that may be present in every intrinsic's YAML configuration file. If
not present, the parsed config dictionary will contain a null value in their place."""

YAML_JSON_FIELDS = ["response_format"]
"""Fields of the YAML file that contain JSON values as strings"""

OLD_LAYOUT_REPOS = [
    "ibm-granite/rag-intrinsics-lib",
    "generative-computing/rag-intrinsics-lib",
    "generative-computing/core-intrinsics-lib",
]
"""Repositories (aka "models") on Hugging Face Hub that use the old layout of
``<task>/<adapter type>/<base model>``.
"""

BASE_MODEL_TO_CANONICAL_NAME = {
    "ibm-granite/granite-3.3-8b-instruct": "granite-3.3-8b-instruct",
    "ibm-granite/granite-3.3-2b-instruct": "granite-3.3-2b-instruct",
    "openai/gpt-oss-20b": "gpt-oss-20b",
    "ibm-granite/granite-4.0-micro": "granite-4.0-micro",
    "ibm-granite/granite-4.1-3b": "granite-4.1-3b",
    "ibm-granite/granite-4.1-8b": "granite-4.1-8b",
    "ibm-granite/granite-4.1-30b": "granite-4.1-30b",
    "granite4:micro": "granite4_micro",
}
"""Base model names that we accept for LoRA/aLoRA adapters in intrinsics libraries.
Each model name maps to the name of the directory that contains (a)LoRA adapters for
that model."""

TOP_LOGPROBS = 10
"""Number of logprobs we request per token when decoding logprobs."""
