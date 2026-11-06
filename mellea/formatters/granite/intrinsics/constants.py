# Copyright IBM Corp. All Rights Reserved.
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
    "provenance",
    "sentence_boundaries",
    "state_template",
    "system_prompt",
    "invocation_text",
]
"""Fields that may be present in every intrinsic's YAML configuration file. If
not present, the parsed config dictionary will contain a null value in their place.

`system_prompt`, `state_template`, and `invocation_text` carry the trained
prompt layout of one-token decision capabilities (see the `first_token`
transformation rule in `output.py`). `provenance` is free-form metadata about
where a configuration came from; it is never sent to the model."""

YAML_JSON_FIELDS = ["response_format"]
"""Fields of the YAML file that contain JSON values as strings"""

OLD_LAYOUT_REPOS = [
    "ibm-granite/rag-intrinsics-lib",
    "generative-computing/rag-intrinsics-lib",
    "generative-computing/core-intrinsics-lib",
]
"""Repositories (aka "models") on Hugging Face Hub that use the old layout of
`<task>/<adapter type>/<base model>`.
"""

BASE_MODEL_TO_CANONICAL_NAME = {
    "ibm-granite/granite-3.3-8b-instruct": "granite-3.3-8b-instruct",
    "ibm-granite/granite-3.3-2b-instruct": "granite-3.3-2b-instruct",
    "openai/gpt-oss-20b": "gpt-oss-20b",
    "ibm-granite/granite-4.0-micro": "granite-4.0-micro",
    "ibm-granite/granite-4.1-3b": "granite-4.1-3b",
    "ibm-granite/granite-4.1-8b": "granite-4.1-8b",
    "ibm-granite/granite-4.1-30b": "granite-4.1-30b",
    "ibm-granite/granite-4.2-3b": "granite-4.2-3b",
    "ibm-granite/granite-4.2-8b": "granite-4.2-8b",
    "ibm-granite/granite-4.2-30b": "granite-4.2-30b",
    "granite4:micro": "granite4_micro",
    "granite4.1:3b": "granite4.1_3b",
    "granite4.1:8b": "granite4.1_8b",
    "granite4.1:30b": "granite4.1_30b",
    "granite4.2:3b": "granite4.2_3b",
    "granite4.2:8b": "granite4.2_8b",
    "granite4.2:30b": "granite4.2_30b",
}
"""Base model names that we accept for LoRA/aLoRA adapters in intrinsics libraries.
Each model name maps to the name of the directory that contains (a)LoRA adapters for
that model."""

TOP_LOGPROBS = 10
"""Number of logprobs we request per token when decoding logprobs."""

TOP_LOGPROBS_MAX = 20
"""Upper bound on `top_logprobs` for one-token decision readouts.

The `first_token` rule sizes `top_logprobs` to its configured label pool,
capped at this value; a deployment must allow at least this many
(vLLM `--max-logprobs`, default 20)."""
