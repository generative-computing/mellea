# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`ModelIdentifier` dataclass and a catalog of pre-defined model IDs.

`ModelIdentifier` is a frozen dataclass that groups the platform-specific name
variants for a model (Hugging Face, Ollama, WatsonX, MLX, OpenAI, Bedrock) so that
a single constant can be passed to any backend without manual string translation.
The module also ships a curated catalog of ready-to-use constants for popular
open-weight models including IBM Granite 4, Meta Llama 4, Mistral, Qwen, and
NVIDIA Nemotron families.
"""

import dataclasses


@dataclasses.dataclass(frozen=True)
class ModelIdentifier:
    """The `ModelIdentifier` class wraps around model identification strings.

    Using model strings is messy:
        1. Different platforms use variations on model id strings.
        2. Using raw strings is annoying because: no autocomplete, typos, hallucinated names, mismatched model and tokenizer names, etc.

    Args:
        hf_model_name (str | None): Hugging Face Hub model repository ID (e.g. `"ibm-granite/granite-3.3-8b-instruct"`).
        ollama_name (str | None): Ollama model tag (e.g. `"granite3.3:8b"`).
        watsonx_name (str | None): WatsonX AI model ID (e.g. `"ibm/granite-3-2b-instruct"`).
        mlx_name (str | None): MLX model identifier for Apple Silicon inference.
        openai_name (str | None): Model name for an OpenAI-compatible chat completions API
            (e.g. `"gpt-5.1"`). Not necessarily a model hosted by OpenAI — see the note below.
        bedrock_name (str | None): AWS Bedrock model ID (e.g. `"openai.gpt-oss-20b"`).
        hf_tokenizer_name (str | None): Hugging Face tokenizer ID; defaults to `hf_model_name` if `None`.

    Note:
        `openai_name` means "name for an OpenAI-compatible endpoint", not "hosted by OpenAI".
        Open-weight models in this catalog use it for the provider that hosts them — the
        `NVIDIA_*` constants, for example, carry their NVIDIA-hosted NIM names, served from
        `https://integrate.api.nvidia.com/v1`. Such a name will 404 against the default
        `https://api.openai.com/v1`, so set `base_url` to match the provider. Self-hosted
        servers such as vLLM serve a model under whatever name it was launched with, which
        by default is `hf_model_name`, not `openai_name`.

    """

    hf_model_name: str | None = None
    ollama_name: str | None = None
    watsonx_name: str | None = None
    mlx_name: str | None = None
    openai_name: str | None = None
    bedrock_name: str | None = None
    bedrock_litellm_name: str | None = None

    hf_tokenizer_name: str | None = None  # if None, is the same as hf_model_name
    context_length: int | None = None


####################
#### IBM models ####
####################

# Granite 4 Hybrid Models (Recommended for general use)
IBM_GRANITE_4_HYBRID_MICRO = ModelIdentifier(
    hf_model_name="ibm-granite/granite-4.0-h-micro",
    ollama_name="granite4:micro-h",
    watsonx_name=None,  # Only h-small available on Watsonx
    context_length=131072,
)

IBM_GRANITE_4_HYBRID_TINY = ModelIdentifier(
    hf_model_name="ibm-granite/granite-4.0-h-tiny",
    ollama_name="granite4:tiny-h",
    watsonx_name=None,  # Only h-small available on Watsonx
    context_length=131072,
)

IBM_GRANITE_4_HYBRID_SMALL = ModelIdentifier(
    hf_model_name="ibm-granite/granite-4.0-h-small",
    ollama_name="granite4:small-h",
    watsonx_name="ibm/granite-4-h-small",
    context_length=131072,
)

IBM_GRANITE_4_HYBRID_1B = ModelIdentifier(
    hf_model_name="ibm-granite/granite-4.0-h-1b",
    ollama_name="granite4:1b-h",
    watsonx_name=None,
    context_length=131072,
)

IBM_GRANITE_4_HYBRID_350m = ModelIdentifier(
    hf_model_name="ibm-granite/granite-4.0-h-350m",
    ollama_name="granite4:350m-h",
    watsonx_name=None,
    context_length=32768,
)

# Granite 4.1 Dense Models
IBM_GRANITE_4_1_3B = ModelIdentifier(
    hf_model_name="ibm-granite/granite-4.1-3b",
    ollama_name="granite4.1:3b",
    watsonx_name=None,
    context_length=131072,
)

IBM_GRANITE_4_1_8B = ModelIdentifier(
    hf_model_name="ibm-granite/granite-4.1-8b",
    ollama_name="granite4.1:8b",
    context_length=131072,
)

IBM_GRANITE_4_1_30B = ModelIdentifier(
    hf_model_name="ibm-granite/granite-4.1-30b",
    ollama_name="granite4.1:30b",
    context_length=131072,
)

IBM_GRANITE_GUARDIAN_4_1_8B = ModelIdentifier(
    hf_model_name="ibm-granite/granite-guardian-4.1-8b", context_length=131072
)


# Deprecated Granite 3 models - kept for backward compatibility
# These maintain their original model references (not upgraded to Granite 4)
IBM_GRANITE_3_2_8B = ModelIdentifier(
    hf_model_name="ibm-granite/granite-3.2-8b-instruct",
    ollama_name="granite3.2:8b",
    watsonx_name="ibm/granite-3-2b-instruct",
    context_length=131072,
)

IBM_GRANITE_3_3_8B = ModelIdentifier(
    hf_model_name="ibm-granite/granite-3.3-8b-instruct",
    ollama_name="granite3.3:8b",
    watsonx_name="ibm/granite-3-3-8b-instruct",
    context_length=131072,
)

IBM_GRANITE_4_MICRO_3B = ModelIdentifier(
    hf_model_name="ibm-granite/granite-4.0-micro",
    ollama_name="granite4:micro",
    watsonx_name="ibm/granite-4-h-small",  # Keeping hybrid version here for backwards compatibility.
    context_length=131072,
)

# Granite 3.3 Vision Model (2B)
IBM_GRANITE_3_3_VISION_2B = ModelIdentifier(
    hf_model_name="ibm-granite/granite-vision-3.3-2b",
    ollama_name="ibm/granite3.3-vision:2b",
    watsonx_name=None,
    context_length=131072,
)

IBM_GRANITE_GUARDIAN_3_0_2B = ModelIdentifier(
    hf_model_name="ibm-granite/granite-guardian-3.0-2b",
    ollama_name="granite3-guardian:2b",
    context_length=4096,
)

IBM_GRANITE_4_TINY_PREVIEW_7B = ModelIdentifier(
    hf_model_name="ibm-granite/granite-4.0-tiny-preview", context_length=131072
)

IBM_GRANITE_4_TINY_PREVIEW_BASE_7B = ModelIdentifier(
    hf_model_name="ibm-granite/granite-4.0-tiny-base-preview", context_length=131072
)

# Pre-Built Granite Switch Models
IBM_GRANITE_SWITCH_4_1_3B_PREVIEW = ModelIdentifier(
    hf_model_name="ibm-granite/granite-switch-4.1-3b-preview", context_length=131072
)
"""Granite Switch Preview Model. Adapters: `citations`, `query_rewrite`, `query_clarification`, `hallucination_detection`, `answerability`, `policy-guardrails`, `guardian-core`, `uncertainty`, `requirement-check`, `context-attribution`, `factuality-detection`, `factuality-correction`."""  # Document what adapters are included by default here.

IBM_GRANITE_SWITCH_4_1_8B_PREVIEW = ModelIdentifier(
    hf_model_name="ibm-granite/granite-switch-4.1-8b-preview", context_length=131072
)
"""Granite Switch Preview Model. Adapters: `citations`, `query_rewrite`, `query_clarification`, `hallucination_detection`, `answerability`, `policy-guardrails`, `guardian-core`, `uncertainty`, `requirement-check`, `context-attribution`, `factuality-detection`, `factuality-correction`."""  # Document what adapters are included by default here.

IBM_GRANITE_SWITCH_4_1_30B_PREVIEW = ModelIdentifier(
    hf_model_name="ibm-granite/granite-switch-4.1-30b-preview", context_length=131072
)
"""Granite Switch Preview Model. Adapters: `citations`, `query_rewrite`, `query_clarification`, `hallucination_detection`, `answerability`, `policy-guardrails`, `guardian-core`, `uncertainty`, `requirement-check`, `context-attribution`, `factuality-detection`, `factuality-correction`."""  # Document what adapters are included by default here.

#####################
#### Meta models ####
#####################

#### LLAMA 4 models ####
META_LLAMA_4_SCOUT_17B_16E_INSTRUCT = ModelIdentifier(
    hf_model_name="unsloth/Llama-4-Scout-17B-16E-Instruct",
    ollama_name="llama4:scout",
    hf_tokenizer_name="unsloth/Llama-4-Scout-17B-16E-Instruct",
    mlx_name="mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit",
    context_length=10485760,
)

META_LLAMA_4_MAVERICK_17B_128E_INSTRUCT = ModelIdentifier(
    hf_model_name="unsloth/Llama-4-Maverick-17B-128E-Instruct",
    ollama_name="llama4:maverick",
    watsonx_name=None,  # NOTE: we do have a fp8 model in watsonx (meta-llama/llama-4-maverick-17b-128e-instruct-fp8) Not sure if we want to include it here.
    hf_tokenizer_name="unsloth/Llama-4-Maverick-17B-128E-Instruct",
    mlx_name="mlx-community/Llama-4-Maverick-17B-128E-Instruct-4bit",
    context_length=1048576,
)

#### LLAMA 3 models ####
META_LLAMA_3_3_70B = ModelIdentifier(
    hf_model_name="unsloth/Llama-3.3-70B-Instruct",
    ollama_name="llama3.3:70b",
    watsonx_name="meta-llama/llama-3-3-70b-instruct",
    hf_tokenizer_name="unsloth/Llama-3.3-70B-Instruct",
    mlx_name="mlx-community/Llama-3.3-70B-Instruct-4bit",
    context_length=131072,
)

META_LLAMA_3_2_3B = ModelIdentifier(
    hf_model_name="unsloth/Llama-3.2-3B-Instruct",
    ollama_name="llama3.2:3b",
    watsonx_name="meta-llama/llama-3-2-3b-instruct",
    context_length=131072,
)

META_LLAMA_GUARD3_1B = ModelIdentifier(
    ollama_name="llama-guard3:1b",
    hf_model_name="meta-llama/Llama-Guard-3-1B",
    context_length=131072,
)

META_LLAMA_3_2_1B = ModelIdentifier(
    ollama_name="llama3.2:1b",
    hf_model_name="unsloth/Llama-3.2-1B",
    context_length=131072,
)

########################
#### Mistral models ####
########################

MISTRALAI_MISTRAL_0_3_7B = ModelIdentifier(
    hf_model_name="mistralai/Mistral-7B-Instruct-v0.3",  # Mistral 7B v0.3
    ollama_name="mistral:7b",  # Ollama
    context_length=32768,
)

MISTRALAI_MISTRAL_SMALL_24B = ModelIdentifier(
    hf_model_name="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    ollama_name="mistral-small:latest",
    watsonx_name="mistralai/mistral-small-3-1-24b-instruct-2503",
    context_length=131072,
)

MISTRALAI_MISTRAL_LARGE_123B = ModelIdentifier(
    hf_model_name="mistralai/Mistral-Large-Instruct-2411",
    ollama_name="mistral-large:latest",
    watsonx_name="mistralai/mistral-large",
    context_length=131072,
)

MISTRALAI_DEVSTRAL_2_123B = ModelIdentifier(
    bedrock_name="mistral.devstral-2-123b",
    bedrock_litellm_name="bedrock/converse/mistral.devstral-2-123b",
)

#####################
#### Qwen models ####
#####################

QWEN3_0_6B = ModelIdentifier(
    hf_model_name="Qwen/Qwen3-0.6B",  # Qwen 0.6B
    ollama_name="qwen3:0.6b",  # Ollama
    context_length=32768,
)

QWEN3_1_7B = ModelIdentifier(
    hf_model_name="Qwen/Qwen3-1.7B",  # Qwen 1.7B
    ollama_name="qwen3:1.7b",  # Ollama
    context_length=32768,
)

QWEN3_8B = ModelIdentifier(
    hf_model_name="Qwen/Qwen3-8B",  # Qwen 8B
    ollama_name="qwen3:8b",  # Ollama
    context_length=40960,  # 8B+ series; smaller Qwen3 models use 32768
)

QWEN3_14B = ModelIdentifier(
    hf_model_name="Qwen/Qwen3-14B",  # Qwen 14B
    ollama_name="qwen3:14b",  # Ollama
    context_length=40960,
)

#######################
#### NVIDIA models ####
#######################

# `openai_name` below is the NVIDIA-hosted NIM name, served from
# `https://integrate.api.nvidia.com/v1` — an OpenAI-compatible endpoint, but *not* OpenAI.
# Pass a matching `base_url` (and an NVIDIA API key) when using these with `OpenAIBackend`.
# These names do not apply to self-hosted vLLM/SGLang, which serve under whatever
# `--served-model-name` they were launched with (by default, the `hf_model_name` repo id).

#### Nemotron 3 models (hybrid Mamba-Transformer, current generation) ####
# HF publishes one repo per precision; the BF16 repos are the reference weights.
NVIDIA_NEMOTRON_3_NANO_4B = ModelIdentifier(
    hf_model_name="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
    ollama_name="nemotron-3-nano:4b",
    mlx_name="mlx-community/NVIDIA-Nemotron-3-Nano-4B-4bit",
    # No NIM or Bedrock endpoint: this size ships for local/edge inference only.
    context_length=262144,
)

NVIDIA_NEMOTRON_3_NANO_30B_A3B = ModelIdentifier(
    hf_model_name="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    ollama_name="nemotron-3-nano:30b",
    mlx_name="mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit",
    openai_name="nvidia/nemotron-3-nano-30b-a3b",
    bedrock_name="nvidia.nemotron-nano-3-30b",
    bedrock_litellm_name="bedrock/converse/nvidia.nemotron-nano-3-30b",
    context_length=262144,
)

NVIDIA_NEMOTRON_3_SUPER_120B_A12B = ModelIdentifier(
    hf_model_name="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
    ollama_name="nemotron-3-super:120b",
    mlx_name="mlx-community/NVIDIA-Nemotron-3-Super-120B-A12B-4bit",
    openai_name="nvidia/nemotron-3-super-120b-a12b",
    bedrock_name="nvidia.nemotron-super-3-120b",
    bedrock_litellm_name="bedrock/converse/nvidia.nemotron-super-3-120b",
    context_length=262144,
)

NVIDIA_NEMOTRON_3_ULTRA_550B_A55B = ModelIdentifier(
    hf_model_name="nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
    ollama_name="nemotron-3-ultra:cloud",  # Ollama ships this size cloud-only; no local weights.
    openai_name="nvidia/nemotron-3-ultra-550b-a55b",
    context_length=262144,
)

# Nemotron 3 Nano Omni: video, audio, image, and text understanding.
NVIDIA_NEMOTRON_3_NANO_OMNI_30B_A3B = ModelIdentifier(
    hf_model_name="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16",
    ollama_name="nemotron3:33b",
    openai_name="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    context_length=131072,  # Multimodal max_sequence_length; the text tower alone allows 262144.
)

#### Nemotron Nano v2 models ####
NVIDIA_NEMOTRON_NANO_9B_V2 = ModelIdentifier(
    hf_model_name="nvidia/NVIDIA-Nemotron-Nano-9B-v2",
    mlx_name="mlx-community/NVIDIA-Nemotron-Nano-9B-v2-4bits",
    openai_name="nvidia/nvidia-nemotron-nano-9b-v2",  # Doubled prefix is NVIDIA's own naming.
    bedrock_name="nvidia.nemotron-nano-9b-v2",
    bedrock_litellm_name="bedrock/converse/nvidia.nemotron-nano-9b-v2",
    context_length=131072,
)

# No hosted endpoint for the text-only 12B v2. NVIDIA and AWS both host only the
# vision-language variant (NIM `nvidia/nemotron-nano-12b-v2-vl`, Bedrock
# `nvidia.nemotron-nano-12b-v2`), which is a different model than the repo below.
NVIDIA_NEMOTRON_NANO_12B_V2 = ModelIdentifier(
    hf_model_name="nvidia/NVIDIA-Nemotron-Nano-12B-v2", context_length=131072
)

#### Llama-Nemotron models ####
NVIDIA_LLAMA_3_1_NEMOTRON_NANO_8B = ModelIdentifier(
    hf_model_name="nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
    openai_name="nvidia/llama-3.1-nemotron-nano-8b-v1",
    context_length=131072,
)

NVIDIA_LLAMA_3_3_NEMOTRON_SUPER_49B = ModelIdentifier(
    hf_model_name="nvidia/Llama-3_3-Nemotron-Super-49B-v1_5",
    openai_name="nvidia/llama-3.3-nemotron-super-49b-v1.5",
    context_length=131072,
)

NVIDIA_LLAMA_3_1_NEMOTRON_ULTRA_253B = ModelIdentifier(
    hf_model_name="nvidia/Llama-3_1-Nemotron-Ultra-253B-v1",
    openai_name="nvidia/llama-3.1-nemotron-ultra-253b-v1",
    context_length=131072,
)

NVIDIA_LLAMA_3_1_NEMOTRON_70B = ModelIdentifier(
    hf_model_name="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
    ollama_name="nemotron:70b",
    mlx_name="mlx-community/Llama-3.1-Nemotron-70B-Instruct-HF-4bit",
    openai_name="nvidia/llama-3.1-nemotron-70b-instruct",
    context_length=131072,
)

#### Nemotron Mini ####
NVIDIA_NEMOTRON_MINI_4B = ModelIdentifier(
    hf_model_name="nvidia/Nemotron-Mini-4B-Instruct",
    ollama_name="nemotron-mini:4b",
    openai_name="nvidia/nemotron-mini-4b-instruct",
    context_length=4096,
)

###########################
#### OpenAI open models ###
###########################

OPENAI_GPT_OSS_20B = ModelIdentifier(
    hf_model_name="openai/gpt-oss-20b",  # OpenAI GPT-OSS 20B
    ollama_name="gpt-oss:20b",  # Ollama
    bedrock_name="openai.gpt-oss-20b",
    bedrock_litellm_name="bedrock/converse/openai.gpt-oss-20b-1:0",
    context_length=131072,
)
OPENAI_GPT_OSS_120B = ModelIdentifier(
    hf_model_name="openai/gpt-oss-120b",  # OpenAI GPT-OSS 120B
    ollama_name="gpt-oss:120b",  # Ollama
    bedrock_name="openai.gpt-oss-120b",
    bedrock_litellm_name="bedrock/converse/openai.gpt-oss-120b-1:0",
    context_length=131072,
)

###########################
#### OpenAI prop models ###
###########################

OPENAI_GPT_5_1 = ModelIdentifier(
    openai_name="gpt-5.1"  # OpenAI GPT-5.1
)

#####################
#### Misc models ####
#####################

GOOGLE_GEMMA_3N_E4B = ModelIdentifier(
    hf_model_name="google/gemma-3n-e4b-it",  # Google Gemma 3N E4B
    ollama_name="gemma3n:e4b",  # Ollama
    context_length=32768,
)

MS_PHI_4_14B = ModelIdentifier(
    hf_model_name="microsoft/phi-4",  # Microsoft Phi-4 14B
    ollama_name="phi4:14b",  # Ollama
    context_length=16384,
)

MS_PHI_4_MINI_REASONING_4B = ModelIdentifier(
    hf_model_name="microsoft/Phi-4-mini-flash-reasoning",
    ollama_name="phi4-mini-reasoning:3.8b",
    context_length=131072,
)


DEEPSEEK_R1_8B = ModelIdentifier(
    hf_model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    ollama_name="deepseek-r1:8b",
    context_length=131072,
)


HF_SMOLLM2_2B = ModelIdentifier(
    ollama_name="smollm2:1.7b",
    hf_model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    mlx_name="mlx-community/SmolLM2-1.7B-Instruct",
    context_length=8192,
)

HF_SMOLLM3_3B_no_ollama = ModelIdentifier(
    hf_model_name="HuggingFaceTB/SmolLM3-3B", ollama_name=None, context_length=65536
)
