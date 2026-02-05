"""Dataclasses for ModelIdentifiers."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class ModelIdentifier:
    """The `ModelIdentifier` class wraps around model identification strings.

    Using model strings is messy:
        1. Different platforms use variations on model id strings.
        2. Using raw strings is annoying because: no autocomplete, typos, hallucinated names, mismatched model and tokenizer names, etc.
    """

    hf_model_name: str | None = None
    ollama_name: str | None = None
    watsonx_name: str | None = None
    mlx_name: str | None = None
    openai_name: str | None = None
    bedrock_name: str | None = None

    hf_tokenizer_name: str | None = None  # if None, is the same as hf_model_name


####################
#### IBM models ####
####################

IBM_GRANITE_4_MICRO_3B = ModelIdentifier(
    hf_model_name="ibm-granite/granite-4.0-micro",
    ollama_name="granite4:micro",
    watsonx_name="ibm/granite-4-h-small",
)
# todo: watsonx model is different from ollama model - should be same.


IBM_GRANITE_3_2_8B = ModelIdentifier(
    hf_model_name="ibm-granite/granite-3.2-8b-instruct",
    ollama_name="granite3.2:8b",
    watsonx_name="ibm/granite-3-2b-instruct",
)

IBM_GRANITE_3_3_8B = ModelIdentifier(
    hf_model_name="ibm-granite/granite-3.3-8b-instruct",
    ollama_name="granite3.3:8b",
    watsonx_name="ibm/granite-3-3-8b-instruct",
)

IBM_GRANITE_GUARDIAN_3_0_2B = ModelIdentifier(
    hf_model_name="ibm-granite/granite-guardian-3.0-2b",
    ollama_name="granite3-guardian:2b",
)

IBM_GRANITE_4_TINY_PREVIEW_7B = ModelIdentifier(
    hf_model_name="ibm-granite/granite-4.0-tiny-preview"
)

IBM_GRANITE_4_TINY_PREVIEW_BASE_7B = ModelIdentifier(
    hf_model_name="ibm-granite/granite-4.0-tiny-base-preview"
)

#####################
#### Meta models ####
#####################

#### LLAMA 4 models ####
META_LLAMA_4_SCOUT_17B_16E_INSTRUCT = ModelIdentifier(
    hf_model_name="unsloth/Llama-4-Scout-17B-16E-Instruct",
    ollama_name="llama4:scout",
    hf_tokenizer_name="unsloth/Llama-4-Scout-17B-16E-Instruct",
    mlx_name="mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit",
    bedrock_name="meta.llama4-scout-17b-instruct-v1:0",
)

META_LLAMA_4_MAVERICK_17B_128E_INSTRUCT = ModelIdentifier(
    hf_model_name="unsloth/Llama-4-Maverick-17B-128E-Instruct",
    ollama_name="llama4:maverick",
    watsonx_name=None,  # NOTE: we do have a fp8 model in watsonx (meta-llama/llama-4-maverick-17b-128e-instruct-fp8) Not sure if we want to include it here.
    hf_tokenizer_name="unsloth/Llama-4-Maverick-17B-128E-Instruct",
    mlx_name="mlx-community/Llama-4-Maverick-17B-128E-Instruct-4bit",
    bedrock_name="meta.llama4-maverick-17b-instruct-v1:0",
)

#### LLAMA 3 models ####
META_LLAMA_3_3_70B = ModelIdentifier(
    hf_model_name="unsloth/Llama-3.3-70B-Instruct",
    ollama_name="llama3.3:70b",
    watsonx_name="meta-llama/llama-3-3-70b-instruct",
    hf_tokenizer_name="unsloth/Llama-3.3-70B-Instruct",
    mlx_name="mlx-community/Llama-3.3-70B-Instruct-4bit",
    bedrock_name="meta.llama3-3-70b-instruct-v1:0",
)

META_LLAMA_3_2_3B = ModelIdentifier(
    hf_model_name="unsloth/Llama-3.2-3B-Instruct",
    ollama_name="llama3.2:3b",
    watsonx_name="meta-llama/llama-3-2-3b-instruct",
    bedrock_name="meta.llama3-2-3b-instruct-v1:0",
)

META_LLAMA_GUARD3_1B = ModelIdentifier(
    ollama_name="llama-guard3:1b", hf_model_name="unsloth/Llama-Guard-3-1B"
)

META_LLAMA_3_2_1B = ModelIdentifier(
    ollama_name="llama3.2:1b",
    hf_model_name="unsloth/Llama-3.2-1B",
    bedrock_name="meta.llama3-2-1b-instruct-v1:0",
)

########################
#### Mistral models ####
########################

MISTRALAI_MISTRAL_0_3_7B = ModelIdentifier(
    hf_model_name="mistralai/Mistral-7B-Instruct-v0.3",  # Mistral 7B v0.3
    ollama_name="mistral:7b",  # Ollama
)

MISTRALAI_MISTRAL_SMALL_24B = ModelIdentifier(
    hf_model_name="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    ollama_name="mistral-small:latest",
    watsonx_name="mistralai/mistral-small-3-1-24b-instruct-2503",
)

MISTRALAI_MISTRAL_LARGE_123B = ModelIdentifier(
    hf_model_name="mistralai/Mistral-Large-Instruct-2411",
    ollama_name="mistral-large:latest",
    watsonx_name="mistralai/mistral-large",
)


#####################
#### Qwen models ####
#####################

QWEN3_0_6B = ModelIdentifier(
    hf_model_name="Qwen/Qwen3-0.6B",  # Qwen 0.6B
    ollama_name="qwen3:0.6b",  # Ollama
)

QWEN3_1_7B = ModelIdentifier(
    hf_model_name="Qwen/Qwen3-1.7B",  # Qwen 1.7B
    ollama_name="qwen3:1.7b",  # Ollama
)

QWEN3_8B = ModelIdentifier(
    hf_model_name="Qwen/Qwen3-8B",  # Qwen 8B
    ollama_name="qwen3:8b",  # Ollama
)

QWEN3_14B = ModelIdentifier(
    hf_model_name="Qwen/Qwen3-14B",  # Qwen 14B
    ollama_name="qwen3:14b",  # Ollama
)

###########################
#### OpenAI open models ###
###########################

OPENAI_GPT_OSS_20B = ModelIdentifier(
    hf_model_name="openai/gpt-oss-20b",  # OpenAI GPT-OSS 20B
    ollama_name="gpt-oss:20b",  # Ollama
    bedrock_name="openai.gpt-oss-20b-1:0",
)
OPENAI_GPT_OSS_120B = ModelIdentifier(
    hf_model_name="openai/gpt-oss-120b",  # OpenAI GPT-OSS 120B
    ollama_name="gpt-oss:120b",  # Ollama
    bedrock_name="openai.gpt-oss-120b-1:0",
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
)

MS_PHI_4_14B = ModelIdentifier(
    hf_model_name="microsoft/phi-4",  # Microsoft Phi-4 14B
    ollama_name="phi4:14b",  # Ollama
)

MS_PHI_4_MINI_REASONING_4B = ModelIdentifier(
    hf_model_name="microsoft/Phi-4-mini-flash-reasoning",
    ollama_name="phi4-mini-reasoning:3.8b",
)


DEEPSEEK_R1_8B = ModelIdentifier(
    hf_model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    ollama_name="deepseek-r1:8b",
)


HF_SMOLLM2_2B = ModelIdentifier(
    ollama_name="smollm2:1.7b",
    hf_model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    mlx_name="mlx-community/SmolLM2-1.7B-Instruct",
)

HF_SMOLLM3_3B_no_ollama = ModelIdentifier(
    hf_model_name="HuggingFaceTB/SmolLM3-3B", ollama_name=""
)
