#!/bin/bash

# @Masa note:
# the following code is a bash snippet Kristian gave me
# for how to run vllm with lora adapter loaded.

# HF_GRANITE_ALORA_SNAPSHOT=${HF_HOME:-$HOME/.cache/huggingface}
# HF_GRANITE_ALORA_SNAPSHOT+=/hub/
# HF_GRANITE_ALORA_SNAPSHOT+=models--ibm-granite--granite-3.2-8b-alora-requirement-check/
# HF_GRANITE_ALORA_SNAPSHOT+=snapshots/d55a7a7f5796609bc938c5c151a864cfcc6ab54e

# vllm serve ibm-granite/granite-3.2-8b-instruct \
#       --enable-lora \
#       --lora-modules "{\"name\": \"ibm-granite/granite-3.2-8b-alora-requirement-check\", \"path\": \"${HF_GRANITE_ALORA_SNAPSHOT}\", \"base_model_name\": \"ibm-granite/granite-3.2-8b-instruct\"}" \
#       --dtype bfloat16 \
#       --max-lora-rank 64 \
#       --enable-prefix-caching

# However, in our test, we do not load the alora when we serve.
# In this test, we use the dynamic loading interface from
# https://docs.vllm.ai/en/stable/features/lora.html#dynamically-serving-lora-adapters

# Using this feature requires the following environment variable.
# If you use conda/miniforge,
# this variable must have been set already when you set up the environment.
# see environment.yml.
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

echo "launching a vllm server. Logs are found in $(readlink -ef $(dirname $0))/vllm.log"
vllm serve ibm-granite/granite-3.2-8b-instruct \
      --enable-activated-lora \
      --enable-lora \
      --dtype bfloat16 \
      --max-lora-rank 64 \
      --enable-prefix-caching \
      > $(readlink -ef $(dirname $0))/vllm.log \
      2> $(readlink -ef $(dirname $0))/vllm.err
