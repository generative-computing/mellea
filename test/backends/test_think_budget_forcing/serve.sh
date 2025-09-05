#!/bin/bash

export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

echo "launching a vllm server. Logs are found in $(readlink -ef $(dirname $0))/vllm.log"
      # At the time of writing this code, Granite 4.4 vLLM serving did not support prefix-caching
      # --enable-prefix-caching \
vllm serve $LOCAL_TEST_MODEL \
      --dtype bfloat16 \
      > $(readlink -ef $(dirname $0))/vllm.log \
      2> $(readlink -ef $(dirname $0))/vllm.err


