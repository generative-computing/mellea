#!/bin/bash


echo "launching a vllm server. Logs are found in $(readlink -ef $(dirname $0))/vllm.log"
vllm serve ibm-granite/granite-4.0-h-micro \
      --enable-lora \
      --dtype bfloat16 \
      --max-lora-rank 64 \
      --enable-prefix-caching \
      > $(readlink -ef $(dirname $0))/vllm.log \
      2> $(readlink -ef $(dirname $0))/vllm.err
