#!/bin/bash

source set_variables.sh

eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

rm $VLLM_LOG $VLLM_ERR

bash ./serve.sh &
VLLM_PID=$!

trap "kill -SIGINT $VLLM_PID ; wait" EXIT

while sleep 1 ; do
    if grep -q "Application startup complete." $VLLM_ERR
    then
        break
    fi
done

bash exec_sampling_test.sh


