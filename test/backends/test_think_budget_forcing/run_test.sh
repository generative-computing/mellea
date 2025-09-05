#!/bin/bash

export PYTHONBREAKPOINT="ipdb.set_trace"
export LOCAL_TEST_MODEL="ibm-granite/granite-4.0-tiny-preview"
# export LOCAL_TEST_MODEL="unsloth/Llama-3.2-1B"

ENV_NAME=mellea_tbf
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

dir=$(readlink -ef $(dirname $0))
rm $dir/vllm.log $dir/vllm.err

bash $dir/serve.sh &
vllm_pid=$!

trap "kill -SIGINT $vllm_pid ; wait" EXIT

while sleep 1 ; do
    if grep -q "Application startup complete." $dir/vllm.err
    then
        break
    fi
done

python test_think_budget_forcing.py


