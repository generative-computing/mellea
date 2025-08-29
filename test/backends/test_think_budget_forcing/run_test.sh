#!/bin/bash

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


