#!/bin/bash

dir=$(readlink -ef $(dirname $0))
rm $dir/vllm.log $dir/vllm.err

bash $dir/serve.sh &
vllm_pid=$!

trap "kill $vllm_pid ; wait" EXIT

while sleep 1 ; do
    if grep -q "Application startup complete." $dir/vllm.err
    then
        break
    fi
done

VLLM_TESTS_ENABLED="1" python $dir/test_openai_vllm.py


# The VLLM process doesn't always get cleaned up. Get the pid of the VLLM::Engine zombie process and kill it.
potential_zombie_process=$( grep -m 1 -oP 'EngineCore_DP0 pid=\K\d+' $(readlink -ef $(dirname $0))/vllm.err)
kill -9 $potential_zombie_process