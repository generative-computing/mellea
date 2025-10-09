#!/bin/bash

PYTHONBREAKPOINT="ipdb.set_trace"
LOCAL_TEST_MODEL="ibm-granite/granite-4.0-tiny-preview"
ENV_NAME=mellea_tbf
DIR=$(readlink -ef $(dirname $0))
VLLM_LOG=$DIR/vllm.log
VLLM_ERR=$DIR/vllm.err
