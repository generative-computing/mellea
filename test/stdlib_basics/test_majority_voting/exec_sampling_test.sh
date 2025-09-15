#!/bin/bash

source set_variables.sh

eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

export LOCAL_TEST_MODEL

python test_majority_voting.py
