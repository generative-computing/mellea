#!/bin/bash -xe

source set_variables.sh

conda env remove -y -n $ENV_NAME || true
conda env create -f $(readlink -f $(dirname $0))/environment.yml

in-conda (){
    conda run -n $ENV_NAME $@
}


cd ../../../
in-conda uv pip install -e .
cd -
in-conda uv pip install pre-commit
in-conda uv pip install pytest
in-conda uv pip install vllm==0.10.0
in-conda uv pip install outlines
# in-conda uv pip install unsloth
in-conda uv pip install ipdb
in-conda uv pip install math-verify[antlr4_13_2]
in-conda uv pip install rouge-score

