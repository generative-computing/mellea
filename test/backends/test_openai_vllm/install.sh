#!/bin/bash -xe

conda env remove -y -n mellea || true
conda env create -f $(readlink -ef $(dirname $0))/environment.yml

in-conda (){
    conda run -n mellea $@
}


in-conda pip install -e '.[vllm]' --group dev
in-conda uv pip install pre-commit
