#!/bin/bash -xe

conda=""
if $(which mamba)
then
    conda=$(which mamba)
fi
if $(which conda)
then
    conda=$(which conda)
fi
if [ -z $conda ]
then
    echo "Error: conda or mamba is not installed or is not in the PATH."
fi

echo "using $conda for environment setup"

$conda env remove -y -n mellea || true
$conda env create -f environment.yml

$conda run -n mellea uv pip install -e .[all] --group dev
