#!/bin/bash -e

CONDA=""
if which mamba > /dev/null
then
    CONDA=$(which mamba)
fi
if which conda > /dev/null
then
    CONDA=$(which conda)
fi

if [ -z $CONDA ]
then
    echo "Error: conda or mamba is not installed or is not in the PATH."
    echo "Go to "
    echo "* https://github.com/conda-forge/miniforge (open source)"
    echo "* https://www.anaconda.com/download/success (registration required)"
    echo "to obtain a conda/mamba installer."
else
    echo "Using $CONDA for environment setup"
fi


usage(){
    echo "Usage: install.sh [-h]"
    echo
    echo "-h : show this help"
    exit 1
}

while getopts "h" OPTNAME ; do
    case "${OPTNAME}" in
        h)
            usage
            ;;
        :)
            # If expected argument omitted:
            echo "Error: -${OPTARG} requires an argument."
            exit 1
            ;;
        *)
            # If unknown (any other) option:
            echo "Error: -${OPTARG} unknown."
            exit 1
            ;;
    esac
done

# note:
# this is a portable way (works in linux and osx) to get the directory of this script.
# readlink -ef may not work on osx.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
$CONDA env create -f $SCRIPT_DIR/environment.yml

$CONDA run -n mellea uv pip install -e .[all] --group dev --group notebook --group docs

$CONDA run -n mellea pre-commit install
