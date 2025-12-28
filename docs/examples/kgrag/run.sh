#!/bin/bash

# Change to the script's directory to ensure correct module paths
cd "$(dirname "$0")"

# Set PYTHONPATH to include the current directory so Python can find kg, utils, etc.
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Set KG_BASE_DIRECTORY to the dataset directory relative to current location
export KG_BASE_DIRECTORY="$(pwd)/dataset"

# uv run --with mellea run/run_kg_preprocess.py
# uv run --with mellea run/run_kg_embed.py
uv run --with mellea run/run_kg_update.py --num-workers 1 --queue-size 1
# uv run --with mellea run/run_qa.py --num-workers 1 --queue-size 1