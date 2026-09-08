#!/bin/bash
# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Build a local Ollama model from the official pinned Granite uncertainty aLoRA.
#
# Prints only the resulting Ollama tag to stdout so callers can assign it:
#   export MELLEA_OLLAMA_UNCERTAINTY_MODEL="$(./test/scripts/build_ollama_uncertainty_adapter.sh)"

set -euo pipefail

log() { echo "[ollama-adapter] $*" >&2; }
die() { log "ERROR: $*"; exit 1; }

OLLAMA_BIN="${OLLAMA_BIN:-}"
if [[ -z "$OLLAMA_BIN" ]]; then
    OLLAMA_BIN="$(command -v ollama || true)"
fi
[[ -n "$OLLAMA_BIN" ]] || die "ollama is not installed or not on PATH."
BASE_MODEL="${OLLAMA_BASE_MODEL:-granite4.1:3b}"
OUTPUT_MODEL="${MELLEA_OLLAMA_UNCERTAINTY_MODEL:-mellea-test/uncertainty-alora:latest}"
ADAPTER_REPO="ibm-granite/granitelib-core-r1.0"
ADAPTER_REVISION="d0a2a96a4cd07e96f0fe7ca29a42bfe088299d43"
ADAPTER_PATH="uncertainty/granite-4.1-3b/alora"
BASE_REPO="ibm-granite/granite-4.1-3b"
BASE_REVISION="c0650403e44e78ec0262dab1c90914c65b196c4e"
LLAMA_CPP_REVISION="e71b80510c848c00175924ecf3c40333ccae8eb5"

if [[ -n "${MELLEA_OLLAMA_ADAPTER_CACHE_DIR:-}" ]]; then
    BUILD_CACHE="$MELLEA_OLLAMA_ADAPTER_CACHE_DIR"
elif [[ -n "${CACHE_DIR:-}" ]]; then
    BUILD_CACHE="${CACHE_DIR}/ollama-adapter-build"
else
    BUILD_CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/mellea/ollama-adapter-build"
fi

BASE_DIR="${BUILD_CACHE}/base"
ADAPTER_DIR="${BUILD_CACHE}/adapter"
HF_HOME_DIR="${BUILD_CACHE}/huggingface"
LLAMA_CPP_DIR="${BUILD_CACHE}/llama.cpp"
CONVERTER_VENV="${BUILD_CACHE}/converter-venv"
ADAPTER_GGUF="${BUILD_CACHE}/uncertainty-alora-f16.gguf"
MODELFILE="${BUILD_CACHE}/Modelfile"

mkdir -p "$BUILD_CACHE"

if [[ ! -f "$ADAPTER_GGUF" ]]; then
    log "Downloading official Granite base and uncertainty aLoRA..."
    HF_HOME="$HF_HOME_DIR" uv run --quiet --frozen --all-extras --all-groups \
        python - "$BASE_DIR" "$ADAPTER_DIR" "$BASE_REPO" "$BASE_REVISION" "$ADAPTER_REPO" "$ADAPTER_REVISION" "$ADAPTER_PATH" <<'PY'
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

base_dir = Path(sys.argv[1])
adapter_dir = Path(sys.argv[2])
base_repo = sys.argv[3]
base_revision = sys.argv[4]
adapter_repo = sys.argv[5]
adapter_revision = sys.argv[6]
adapter_path = sys.argv[7]
snapshot_download(
    repo_id=base_repo,
    revision=base_revision,
    local_dir=base_dir,
    ignore_patterns=["*.gguf", "*.onnx", "*.tflite"],
)
adapter_dir.mkdir(parents=True, exist_ok=True)
for filename in (
    "adapter_config.json",
    "adapter_model.safetensors",
    "io.yaml",
    "model.sig",
):
    source = hf_hub_download(
        repo_id=adapter_repo,
        filename=f"{adapter_path}/{filename}",
        revision=adapter_revision,
    )
    shutil.copy2(source, adapter_dir / filename)
PY

    if [[ ! -d "$LLAMA_CPP_DIR/.git" ]]; then
        log "Cloning llama.cpp at ${LLAMA_CPP_REVISION}..."
        git clone --quiet https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP_DIR"
    fi
    git -C "$LLAMA_CPP_DIR" fetch --quiet --depth 1 origin "$LLAMA_CPP_REVISION"
    git -C "$LLAMA_CPP_DIR" checkout --quiet --detach "$LLAMA_CPP_REVISION"

    if [[ ! -x "$CONVERTER_VENV/bin/python" ]]; then
        log "Installing pinned converter dependencies..."
        uv venv "$CONVERTER_VENV"
        uv pip install --python "$CONVERTER_VENV/bin/python" \
            -r "$LLAMA_CPP_DIR/requirements/requirements-convert_lora_to_gguf.txt"
    fi

    log "Converting official uncertainty aLoRA to GGUF..."
    "$CONVERTER_VENV/bin/python" "$LLAMA_CPP_DIR/convert_lora_to_gguf.py" \
        "$ADAPTER_DIR" \
        --base "$BASE_DIR" \
        --outtype f16 \
        --outfile "$ADAPTER_GGUF"

    METADATA_DUMP="${BUILD_CACHE}/uncertainty-alora-metadata.txt"
    "$CONVERTER_VENV/bin/python" \
        "$LLAMA_CPP_DIR/gguf-py/gguf/scripts/gguf_dump.py" "$ADAPTER_GGUF" \
        > "$METADATA_DUMP"
    if ! grep -q "adapter.alora.invocation_tokens" "$METADATA_DUMP"; then
        die "Converted adapter is missing aLoRA invocation-token metadata."
    fi
fi

printf 'FROM %s\nADAPTER %s\n' "$BASE_MODEL" "$ADAPTER_GGUF" > "$MODELFILE"
log "Creating Ollama model ${OUTPUT_MODEL}..."
"$OLLAMA_BIN" create "$OUTPUT_MODEL" -f "$MODELFILE" >/dev/null

printf '%s\n' "$OUTPUT_MODEL"
