#!/usr/bin/env bash
# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# start_llamacpp_switch.sh
# Serves a Granite Switch model with llama-server so Mellea's OpenAIBackend can
# run adapter functions against it (issue #1735). Downloads the checkpoint from
# the Hugging Face Hub, converts it to GGUF (f16), starts llama-server with
# --jinja, waits until it can serve, and prints the Mellea snippet to connect.
# Runs in the foreground; Ctrl-C shuts the server down.
#
# Why this exists alongside scripts/start_llamacpp.sh: the plain-Granite script
# serves a prebuilt GGUF, but IBM publishes no Switch GGUF. llama.cpp's
# converter handles GraniteSwitchForCausalLM directly (ggml-org/llama.cpp#25107;
# see LLAMA_CPP_REVISION below for the revision current checkpoints need), so
# this script converts from the HF checkpoint. The server is tested on v0.6.0.
#
# Usage:
#   ./scripts/start_llamacpp_switch.sh                      # 3b, port 8081
#   LLAMACPP_SWITCH_MODEL=ibm-granite/granite-switch-4.1-8b-preview \
#     ./scripts/start_llamacpp_switch.sh                    # 8b
#   LLAMACPP_SWITCH_PORT=9090 ./scripts/start_llamacpp_switch.sh
#   LLAMACPP_SWITCH_QUANT=q8_0 ./scripts/start_llamacpp_switch.sh   # untested
#
# Requires `llama-server` on PATH (v0.1.0 or newer, for the granite-switch
# architecture; tested on v0.6.0), `git`, `curl`, and `uv` for the converter
# dependencies:
#   macOS:  brew install llama.cpp
#   Linux:  https://github.com/ggml-org/llama.cpp/releases (or build from source)
#
# The first run downloads the checkpoint (~7 GB for the 3b) and the pinned
# llama.cpp sources, and builds a converter venv under
# ${XDG_CACHE_HOME:-~/.cache}/mellea/llamacpp-switch-build. Later runs reuse
# everything and start in seconds.

set -euo pipefail

log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { log "ERROR: $*" >&2; exit 1; }

# --- Configuration ---------------------------------------------------------
# Any Granite Switch checkpoint works, as long as llama-server (see below) and
# the converter both support it. The alias must equal what Mellea sends: the
# OpenAI backend resolves a ModelIdentifier without an openai_name to its
# hf_model_name.
LLAMACPP_SWITCH_MODEL="${LLAMACPP_SWITCH_MODEL:-ibm-granite/granite-switch-4.1-3b-preview}"
LLAMACPP_ALIAS="$LLAMACPP_SWITCH_MODEL"
# The port deliberately differs from scripts/start_llamacpp.sh's 8080, so a
# Switch server and the plain-Granite server can run side by side, and so the
# `llamacpp` pytest marker (which gates on a server at LLAMACPP_BASE_URL,
# default 8080) keeps meaning "the base-Granite server". Examples that need
# this server read LLAMACPP_SWITCH_BASE_URL instead, defaulting to 8081.
LLAMACPP_SWITCH_PORT="${LLAMACPP_SWITCH_PORT:-8081}"
LLAMACPP_SWITCH_HOST="${LLAMACPP_SWITCH_HOST:-127.0.0.1}"
LLAMACPP_SWITCH_QUANT="${LLAMACPP_SWITCH_QUANT:-f16}"
LLAMACPP_SWITCH_CTX_SIZE="${LLAMACPP_SWITCH_CTX_SIZE:-32768}"
LLAMACPP_SWITCH_READY_TIMEOUT="${LLAMACPP_SWITCH_READY_TIMEOUT:-600}"
LLAMACPP_BIN="${LLAMACPP_BIN:-$(command -v llama-server 2>/dev/null || true)}"

# Pinned llama.cpp revision used only for convert_hf_to_gguf.py. This is build
# b11675, the first with ggml-org/llama.cpp#30621: current Switch checkpoints
# (MultiSwitch, which reserves two cache slots) need it, and every earlier
# converter, v0.7.0 included, writes a block count one too high, so the GGUF
# converts but llama-server refuses to load it. The GGUFs it writes load in a
# v0.6.0 server.
LLAMA_CPP_REVISION="${LLAMA_CPP_REVISION:-2851b3bcbf8d993a2243ae751c0a7dfa8e944a40}"

BUILD_CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/mellea/llamacpp-switch-build"
GGUF="$BUILD_CACHE/$(basename "$LLAMACPP_SWITCH_MODEL")-$LLAMACPP_SWITCH_QUANT.gguf"
BASE_URL="http://${LLAMACPP_SWITCH_HOST}:${LLAMACPP_SWITCH_PORT}/v1"
SERVER_PID=""
INTERRUPTED=""

# --- Preflight -------------------------------------------------------------
if [[ -z "$LLAMACPP_BIN" ]]; then
    die "llama-server not found on PATH.
  macOS:  brew install llama.cpp
  Linux:  download a release from https://github.com/ggml-org/llama.cpp/releases
  Then re-run, or point LLAMACPP_BIN at the binary."
fi
[[ -x "$LLAMACPP_BIN" ]] || die "LLAMACPP_BIN is not executable: $LLAMACPP_BIN"
command -v git >/dev/null || die "git is required (converter sources)."
command -v curl >/dev/null || die "curl is required (port and readiness checks)."
command -v uv >/dev/null || die "uv is required (converter dependencies)."

# Refuse to adopt or fight over an occupied port. Silently attaching to whatever
# is already listening there would be worse: a stale server on a different model
# looks like success but answers every request with the wrong weights. Any HTTP
# response counts as occupied, including the 503 a llama-server still loading
# its weights answers with; 000 means nothing could be connected to at all.
port_code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "${BASE_URL}/models" || true)"
if [[ "$port_code" != "000" ]]; then
    die "Something is already answering (HTTP ${port_code}) on port ${LLAMACPP_SWITCH_PORT}.
  If that is a llama-server still loading weights, wait for it to finish.
  Otherwise stop it, or set LLAMACPP_SWITCH_PORT to a free port."
fi

stop_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        log "Shutting down llama-server (PID $SERVER_PID)..."
        # A terminal Ctrl-C reaches the whole process group, so after a signal
        # llama-server is usually shutting down already. A second signal makes
        # it terminate immediately, which can trip a Metal teardown assert, so
        # give it a few seconds to exit on its own before sending one.
        if [[ -n "$INTERRUPTED" ]]; then
            for _ in 1 2 3 4 5; do
                kill -0 "$SERVER_PID" 2>/dev/null || break
                sleep 1
            done
        fi
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        log "llama-server stopped."
    fi
    SERVER_PID=""
}
trap 'stop_server; rm -f "$GGUF.partial"' EXIT
# Route signal kills through the EXIT trap so the server is always torn down.
trap 'INTERRUPTED=1; exit 130' INT
trap 'INTERRUPTED=1; exit 143' TERM

# --- Convert the checkpoint to GGUF (once) ---------------------------------
if [[ ! -f "$GGUF" ]]; then
    log "No GGUF at $GGUF; converting $LLAMACPP_SWITCH_MODEL (${LLAMACPP_SWITCH_QUANT})."

    LLAMA_CPP_DIR="$BUILD_CACHE/llama.cpp"
    CONVERTER_VENV="$BUILD_CACHE/converter-venv"
    SHADOW_DIR="$BUILD_CACHE/shadow-$(basename "$LLAMACPP_SWITCH_MODEL")"

    if [[ ! -d "$LLAMA_CPP_DIR/.git" ]]; then
        log "Fetching llama.cpp sources (pinned revision)..."
        git init --quiet "$LLAMA_CPP_DIR"
    fi
    # Idempotent: a no-op when the remote is already configured, which also
    # covers a checkout left behind by an interrupted earlier run.
    git -C "$LLAMA_CPP_DIR" remote add origin https://github.com/ggml-org/llama.cpp.git 2>/dev/null || true
    git -C "$LLAMA_CPP_DIR" fetch --quiet --depth 1 origin "$LLAMA_CPP_REVISION"
    git -C "$LLAMA_CPP_DIR" checkout --quiet --detach "$LLAMA_CPP_REVISION"

    if [[ ! -x "$CONVERTER_VENV/bin/python" ]]; then
        log "Creating converter venv (Python 3.12)..."
        # Pin the Python version: without this, uv picks 3.12 from the repo's
        # .python-version when run from the repo root but its own default
        # elsewhere, and the converter stack is only validated on 3.12.
        uv venv --python 3.12 "$CONVERTER_VENV"
    fi
    # Always run the install, not just on first creation: a previously failed
    # install would otherwise leave a venv that passes the existence check
    # above but is missing packages. With everything already installed this
    # returns in seconds.
    # huggingface_hub is installed explicitly: the checkpoint download below
    # runs on this venv, and the requirements file only pulls it in
    # transitively through transformers.
    # The requirements file pins transformers==4.57.6, which cannot load
    # current Switch tokenizers (their tokenizer_class, TokenizersBackend, is a
    # transformers 5 class), so override it to the range Mellea itself tests.
    # Running from the build cache keeps a surrounding project's uv settings
    # (mellea's own [tool.uv] override-dependencies, say) out of the resolve,
    # so the result is the same whatever directory the script is run from.
    CONVERTER_OVERRIDES="$BUILD_CACHE/converter-overrides.txt"
    echo 'transformers>5.5.0,<6.0.0' > "$CONVERTER_OVERRIDES"
    uv --directory "$BUILD_CACHE" pip install --python "$CONVERTER_VENV/bin/python" \
        --override "$CONVERTER_OVERRIDES" \
        -r "$LLAMA_CPP_DIR/requirements/requirements-convert_hf_to_gguf.txt" \
        huggingface_hub

    # Current transformers refuses the Switch checkpoints' tokenizer_config.json
    # because extra_special_tokens is a list; it expects a mapping. The control
    # tokens it lists are already in tokenizer.json, so a shadow copy of the
    # checkpoint with the key dropped converts fine and loses nothing.
    # snapshot_download writes a full copy of the checkpoint (~7 GB) that
    # persists next to the GGUF (~16 GB steady state), so re-converting with a
    # different LLAMACPP_SWITCH_QUANT later needs no re-download.
    log "Fetching checkpoint from the Hugging Face Hub..."
    mkdir -p "$SHADOW_DIR"
    # Run on the converter venv, not the caller's project environment: this
    # must not sync the user's mellea venv with every extra and group just to
    # download a checkpoint, and it keeps the script independent of cwd.
    "$CONVERTER_VENV/bin/python" - "$LLAMACPP_SWITCH_MODEL" "$SHADOW_DIR" <<'PY'
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
shadow = Path(sys.argv[2])
snapshot_download(
    repo_id=repo_id,
    local_dir=shadow,
    ignore_patterns=["*.gguf", "*.onnx", "*.tflite", "*.msgpack", "*.h5"],
)

# Drop the list-form extra_special_tokens that transformers rejects; see above.
config_path = shadow / "tokenizer_config.json"
config = json.loads(config_path.read_text())
if isinstance(config.get("extra_special_tokens"), list):
    del config["extra_special_tokens"]
    config_path.write_text(json.dumps(config))
PY

    log "Converting to GGUF (this takes a few minutes and ~9 GB of disk)..."
    # Write to a .partial file and move it into place on success: the converter
    # writes its output directly, so a Ctrl-C, full disk, or OOM kill midway
    # would otherwise leave a truncated GGUF that the cache check above accepts
    # on every later run. The EXIT trap removes this run's leftover .partial
    # file, and only that one, so a concurrent run for another model keeps its own.
    "$CONVERTER_VENV/bin/python" "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
        "$SHADOW_DIR" \
        --outtype "$LLAMACPP_SWITCH_QUANT" \
        --outfile "$GGUF.partial"
    mv "$GGUF.partial" "$GGUF"
fi

# The GGUF is a conversion of whatever checkpoint revision the first run
# downloaded. Upstream checkpoint updates are not picked up while this file
# exists; delete it to re-convert from the current revision.

# --- Serve ------------------------------------------------------------------
log "Starting llama-server"
log "  model:   $LLAMACPP_SWITCH_MODEL (${LLAMACPP_SWITCH_QUANT})"
log "  alias:   $LLAMACPP_ALIAS"
log "  address: ${LLAMACPP_SWITCH_HOST}:${LLAMACPP_SWITCH_PORT} (context ${LLAMACPP_SWITCH_CTX_SIZE})"

# --jinja is required: the Switch chat template injects the adapter control
# token from chat_template_kwargs.adapter_name, and Mellea's requests rely on
# the server rendering that template.
"$LLAMACPP_BIN" \
    -m "$GGUF" \
    --alias "$LLAMACPP_ALIAS" \
    --host "$LLAMACPP_SWITCH_HOST" \
    --port "$LLAMACPP_SWITCH_PORT" \
    --ctx-size "$LLAMACPP_SWITCH_CTX_SIZE" \
    --jinja &
SERVER_PID=$!

# --- Wait for readiness -----------------------------------------------------
# /health returns 503 while weights load and 200 once the server can serve, so
# it is the honest readiness signal. A bound port is not: it accepts connections
# well before the model is loaded.
log "Waiting for llama-server to be ready (timeout ${LLAMACPP_SWITCH_READY_TIMEOUT}s)..."
ready=0
for i in $(seq 1 "$LLAMACPP_SWITCH_READY_TIMEOUT"); do
    if curl -sf "http://${LLAMACPP_SWITCH_HOST}:${LLAMACPP_SWITCH_PORT}/health" >/dev/null 2>&1; then
        log "llama-server ready after ${i}s"
        ready=1
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        SERVER_PID=""
        die "llama-server died during startup (see its output above)."
    fi
    sleep 1
done
[[ "$ready" -eq 1 ]] || die "llama-server was not ready within ${LLAMACPP_SWITCH_READY_TIMEOUT}s."

cat <<EOF

llama-server is serving ${LLAMACPP_ALIAS} at ${BASE_URL}

Connect from Mellea (the OpenAI backend requires an api_key even though
llama-server ignores it, so pass a placeholder):

    from mellea.backends.openai import OpenAIBackend
    from mellea.formatters import TemplateFormatter
    from mellea.stdlib.components import Document, Message
    from mellea.stdlib.components.intrinsic import rag
    from mellea.stdlib.context import ChatContext

    backend = OpenAIBackend(
        model_id="$LLAMACPP_SWITCH_MODEL",
        formatter=TemplateFormatter(model_id="$LLAMACPP_SWITCH_MODEL"),
        base_url="${BASE_URL}",
        api_key="llamacpp",
        load_embedded_adapters=True,
    )
    context = ChatContext().add(Message("assistant", "Hello there, how can I help you?"))
    print(rag.check_answerability(
        "What is the square root of 4?",
        [Document("The square root of 4 is 2.")],
        context,
        backend,
    ))

See docs/examples/granite-switch/ for runnable examples.

Press Ctrl-C to stop the server.
EOF

# Hold the script open for the server's lifetime so Ctrl-C reaches the trap.
wait "$SERVER_PID"
