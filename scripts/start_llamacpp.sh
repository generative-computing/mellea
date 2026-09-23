#!/usr/bin/env bash
# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# start_llamacpp.sh
# Starts a local llama.cpp server (llama-server) hosting Granite 4.2 3b, waits
# until it is actually ready to serve, and prints the Mellea snippet to connect
# to it. Runs in the foreground; Ctrl-C shuts the server down.
#
# Why this exists: llama-server speaks the OpenAI chat-completions API, so
# Mellea reaches it through the existing `openai` backend with a `base_url`.
# No new backend class is involved.
#
# Usage:
#   ./scripts/start_llamacpp.sh                     # Granite 4.2 3b, Q4_K_M, port 8080
#   LLAMACPP_PORT=9090 ./scripts/start_llamacpp.sh   # different port
#   LLAMACPP_QUANT=Q8_0 ./scripts/start_llamacpp.sh  # higher-fidelity quant (3.9 GB)
#   LLAMACPP_CTX_SIZE=8192 ./scripts/start_llamacpp.sh
#
# Requires `llama-server` on PATH:
#   macOS:  brew install llama.cpp
#   Linux:  https://github.com/ggml-org/llama.cpp/releases (or build from source)

set -euo pipefail

log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { log "ERROR: $*" >&2; exit 1; }

# --- Configuration ---
# The GGUF repo holding IBM's official quantized builds. Its Q4_K_M weights
# correspond to IBM_GRANITE_4_2_3B, which is what `start_session` defaults to.
LLAMACPP_MODEL_REPO="${LLAMACPP_MODEL_REPO:-ibm-granite/granite-4.2-3b-GGUF}"
LLAMACPP_QUANT="${LLAMACPP_QUANT:-Q4_K_M}"
# The name llama-server reports on /v1/models and accepts in requests. Must match
# what Mellea sends: `_resolve_model_id_str` maps a ModelIdentifier for the openai
# backend via `openai_name`, which is unset on the Granite ids, so it falls back to
# `hf_model_name`. Keeping the alias equal to hf_model_name lets callers pass
# IBM_GRANITE_4_2_3B straight through with no string juggling.
LLAMACPP_ALIAS="${LLAMACPP_ALIAS:-ibm-granite/granite-4.2-3b}"
LLAMACPP_HOST="${LLAMACPP_HOST:-127.0.0.1}"
LLAMACPP_PORT="${LLAMACPP_PORT:-8080}"
# The model advertises 131072, but a full-length KV cache is a lot of RAM for a
# dev server. 32k covers Mellea's prompts with room to spare; raise if you need it.
LLAMACPP_CTX_SIZE="${LLAMACPP_CTX_SIZE:-32768}"
# Readiness timeout in seconds, measured from process start. A warm cache loads a
# 3b in a few seconds; a cold start also downloads ~2.2 GB of weights, so the
# default leaves room for the download on a slow link. The loop exits on readiness,
# so a generous default costs nothing once the weights are cached.
LLAMACPP_READY_TIMEOUT="${LLAMACPP_READY_TIMEOUT:-600}"
LLAMACPP_BIN="${LLAMACPP_BIN:-$(command -v llama-server 2>/dev/null || true)}"

BASE_URL="http://${LLAMACPP_HOST}:${LLAMACPP_PORT}/v1"
SERVER_PID=""

# --- Preflight ---
if [[ -z "$LLAMACPP_BIN" ]]; then
    die "llama-server not found on PATH.
  macOS:  brew install llama.cpp
  Linux:  download a release from https://github.com/ggml-org/llama.cpp/releases
  Then re-run, or point LLAMACPP_BIN at the binary."
fi
[[ -x "$LLAMACPP_BIN" ]] || die "LLAMACPP_BIN is not executable: $LLAMACPP_BIN"

# Refuse to adopt or fight over an occupied port. Silently attaching to whatever
# is already listening there would be worse: a stale server on a different model
# looks like success but answers every request with the wrong weights.
if curl -sf "${BASE_URL}/models" >/dev/null 2>&1; then
    die "Something is already serving an OpenAI-compatible API on port ${LLAMACPP_PORT}.
  Stop it, or set LLAMACPP_PORT to a free port."
fi

stop_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        log "Shutting down llama-server (PID $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        log "llama-server stopped."
    fi
    SERVER_PID=""
}
trap stop_server EXIT
# Route signal kills through the EXIT trap so the server is always torn down.
trap 'exit 130' INT
trap 'exit 143' TERM

# --- Start ---
log "Starting llama-server"
log "  model:   ${LLAMACPP_MODEL_REPO}:${LLAMACPP_QUANT}"
log "  alias:   ${LLAMACPP_ALIAS}"
log "  address: ${LLAMACPP_HOST}:${LLAMACPP_PORT} (context ${LLAMACPP_CTX_SIZE})"
log "First run downloads the weights to \${LLAMA_CACHE:-~/.cache/llama.cpp} (~2.2 GB for Q4_K_M)."

# --jinja is required for Granite's chat template and tool calling. It defaults to
# enabled on recent builds, but passing it explicitly keeps the script correct on
# older ones rather than silently falling back to a generic template.
"$LLAMACPP_BIN" \
    -hf "${LLAMACPP_MODEL_REPO}:${LLAMACPP_QUANT}" \
    --alias "$LLAMACPP_ALIAS" \
    --host "$LLAMACPP_HOST" \
    --port "$LLAMACPP_PORT" \
    --ctx-size "$LLAMACPP_CTX_SIZE" \
    --jinja &
SERVER_PID=$!

# --- Wait for readiness ---
# /health returns 503 while weights load and 200 once the server can serve, so it
# is the honest readiness signal. A bound port is not: it accepts connections well
# before the model is loaded.
log "Waiting for llama-server to be ready (timeout ${LLAMACPP_READY_TIMEOUT}s)..."
ready=0
for i in $(seq 1 "$LLAMACPP_READY_TIMEOUT"); do
    if curl -sf "http://${LLAMACPP_HOST}:${LLAMACPP_PORT}/health" >/dev/null 2>&1; then
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
[[ "$ready" -eq 1 ]] || die "llama-server was not ready within ${LLAMACPP_READY_TIMEOUT}s."

cat <<EOF

llama-server is serving ${LLAMACPP_ALIAS} at ${BASE_URL}

Connect from Mellea (the openai backend requires an api_key even though
llama-server ignores it, so pass a placeholder):

    from mellea import start_session
    from mellea.backends.model_ids import IBM_GRANITE_4_2_3B

    session = start_session(
        "openai",
        IBM_GRANITE_4_2_3B,
        base_url="${BASE_URL}",
        api_key="llamacpp",
    )
    print(session.instruct("Write a haiku about local inference."))

Or point the env at it and let the backend pick it up:

    export OPENAI_BASE_URL="${BASE_URL}"
    export OPENAI_API_KEY="llamacpp"

Press Ctrl-C to stop the server.
EOF

# Hold the script open for the server's lifetime so Ctrl-C reaches the trap.
wait "$SERVER_PID"
