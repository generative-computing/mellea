#!/bin/bash
# run_tests_with_ollama.sh
# Starts a local ollama server (no sudo), pulls required models, runs tests,
# and shuts everything down cleanly.
#
# Usage:
#   ./run_tests_with_ollama.sh                              # run all tests
#   ./run_tests_with_ollama.sh -m ollama                    # only ollama tests
#   ./run_tests_with_ollama.sh --group-by-backend -v -s     # custom pytest args
#
# LSF example:
#   bsub -n 1 -G grp_preemptable -q preemptable \ # codespell:ignore
#     -gpu "num=1/task:mode=shared:j_exclusive=yes" \
#     "./run_tests_with_ollama.sh --group-by-backend -v -s"

set -euo pipefail

# --- Helper functions ---
log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { log "ERROR: $*" >&2; exit 1; }

# --- Configuration ---
OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1}"
OLLAMA_PORT="${OLLAMA_PORT:-11434}"
if [[ -n "${CACHE_DIR:-}" ]]; then
    OLLAMA_DIR="${CACHE_DIR}/ollama"
else
    log "WARNING: CACHE_DIR not set. Ollama models will download to ~/.ollama (default)"
    OLLAMA_DIR="$HOME/.ollama"
fi
OLLAMA_BIN="${OLLAMA_BIN:-$(command -v ollama 2>/dev/null || echo "$HOME/.local/bin/ollama")}"
OLLAMA_MODELS=(
    "granite4:micro"
    "granite4:micro-h"
    "granite3.2-vision"
)

# Log directory
LOGDIR="logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

cleanup() {
    log "Shutting down ollama server..."
    if [[ -n "${OLLAMA_PID:-}" ]] && kill -0 "$OLLAMA_PID" 2>/dev/null; then
        kill "$OLLAMA_PID" 2>/dev/null
        wait "$OLLAMA_PID" 2>/dev/null || true
    fi
    log "Ollama stopped."
}
trap cleanup EXIT

# --- Install ollama binary if missing ---
if [[ ! -x "$OLLAMA_BIN" ]]; then
    log "Ollama binary not found at $OLLAMA_BIN — downloading..."
    mkdir -p "$(dirname "$OLLAMA_BIN")"
    curl -fsSL https://ollama.com/download/ollama-linux-amd64 -o "$OLLAMA_BIN"
    chmod +x "$OLLAMA_BIN"
    log "Installed ollama to $OLLAMA_BIN"
fi

# --- Check if ollama is already running ---
if curl -sf "http://${OLLAMA_HOST}:${OLLAMA_PORT}/api/tags" >/dev/null 2>&1; then
    log "Ollama already running on ${OLLAMA_HOST}:${OLLAMA_PORT} — using existing server"
    OLLAMA_PID=""
else
    # Find a free port starting from OLLAMA_PORT
    while ss -tln 2>/dev/null | grep -q ":${OLLAMA_PORT} " || \
          netstat -tln 2>/dev/null | grep -q ":${OLLAMA_PORT} "; do
        log "Port $OLLAMA_PORT in use, trying $((OLLAMA_PORT + 1))..."
        OLLAMA_PORT=$((OLLAMA_PORT + 1))
    done

    # --- Start ollama server ---
    log "Starting ollama server on ${OLLAMA_HOST}:${OLLAMA_PORT}..."
    export OLLAMA_HOST="${OLLAMA_HOST}:${OLLAMA_PORT}"
    export OLLAMA_MODELS_DIR="${OLLAMA_DIR}/models"
    mkdir -p "$OLLAMA_MODELS_DIR"

    "$OLLAMA_BIN" serve > "$LOGDIR/ollama.log" 2>&1 &
    OLLAMA_PID=$!
    log "Ollama server PID: $OLLAMA_PID"

    # Wait for server to be ready
    log "Waiting for ollama to be ready..."
    for i in $(seq 1 30); do
        if curl -sf "http://127.0.0.1:${OLLAMA_PORT}/api/tags" >/dev/null 2>&1; then
            log "Ollama ready after ${i}s"
            break
        fi
        if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
            die "Ollama process died during startup. Check $LOGDIR/ollama.log"
        fi
        sleep 1
    done

    if ! curl -sf "http://127.0.0.1:${OLLAMA_PORT}/api/tags" >/dev/null 2>&1; then
        die "Ollama failed to start within 30s. Check $LOGDIR/ollama.log"
    fi
fi

# --- Pull required models ---
export OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}"
for model in "${OLLAMA_MODELS[@]}"; do
    if "$OLLAMA_BIN" list 2>/dev/null | grep -q "^${model}"; then
        log "Model $model already pulled"
    else
        log "Pulling $model ..."
        "$OLLAMA_BIN" pull "$model" 2>&1 | tail -1
    fi
done

log "All models ready."

# --- Run tests ---
log "Starting pytest..."
log "Log directory: $LOGDIR"
log "Pytest args: ${*:---group-by-backend -v}"

uv run --quiet --frozen --all-groups --all-extras \
    pytest test/ "${@:---group-by-backend -v}" \
    2>&1 | tee "$LOGDIR/pytest_full.log"

EXIT_CODE=${PIPESTATUS[0]}

log "Tests finished with exit code: $EXIT_CODE"
log "Logs: $LOGDIR/"
exit $EXIT_CODE