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
#   bsub -n 1 -G grp_preemptable -q preemptable \
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

# Log directory - use MELLEA_LOGDIR if set (from nightly.py), otherwise create standalone
if [[ -n "${MELLEA_LOGDIR:-}" ]]; then
    LOGDIR="$MELLEA_LOGDIR"
    log "Using provided log directory: $LOGDIR"
else
    LOGDIR="logs/$(date +%Y-%m-%d-%H:%M:%S)"
    log "Using standalone log directory: $LOGDIR"
fi
mkdir -p "$LOGDIR"

cleanup() {
    if [[ "${OLLAMA_EXTERNAL:-0}" == "1" ]]; then
        log "Ollama managed externally (OLLAMA_EXTERNAL=1) — skipping shutdown"
        return
    fi
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
    log "Ollama binary not found at $OLLAMA_BIN — downloading latest release..."
    OLLAMA_INSTALL_DIR="$(dirname "$OLLAMA_BIN")"
    mkdir -p "$OLLAMA_INSTALL_DIR"

    # Get latest release tag from GitHub API
    OLLAMA_VERSION=$(curl -fsSL https://api.github.com/repos/ollama/ollama/releases/latest \
        | grep '"tag_name"' | head -1 | cut -d'"' -f4)
    log "Latest ollama version: $OLLAMA_VERSION"

    DOWNLOAD_URL="https://github.com/ollama/ollama/releases/download/${OLLAMA_VERSION}/ollama-linux-amd64.tar.zst"
    log "Downloading from $DOWNLOAD_URL (includes CUDA libs, ~1.9GB)..."

    # Extract everything (bin/ollama + lib/ollama/cuda_v*/) into OLLAMA_INSTALL_DIR's parent
    # Archive structure: bin/ollama, lib/ollama/cuda_v12/*, lib/ollama/cuda_v13/*
    # Install into ~/.local/ so we get ~/.local/bin/ollama and ~/.local/lib/ollama/
    OLLAMA_PREFIX="$(dirname "$OLLAMA_INSTALL_DIR")"
    curl -fsSL "$DOWNLOAD_URL" | tar --use-compress-program=unzstd -x -C "$OLLAMA_PREFIX"
    chmod +x "$OLLAMA_BIN"
    log "Installed ollama $OLLAMA_VERSION to $OLLAMA_PREFIX (bin + CUDA libs)"
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

    # Ensure ollama can find system CUDA libraries
    if [[ -d "/usr/local/cuda" ]]; then
        export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
        log "Added system CUDA to LD_LIBRARY_PATH"
    fi

    "$OLLAMA_BIN" serve > "$LOGDIR/ollama.log" 2>&1 &
    OLLAMA_PID=$!
    log "Ollama server PID: $OLLAMA_PID"

    # Wait for server to be ready
    log "Waiting for ollama to be ready..."
    for i in $(seq 1 120); do
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

# --- Warm up models (first load into memory is slow) ---
if [[ "${OLLAMA_SKIP_WARMUP:-0}" == "1" ]]; then
    log "Skipping model warmup (OLLAMA_SKIP_WARMUP=1)"
else
    log "Warming up models..."
    for model in "${OLLAMA_MODELS[@]}"; do
        log "  Warming $model ..."
        curl -sf "http://127.0.0.1:${OLLAMA_PORT}/api/generate" \
            -d "{\"model\": \"$model\", \"prompt\": \"hi\", \"stream\": false}" \
            -o /dev/null --max-time 120 || log "  Warning: warmup for $model timed out (will load on first test)"
    done
    log "Warmup complete."
fi

# --- Run tests ---
log "Starting pytest..."
log "Log directory: $LOGDIR"
log "Pytest args: ${*---group-by-backend}"
${UV_PYTHON:+log "Python version: $UV_PYTHON"}

# Use UV_PYTHON env var if set, otherwise use default Python
UV_PYTHON_ARG=""
if [[ -n "${UV_PYTHON:-}" ]]; then
    UV_PYTHON_ARG="--python $UV_PYTHON"
fi

uv run --quiet --frozen --all-groups --all-extras $UV_PYTHON_ARG \
    pytest test/ ${@---group-by-backend} \
    2>&1 | tee "$LOGDIR/pytest_full.log"

EXIT_CODE=${PIPESTATUS[0]}

log "Tests finished with exit code: $EXIT_CODE"
log "Logs: $LOGDIR/"
exit $EXIT_CODE