#!/bin/bash
# run_tests_with_ollama_and_vllm.sh
# Starts a local ollama server (no sudo), optionally starts a local vLLM server,
# pulls/installs required models, runs tests, and shuts everything down cleanly.
#
# vLLM is enabled automatically when a CUDA GPU is detected, or explicitly with
# WITH_VLLM=1. Override with WITH_VLLM=0 to force-disable even on GPU hosts.
#
# Usage:
#   ./run_tests_with_ollama_and_vllm.sh                              # auto (vLLM on GPU hosts)
#   ./run_tests_with_ollama_and_vllm.sh -m ollama                    # only ollama tests
#   ./run_tests_with_ollama_and_vllm.sh --group-by-backend -v -s     # custom pytest args
#   WITH_VLLM=1 ./run_tests_with_ollama_and_vllm.sh                  # force-enable vLLM
#   WITH_VLLM=0 ./run_tests_with_ollama_and_vllm.sh                  # force-disable vLLM
#   SKIP_WARMUP=1 ./run_tests_with_ollama_and_vllm.sh                 # skip ollama model warmup
#   WITH_VLLM=1 VLLM_MODEL=ibm-granite/granite-3.3-8b-instruct \
#     ./run_tests_with_ollama_and_vllm.sh --group-by-backend -v -s
#
# LSF example:
#   bsub -n 1 -G grp_preemptable -q preemptable \
#     -gpu "num=1/task:mode=shared:j_exclusive=yes" \
#     "./run_tests_with_ollama_and_vllm.sh --group-by-backend -v -s"

set -euo pipefail

# --- Helper functions ---
log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { log "ERROR: $*" >&2; exit 1; }

# --- Ollama configuration ---
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

# --- vLLM configuration ---
# Auto-enable vLLM when a CUDA GPU is available (nvidia-smi honours CUDA_VISIBLE_DEVICES
# so this is safe on multi-tenant LSF hosts). Override with WITH_VLLM=0 or WITH_VLLM=1.
if [[ -z "${WITH_VLLM:-}" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        WITH_VLLM=1
        log "CUDA GPU detected — enabling vLLM (set WITH_VLLM=0 to disable)"
    else
        WITH_VLLM=0
    fi
fi
VLLM_PORT="${VLLM_PORT:-8100}"
VLLM_MODEL="${VLLM_MODEL:-ibm-granite/granite-4.0-micro}"
VLLM_GPU_MEM="${VLLM_GPU_MEM:-0.4}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}"
VLLM_VENV="${CACHE_DIR:+${CACHE_DIR}/.vllm-venv}"
VLLM_VENV="${VLLM_VENV:-.vllm-venv}"
VLLM_PID=""

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
    else
        log "Shutting down ollama server..."
        if [[ -n "${OLLAMA_PID:-}" ]] && kill -0 "$OLLAMA_PID" 2>/dev/null; then
            kill "$OLLAMA_PID" 2>/dev/null
            wait "$OLLAMA_PID" 2>/dev/null || true
        fi
        log "Ollama stopped."
    fi

    if [[ "$WITH_VLLM" == "1" ]]; then
        if [[ "${VLLM_EXTERNAL:-0}" == "1" ]]; then
            log "vLLM managed externally (VLLM_EXTERNAL=1) — skipping shutdown"
        elif [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
            log "Shutting down vLLM server..."
            kill "$VLLM_PID" 2>/dev/null
            wait "$VLLM_PID" 2>/dev/null || true
            log "vLLM stopped."
        fi
        if [[ "${KEEP_VLLM_VENV:-0}" != "1" ]]; then
            rm -rf "$VLLM_VENV"
        fi
    fi
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

log "All ollama models ready."

# --- Warm up models (first load into memory is slow) ---
# Disable with SKIP_WARMUP=1 (covers all backends) or OLLAMA_SKIP_WARMUP=1 (ollama only).
# Note: vLLM has no warmup step — it serves immediately after the readiness check.
if [[ "${SKIP_WARMUP:-0}" == "1" || "${OLLAMA_SKIP_WARMUP:-0}" == "1" ]]; then
    log "Skipping model warmup"
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

# --- vLLM server (optional, auto-enabled on CUDA hosts) ---
if [[ "$WITH_VLLM" == "1" ]]; then
    if [[ "${VLLM_EXTERNAL:-0}" == "1" ]]; then
        log "vLLM managed externally (VLLM_EXTERNAL=1) — skipping startup"
        if ! curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
            die "VLLM_EXTERNAL=1 but no vLLM server found on port ${VLLM_PORT}"
        fi
    else
        # Find a free port starting from VLLM_PORT
        while ss -tln 2>/dev/null | grep -q ":${VLLM_PORT} " || \
              netstat -tln 2>/dev/null | grep -q ":${VLLM_PORT} "; do
            log "Port $VLLM_PORT in use, trying $((VLLM_PORT + 1))..."
            VLLM_PORT=$((VLLM_PORT + 1))
        done

        # Install vLLM into an isolated venv so it never enters mellea's own deps
        if [[ -d "$VLLM_VENV" ]] && [[ "${KEEP_VLLM_VENV:-0}" == "1" ]]; then
            log "Reusing existing vLLM venv at $VLLM_VENV (KEEP_VLLM_VENV=1)"
        else
            log "Creating isolated vLLM venv at $VLLM_VENV ..."
            uv venv "$VLLM_VENV" --python 3.11 --clear
            log "Installing vllm into $VLLM_VENV ..."
            uv pip install --python "$VLLM_VENV/bin/python" vllm \
                > "$LOGDIR/vllm_install.log" 2>&1 \
                || die "vllm install failed. Check $LOGDIR/vllm_install.log"
            log "vllm installed."
        fi

        # Start vllm serve in the background
        log "Starting vLLM server — model: $VLLM_MODEL, port: $VLLM_PORT ..."
        "$VLLM_VENV/bin/python" -m vllm.entrypoints.openai.api_server \
            --model "$VLLM_MODEL" \
            --port "$VLLM_PORT" \
            --gpu-memory-utilization "$VLLM_GPU_MEM" \
            --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
            --max-model-len "$VLLM_MAX_MODEL_LEN" \
            > "$LOGDIR/vllm.log" 2>&1 &
        VLLM_PID=$!
        log "vLLM server PID: $VLLM_PID"

        # Wait for vLLM to be ready (model load can take 60-90s)
        log "Waiting for vLLM to be ready..."
        for i in $(seq 1 120); do
            if curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
                log "vLLM ready after ${i}s"
                break
            fi
            if ! kill -0 "$VLLM_PID" 2>/dev/null; then
                die "vLLM process died during startup. Check $LOGDIR/vllm.log"
            fi
            sleep 1
        done

        if ! curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
            die "vLLM failed to start within 120s. Check $LOGDIR/vllm.log"
        fi
    fi

    # Export for pytest fixtures
    export VLLM_TEST_BASE_URL="http://127.0.0.1:${VLLM_PORT}"
    export VLLM_TEST_MODEL="$VLLM_MODEL"
    export VLLM_VENV_PATH="$VLLM_VENV"
    log "vLLM ready. VLLM_TEST_BASE_URL=$VLLM_TEST_BASE_URL"
else
    log "vLLM disabled (WITH_VLLM=0). Pass WITH_VLLM=1 to enable, or run on a CUDA host for auto-detection."
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