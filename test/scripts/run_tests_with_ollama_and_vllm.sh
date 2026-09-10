#!/bin/bash
# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# run_tests_with_ollama_and_vllm.sh
# Starts a local ollama server (no sudo), optionally starts a local vLLM server,
# pulls/installs required models, runs tests, and shuts everything down cleanly.
#
# vLLM is enabled automatically when a CUDA GPU is detected, or explicitly with
# WITH_VLLM=1. Override with WITH_VLLM=0 to force-disable even on GPU hosts.
#
# Usage:
#   ./run_tests_with_ollama_and_vllm.sh                              # auto (vLLM on GPU hosts), phased (default)
#   ./run_tests_with_ollama_and_vllm.sh -k test_ollama               # selection/verbosity args only
#   SERIAL_PHASES=0 ./run_tests_with_ollama_and_vllm.sh -m ollama    # legacy mode honours -m (single process)
#   SERIAL_PHASES=0 ./run_tests_with_ollama_and_vllm.sh --group-by-backend -v -s
#   WITH_VLLM=1 ./run_tests_with_ollama_and_vllm.sh                  # force-enable vLLM
#   WITH_VLLM=0 ./run_tests_with_ollama_and_vllm.sh                  # force-disable vLLM
#   SKIP_WARMUP=1 ./run_tests_with_ollama_and_vllm.sh                 # skip ollama model warmup
#   WITH_EXAMPLES=1 ./run_tests_with_ollama_and_vllm.sh               # include docs/examples/
#   WITH_TOOLING_TESTS=1 ./run_tests_with_ollama_and_vllm.sh          # include test/tooling/
#   WITH_VLLM=1 VLLM_MODEL=ibm-granite/granite-4.2-3b \
#     ./run_tests_with_ollama_and_vllm.sh --group-by-backend -v -s
# Execution modes
#   Phased (DEFAULT, SERIAL_PHASES=1): each backend group runs as its own
#   pytest process with per-phase server lifecycles — only one CUDA context
#   is alive at a time. Peak GPU memory drops from ~72 GiB (both servers up)
#   to ~33 GiB (the vLLM phase), in-process HF models are released between
#   phases, and the single-context phases (hf/vllm/base) can run on an
#   exclusive GPU. SERIAL_PHASES=0 selects the legacy single-pytest-process
#   run with both servers up for the whole run.
#   PHASES=hf,ollama,vllm,base (default "all") selects which phases run, so
#   one script can drive separate jobs with different -gpu requests.
#
# Caller arguments
#   Pass only test-selection (-k) and verbosity arguments.
#   Full per-test durations and a JSON report are recorded under the log
#   directory by the script itself (override with your own
#   --durations/--json-report if needed):
#     single-run:  pytest_full.log + pytest_report.json
#     phased:      pytest_full.log (all phases appended) +
#                  phase_<name>.log + pytest_report_<name>.json per phase
#                  (+ gpu_release_gate.log if a release gate times out)
#   NOTE: a caller -m conflicts with the per-phase marker selection, so the
#   script falls back to single-run mode with a warning.
#   NOTE: a caller-supplied --json-report-file in phased mode is NOT written
#   to directly by each phase (that would let phase 4 silently overwrite
#   phases 1-3). Instead each phase keeps writing its own
#   pytest_report_<name>.json, and after all phases finish they are merged
#   into the caller's requested path — so a driver like mellea-oc's
#   lsf/nightly.py, which passes one --json-report-file for the whole run,
#   still gets every phase's failures in it.
#
# LSF examples
#   Single job (shared GPU request; both/phase servers as needed):
#     bsub -n 1 -G grp_runtime -q normal \
#       -gpu "num=1/task:mode=shared:gmem=65G" \
#       "./run_tests_with_ollama_and_vllm.sh"
#   Two jobs (recommended: exclusive GPU for the single-context phases,
#   shared for the Ollama phase, whose per-model llama-server workers need
#   multi-process GPU access and cannot run under exclusive mode). Each job
#   needs its own MELLEA_LOGDIR and COVERAGE_FILE:
#     bsub -n 1 -G grp_runtime -q normal -gpu "num=1" \
#       "MELLEA_LOGDIR=logs/nightly-hf COVERAGE_FILE=.coverage.hf \
#        PHASES=hf,vllm,base ./run_tests_with_ollama_and_vllm.sh"
#     bsub -n 1 -G grp_runtime -q normal \
#       -gpu "num=1/task:mode=shared:gmem=20G" \
#       "MELLEA_LOGDIR=logs/nightly-ollama COVERAGE_FILE=.coverage.ollama \
#        PHASES=ollama ./run_tests_with_ollama_and_vllm.sh"
#   Legacy single-process run:
#     bsub -n 1 -G grp_runtime -q normal \
#       -gpu "num=1/task:mode=shared:gmem=65G" \
#       "SERIAL_PHASES=0 ./run_tests_with_ollama_and_vllm.sh --group-by-backend"

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
OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-2048}"
# Keep in sync with the models the test suite actually requests.
# llama3.2:1b is used by the SOFAI tests (test/stdlib/sampling/test_sofai_*.py);
# it is NOT the same tag as plain "llama3.2" (3B) and must be pulled explicitly,
# otherwise the server pulls it on demand mid-test (hidden network dependency).
# Models used only by docs/examples/ are intentionally excluded — the nightly
# does not run examples (WITH_EXAMPLES=0).
OLLAMA_MODEL_LIST=(
    "granite4.2:3b"
    "granite4:micro-h"
    "hf.co/ibm-granite/granite-vision-4.1-4b-GGUF:Q4_K_M"
    "llama3.2:1b"
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
VLLM_MODEL="${VLLM_MODEL:-ibm-granite/granite-4.2-3b}"
VLLM_GPU_MEM="${VLLM_GPU_MEM:-0.4}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}"
# Readiness timeout in seconds. vLLM startup (package import, weight load,
# torch.compile, CUDA graph capture, API server startup) measured 54-181 s
# on H100s depending on compile-cache warmth and concurrent starts; the
# default covers the cold-start worst case. The loop breaks on readiness, so
# a high default costs nothing on a fast start.
VLLM_READY_TIMEOUT="${VLLM_READY_TIMEOUT:-300}"
VLLM_VENV="${CACHE_DIR:+${CACHE_DIR}/.vllm-venv}"
VLLM_VENV="${VLLM_VENV:-.vllm-venv}"
VLLM_PID=""
# Set to 1 when THIS invocation created the venv; stop_vllm only removes a
# venv it owns, so concurrent jobs sharing the venv path (e.g. the
# recommended two-job split) never delete each other's live vLLM install.
VLLM_VENV_OWNED=0

# SERIAL_PHASES: phased execution is the DEFAULT. Each backend group runs as
# its own pytest process with per-phase server lifecycles (only one CUDA
# context alive at a time): peak GPU memory drops from ~72 GiB (both servers
# up) to ~33 GiB (the vLLM phase), in-process HF models are released between
# phases, and the single-context phases can run on an exclusive GPU.
# Set SERIAL_PHASES=0 for the legacy single-pytest-process run with both
# servers up for the whole run.
SERIAL_PHASES="${SERIAL_PHASES:-1}"
# PHASES: comma list of which serial phases to run (hf, ollama, vllm, base).
# Default "all". Lets one script drive subsets across separate LSF jobs with
# different -gpu requests (e.g. exclusive for hf/vllm/base, mode=shared for
# ollama, whose per-model llama-server workers need multi-process GPU access).
PHASES="${PHASES:-all}"

# phase_enabled NAME — true if PHASES includes NAME (or PHASES=all).
phase_enabled() {
    [[ "$PHASES" == "all" || ",$PHASES," == *",$1,"* ]]
}

if [[ "$SERIAL_PHASES" == "1" ]]; then
    IFS=',' read -r -a _PHASE_TOKENS <<< "$PHASES"
    if [[ ${#_PHASE_TOKENS[@]} -eq 0 ]]; then
        die "PHASES is empty; expected 'all' or a comma-separated subset of hf,ollama,vllm,base"
    fi
    for _t in "${_PHASE_TOKENS[@]}"; do
        case "$_t" in
            hf|ollama|vllm|base|all) ;;
            *) die "Unknown PHASES token '$_t' (expected: all, or a comma-separated subset of hf,ollama,vllm,base)" ;;
        esac
    done
fi
# GPU release gate (serial mode): the GPU counts as released once
# memory.used drops below GPU_FREE_THRESHOLD_MB (MiB); wait at most
# GPU_FREE_TIMEOUT seconds before warning and continuing.
GPU_FREE_THRESHOLD_MB="${GPU_FREE_THRESHOLD_MB:-2048}"
GPU_FREE_TIMEOUT="${GPU_FREE_TIMEOUT:-120}"

# Log directory - use MELLEA_LOGDIR if set (from nightly.py), otherwise create standalone
if [[ -n "${MELLEA_LOGDIR:-}" ]]; then
    LOGDIR="$MELLEA_LOGDIR"
    log "Using provided log directory: $LOGDIR"
else
    LOGDIR="logs/$(date +%Y-%m-%d-%H:%M:%S)"
    log "Using standalone log directory: $LOGDIR"
fi
mkdir -p "$LOGDIR"

stop_ollama() {
    if [[ "${OLLAMA_EXTERNAL:-0}" == "1" ]]; then
        log "Ollama managed externally (OLLAMA_EXTERNAL=1) — skipping shutdown"
    elif [[ -n "${OLLAMA_PID:-}" ]] && kill -0 "$OLLAMA_PID" 2>/dev/null; then
        log "Shutting down ollama server..."
        kill "$OLLAMA_PID" 2>/dev/null
        wait "$OLLAMA_PID" 2>/dev/null || true
        # Ollama's llama-server workers are children of the serve process and
        # survive its SIGTERM, holding VRAM as orphans. Kill any still alive,
        # scoped to this install's runtime dir. NOTE: the scope is per
        # OLLAMA_BIN — concurrent same-user runs must use distinct OLLAMA_BIN
        # values (e.g. under separate CACHE_DIRs), because with the shared
        # $HOME default one run's cleanup would kill the other's workers.
        local lib_dir
        lib_dir="$(dirname "$(dirname "$OLLAMA_BIN")")/lib/ollama"
        if pkill -f "${lib_dir}/llama-server" 2>/dev/null; then
            log "Killed orphaned llama-server worker(s) from ${lib_dir}"
        fi
        log "Ollama stopped."
    else
        log "Ollama not running — nothing to stop."
    fi
    OLLAMA_PID=""
}

stop_vllm() {
    if [[ "$WITH_VLLM" != "1" ]]; then
        return 0
    fi
    if [[ "${VLLM_EXTERNAL:-0}" == "1" ]]; then
        log "vLLM managed externally (VLLM_EXTERNAL=1) — skipping shutdown"
    elif [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        log "Shutting down vLLM server..."
        kill "$VLLM_PID" 2>/dev/null
        wait "$VLLM_PID" 2>/dev/null || true
        log "vLLM stopped."
    fi
    VLLM_PID=""
    # Only remove a venv this invocation created: concurrent jobs sharing the
    # venv path must not delete the venv a sibling job is running from.
    # Reused venvs are kept for the next run.
    if [[ "$VLLM_VENV_OWNED" == "1" && "${KEEP_VLLM_VENV:-0}" != "1" ]]; then
        rm -rf "$VLLM_VENV"
        VLLM_VENV_OWNED=0
    fi
}

cleanup() {
    stop_ollama
    stop_vllm
}
trap cleanup EXIT
# Route signal kills (bkill, wall-time limit) through the EXIT trap so the
# servers are torn down instead of being left holding VRAM.
trap 'exit 130' INT
trap 'exit 143' TERM

# wait_gpu_free LABEL
# Serial-phase gate: after a server is stopped, wait until the GPU's
# memory.used drops below GPU_FREE_THRESHOLD_MB. On timeout, log the
# resident compute processes and continue — the gate is diagnostic, not
# fatal. On a shared GPU the resident memory may belong to co-tenant jobs,
# in which case the timeout warning is expected, not a leak.
wait_gpu_free() {
    local label="$1"
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        log "GPU release gate skipped (nvidia-smi unavailable) after ${label}"
        return 0
    fi
    local deadline=$(( $(date +%s) + GPU_FREE_TIMEOUT )) used
    while :; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '[:space:]') || used=""
        if [[ -n "$used" && "$used" -lt "$GPU_FREE_THRESHOLD_MB" ]]; then
            log "GPU released after ${label} (memory.used=${used} MiB)"
            return 0
        fi
        if (( $(date +%s) >= deadline )); then
            log "WARNING: GPU still in use ${GPU_FREE_TIMEOUT}s after ${label} (memory.used=${used:-unknown} MiB) — resident processes logged to gpu_release_gate.log; expected on a shared GPU, otherwise a possible leaked context. Continuing."
            nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader 2>/dev/null | tee -a "$LOGDIR/gpu_release_gate.log" || true
            return 0
        fi
        log "Waiting for GPU release after ${label} (memory.used=${used:-unknown} MiB)..."
        sleep 5
    done
}

# run_phase NAME PYTEST_ARGS...
# Run one backend group as its own pytest process (serial mode). Each phase
# writes phase_NAME.log + pytest_report_NAME.json and appends to
# pytest_full.log for a single consolidated view.
OVERALL_RC=0
RAN_ANY=0
run_phase() {
    local name="$1"; shift
    local report_args=()
    if [[ $CALLER_JSON_REPORT -eq 0 ]]; then
        report_args=(--json-report --json-report-file="$LOGDIR/pytest_report_${name}.json")
    fi
    log "=== PHASE ${name}: pytest $* ==="
    local rc=0
    set +e
    uv run --quiet --frozen --all-groups --all-extras $UV_PYTHON_ARG \
        pytest "$PYTEST_DIR" "$@" "${DURATIONS_ARG[@]}" "${report_args[@]}" \
        2>&1 | tee "$LOGDIR/phase_${name}.log" | tail -3
    rc=${PIPESTATUS[0]}
    set -e
    if [[ $rc -eq 5 ]]; then
        # Exit 5 = nothing selected in this phase (e.g. a -k that only
        # matches another phase) — not a failure.
        log "PHASE ${name}: no tests selected — treating as pass"
        rc=0
    fi
    RAN_ANY=1
    cat "$LOGDIR/phase_${name}.log" >> "$LOGDIR/pytest_full.log"
    log "PHASE ${name} exit code: ${rc}"
    if (( OVERALL_RC == 0 && rc != 0 )); then
        OVERALL_RC=$rc
    fi
}

# --- Install a compatible Ollama binary ---
# Only when this run will use Ollama: the recommended two-job split runs the
# ollama phase in a single job, and concurrent jobs sharing one install path
# race on download+extract.
if [[ "$SERIAL_PHASES" != "1" ]] || phase_enabled ollama; then
    OLLAMA_MIN_VERSION="${OLLAMA_MIN_VERSION:-0.32.2}"
    ollama_current_version=""
    if [[ -x "$OLLAMA_BIN" ]]; then
        # `ollama --version` prints a multi-line warning block to stdout
        # (e.g. "Warning: client version is 0.32.2"); extract the first
        # dotted version number rather than trusting the last field of the
        # last line, which yields "instance\n0.32.2" and always fails the
        # version comparison below.
        ollama_current_version=$("$OLLAMA_BIN" --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    fi

    if [[ ! -x "$OLLAMA_BIN" ]] || ! printf '%s\n%s\n' "$OLLAMA_MIN_VERSION" "$ollama_current_version" | sort -V -C; then
        log "Installing Ollama $OLLAMA_MIN_VERSION (current: ${ollama_current_version:-missing})..."
        OLLAMA_INSTALL_DIR="$(dirname "$OLLAMA_BIN")"
        mkdir -p "$OLLAMA_INSTALL_DIR"

        DOWNLOAD_URL="https://github.com/ollama/ollama/releases/download/v${OLLAMA_MIN_VERSION}/ollama-linux-amd64.tar.zst"
        log "Downloading from $DOWNLOAD_URL (includes CUDA libs, ~1.9GB)..."

        # Extract everything (bin/ollama + lib/ollama/cuda_v*/) into OLLAMA_INSTALL_DIR's parent
        # Archive structure: bin/ollama, lib/ollama/cuda_v12/*, lib/ollama/cuda_v13/*
        # Install into ~/.local/ so we get ~/.local/bin/ollama and ~/.local/lib/ollama/
        OLLAMA_PREFIX="$(dirname "$OLLAMA_INSTALL_DIR")"
        curl -fsSL "$DOWNLOAD_URL" | tar --use-compress-program=unzstd -x -C "$OLLAMA_PREFIX"
        chmod +x "$OLLAMA_BIN"
        log "Installed Ollama $OLLAMA_MIN_VERSION to $OLLAMA_PREFIX (bin + CUDA libs)"
    fi
fi

# start_ollama: start (or adopt) the Ollama server, pull models, warm up.
start_ollama() {
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
        export OLLAMA_MODELS="${OLLAMA_DIR}/models"
        export OLLAMA_CONTEXT_LENGTH
        mkdir -p "$OLLAMA_MODELS"

        # Ensure ollama can find system CUDA libraries
        if [[ -d "/usr/local/cuda" ]]; then
            export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
            log "Added system CUDA to LD_LIBRARY_PATH"
        fi

        log "Using Ollama default context length: $OLLAMA_CONTEXT_LENGTH"
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
            die "Ollama failed to start within 120s. Check $LOGDIR/ollama.log"
        fi
    fi

    # --- Pull required models ---
    export OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}"
    for model in "${OLLAMA_MODEL_LIST[@]}"; do
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
        for model in "${OLLAMA_MODEL_LIST[@]}"; do
            log "  Warming $model ..."
            curl -sf "http://127.0.0.1:${OLLAMA_PORT}/api/generate" \
                -d "{\"model\": \"$model\", \"prompt\": \"hi\", \"stream\": false}" \
                -o /dev/null --max-time 120 || log "  Warning: warmup for $model timed out (will load on first test)"
        done
        log "Warmup complete."
    fi
}

# start_vllm: start (or adopt) the vLLM server and export the test env.
start_vllm() {
    if [[ "${VLLM_EXTERNAL:-0}" == "1" ]]; then
        log "vLLM managed externally (VLLM_EXTERNAL=1) — skipping startup"
        if ! curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
            log "ERROR: VLLM_EXTERNAL=1 but no vLLM server found on port ${VLLM_PORT}"
            return 1
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
            uv venv "$VLLM_VENV" --python 3.12 --clear
            log "Installing vllm into $VLLM_VENV ..."
            if ! uv pip install --python "$VLLM_VENV/bin/python" vllm \
                > "$LOGDIR/vllm_install.log" 2>&1; then
                log "ERROR: vllm install failed. Check $LOGDIR/vllm_install.log"
                return 1
            fi
            log "vllm installed."
            VLLM_VENV_OWNED=1
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

        # Wait for vLLM to be ready (startup can exceed 120s on modern vLLM)
        log "Waiting for vLLM to be ready (timeout ${VLLM_READY_TIMEOUT}s)..."
        for i in $(seq 1 "$VLLM_READY_TIMEOUT"); do
            if curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
                log "vLLM ready after ${i}s"
                break
            fi
            if ! kill -0 "$VLLM_PID" 2>/dev/null; then
                log "ERROR: vLLM process died during startup. Check $LOGDIR/vllm.log"
                return 1
            fi
            sleep 1
        done

        if ! curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
            log "ERROR: vLLM failed to start within ${VLLM_READY_TIMEOUT}s. Check $LOGDIR/vllm.log"
            return 1
        fi
    fi

    # Export for pytest fixtures
    export VLLM_TEST_BASE_URL="http://127.0.0.1:${VLLM_PORT}"
    export VLLM_TEST_MODEL="$VLLM_MODEL"
    export VLLM_VENV_PATH="$VLLM_VENV"
    log "vLLM ready. VLLM_TEST_BASE_URL=$VLLM_TEST_BASE_URL"
}

# WITH_EXAMPLES=1 runs pytest on the whole repo (includes docs/examples/)
if [[ "${WITH_EXAMPLES:-0}" == "1" ]]; then
    PYTEST_DIR="."
else
    PYTEST_DIR="test/"
    log "Examples disabled (WITH_EXAMPLES=0). Pass WITH_EXAMPLES=1 to include docs/examples/."
fi

# WITH_TOOLING_TESTS=1 includes test/tooling/ (ignored by default)
PYTEST_ARGS=()
if [[ "${WITH_TOOLING_TESTS:-0}" != "1" ]]; then
    PYTEST_ARGS+=("--ignore=tooling")
    log "Tooling tests disabled (WITH_TOOLING_TESTS=0). Pass WITH_TOOLING_TESTS=1 to include test/tooling/."
fi

if [[ "$#" -eq 0 && "$SERIAL_PHASES" != "1" ]]; then
    PYTEST_ARGS+=("--group-by-backend")
elif [[ "$#" -gt 0 ]]; then
    PYTEST_ARGS+=("$@")
fi

# A caller-supplied -m would override the per-phase marker selections (pytest
# keeps the last -m on the command line) and run the same selection in every
# phase, so marker selection is only meaningful in single-run mode. Fall back
# to single-run mode with a loud warning instead.
USER_MARKER=0
for arg in "$@"; do
    if [[ "$arg" == "-m" || "$arg" == -m?* ]]; then
        USER_MARKER=1
        break
    fi
done
if [[ "$SERIAL_PHASES" == "1" && "$USER_MARKER" == "1" ]]; then
    log "WARNING: caller passed -m, which conflicts with per-phase marker selection — falling back to single-run mode (SERIAL_PHASES=0). Use PHASES= to scope phases explicitly."
    SERIAL_PHASES=0
fi

# The script owns the diagnostic report contract: every run records full
# per-test durations and a JSON report under $LOGDIR (pytest_full.log plus
# pytest_report.json in single-run mode; phase_<name>.log plus
# pytest_report_<name>.json per phase in serial mode) unless the caller
# supplies its own --durations/--json-report arguments. Callers should pass
# only test-selection and verbosity arguments.
CALLER_DURATIONS=0
CALLER_JSON_REPORT=0
CALLER_JSON_REPORT_FILE=""
for arg in "$@"; do
    case "$arg" in
        --durations=*) CALLER_DURATIONS=1 ;;
        --json-report-file=*)
            CALLER_JSON_REPORT=1
            CALLER_JSON_REPORT_FILE="${arg#--json-report-file=}"
            ;;
        --json-report*) CALLER_JSON_REPORT=1 ;;
    esac
done
DURATIONS_ARG=()
if [[ $CALLER_DURATIONS -eq 0 ]]; then
    DURATIONS_ARG=(--durations=0)
fi

# Phased mode can't hand a caller-supplied --json-report-file straight to
# pytest per phase — each phase's run would overwrite it, leaving only the
# last phase's results. Strip it out of the per-phase pytest args (so
# run_phase falls back to its own pytest_report_<name>.json naming) and
# merge the per-phase reports into the caller's path once all phases finish.
if [[ "$SERIAL_PHASES" == "1" && $CALLER_JSON_REPORT -eq 1 ]]; then
    _FILTERED_PYTEST_ARGS=()
    for arg in "${PYTEST_ARGS[@]}"; do
        case "$arg" in
            --json-report|--json-report-file=*) ;;
            *) _FILTERED_PYTEST_ARGS+=("$arg") ;;
        esac
    done
    PYTEST_ARGS=("${_FILTERED_PYTEST_ARGS[@]}")
    CALLER_JSON_REPORT=0
    log "Phased mode: deferring caller's --json-report-file — per-phase reports will be merged into ${CALLER_JSON_REPORT_FILE} after all phases finish."
fi

# --- Run tests ---
log "Log directory: $LOGDIR"
${UV_PYTHON:+log "Python version: $UV_PYTHON"}

# Use UV_PYTHON env var if set, otherwise use default Python
UV_PYTHON_ARG=""
if [[ -n "${UV_PYTHON:-}" ]]; then
    UV_PYTHON_ARG="--python $UV_PYTHON"
fi

# Download NLTK data required by granite formatter tests
log "Downloading NLTK punkt_tab data..."
uv run --quiet --frozen --all-groups --all-extras $UV_PYTHON_ARG \
    python -c "import nltk; nltk.download('punkt_tab', quiet=True)" || true

# phase_enabled NAME — true if PHASES includes NAME (or PHASES=all).
# (defined in the configuration section above; used by the install gate too)
if [[ "$SERIAL_PHASES" == "1" ]]; then
    # Phased execution: one pytest process per backend group, one CUDA
    # context at a time. Intended for exclusive-GPU jobs (bare -gpu "num=1").
    # The phase marker expressions mirror the --group-by-backend groups
    # (huggingface -> ollama -> openai_vllm -> api/other); "not slow" is
    # re-stated because a command-line -m replaces addopts' "-m not slow".
    log "SERIAL_PHASES=1 — phased execution (phases: ${PHASES}; per-phase base args: ${PYTEST_ARGS[*]})"
    # --group-by-backend triggers the conftest group warm-up/eviction hooks;
    # each phase contains a single group, so it only reorders within the group.
    # Each script run owns one coverage dataset (--cov-append across phases),
    # so start from a clean file (honours per-job COVERAGE_FILE).
    rm -f "${COVERAGE_FILE:-.coverage}"
    PHASE_ARGS=("${PYTEST_ARGS[@]}" --cov-append --group-by-backend)

    if phase_enabled hf; then
        log "Starting phase 1 (huggingface, in-process — no servers)..."
        run_phase p1_huggingface "${PHASE_ARGS[@]}" -m "huggingface and not slow"
        wait_gpu_free "hf phase"
    else
        log "Phase p1_huggingface skipped (PHASES=${PHASES})"
    fi

    if phase_enabled ollama; then
        start_ollama
        log "Starting phase 2 (ollama)..."
        run_phase p2_ollama "${PHASE_ARGS[@]}" -m "ollama and not slow"
        stop_ollama
        wait_gpu_free "ollama phase"
    else
        log "Phase p2_ollama skipped (PHASES=${PHASES})"
    fi

    if phase_enabled vllm; then
        if [[ "$WITH_VLLM" == "1" ]]; then
            if ! start_vllm; then
                log "Phase p3_openai_vllm skipped (vLLM failed to start)"
                OVERALL_RC=1
            else
                log "Starting phase 3 (openai/vllm; openai+ollama tests already ran in phase 2)..."
                run_phase p3_openai_vllm "${PHASE_ARGS[@]}" -m "(openai or vllm) and not ollama and not slow"
                stop_vllm
                wait_gpu_free "vllm phase"
            fi
        else
            log "Phase 3 skipped (WITH_VLLM=0)"
        fi
    else
        log "Phase p3_openai_vllm skipped (PHASES=${PHASES})"
    fi

    if phase_enabled base; then
        log "Starting phase 4 (base/unit/api — no servers)..."
        run_phase p4_base "${PHASE_ARGS[@]}" -m "not huggingface and not ollama and not openai and not vllm and not slow"
    else
        log "Phase p4_base skipped (PHASES=${PHASES})"
    fi

    if [[ $RAN_ANY -eq 0 ]]; then
        die "No phases executed (PHASES=${PHASES}) — nothing was tested"
    fi

    if [[ -n "$CALLER_JSON_REPORT_FILE" ]]; then
        log "Merging per-phase JSON reports into ${CALLER_JSON_REPORT_FILE} ..."
        uv run --quiet --frozen --all-groups --all-extras $UV_PYTHON_ARG python - "$LOGDIR" "$CALLER_JSON_REPORT_FILE" <<'PYEOF'
import glob
import json
import os
import sys

logdir, out_path = sys.argv[1], sys.argv[2]
reports = sorted(glob.glob(os.path.join(logdir, "pytest_report_p*.json")))
merged = None
for path in reports:
    with open(path) as f:
        data = json.load(f)
    if merged is None:
        merged = data
        merged["tests"] = list(data.get("tests", []))
        continue
    merged["tests"].extend(data.get("tests", []))
    summary = merged.setdefault("summary", {})
    for key, val in data.get("summary", {}).items():
        summary[key] = summary.get(key, 0) + val
    merged["duration"] = merged.get("duration", 0) + data.get("duration", 0)

if merged is not None:
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(merged, f)
PYEOF
    fi

    EXIT_CODE=$OVERALL_RC
else
    log "Starting pytest (single-run mode; both servers up for the whole run)..."
    log "Pytest args: ${PYTEST_ARGS[*]}"

    start_ollama
    if [[ "$WITH_VLLM" == "1" ]]; then
        start_vllm
    else
        log "vLLM disabled (WITH_VLLM=0). Pass WITH_VLLM=1 to enable, or run on a CUDA host for auto-detection."
    fi

    REPORT_ARGS=()
    if [[ $CALLER_JSON_REPORT -eq 0 ]]; then
        REPORT_ARGS=(--json-report --json-report-file="$LOGDIR/pytest_report.json")
    fi
    set +e
    uv run --quiet --frozen --all-groups --all-extras $UV_PYTHON_ARG \
        pytest "$PYTEST_DIR" "${PYTEST_ARGS[@]}" "${DURATIONS_ARG[@]}" "${REPORT_ARGS[@]}" \
        2>&1 | tee "$LOGDIR/pytest_full.log"
    EXIT_CODE=${PIPESTATUS[0]}
    set -e
fi

log "Tests finished with exit code: $EXIT_CODE"
log "Logs: $LOGDIR/"
exit $EXIT_CODE
