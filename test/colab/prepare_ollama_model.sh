#!/usr/bin/env bash
# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Install ollama, serve it, and stage the session-default model so notebook execution never
# spends its per-cell timeout on a multi-GB download.
#
# Same pins and rationale as .github/workflows/notebooks.yml: OLLAMA_VERSION is pinned per
# #1388, and the published granite4.2:3b tag ships num_ctx=131072, a ~6 GB KV cache at load,
# which is too much for a 16 GB runner. Both jobs in colab-notebooks.yml call this, and it is
# safe to run locally: an existing ollama install, a running daemon, and an already-clamped
# model are all left alone.
set -euo pipefail

OLLAMA_VERSION="${OLLAMA_VERSION:-0.33.1}"
OLLAMA_MODEL="${OLLAMA_MODEL:-granite4.2:3b}"
OLLAMA_NUM_CTX="${OLLAMA_NUM_CTX:-8192}"
OLLAMA_LOG="${OLLAMA_LOG:-/tmp/ollama-serve.log}"

if command -v ollama >/dev/null 2>&1; then
  echo "ollama already installed: $(ollama --version 2>&1 | head -1)"
else
  echo "Installing ollama ${OLLAMA_VERSION}"
  curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION="${OLLAMA_VERSION}" sh
fi

if ollama list >/dev/null 2>&1; then
  echo "ollama already serving on ${OLLAMA_HOST:-default host}"
else
  echo "Starting ollama serve, logging to ${OLLAMA_LOG}"
  nohup ollama serve >"${OLLAMA_LOG}" 2>&1 &
  for _ in $(seq 1 30); do
    ollama list >/dev/null 2>&1 && break
    sleep 2
  done
  ollama list >/dev/null 2>&1 || {
    echo "::error::ollama serve did not become ready"
    tail -n 50 "${OLLAMA_LOG}" || true
    exit 1
  }
fi

if ollama show "${OLLAMA_MODEL}" --modelfile 2>/dev/null |
  grep -qx "PARAMETER num_ctx ${OLLAMA_NUM_CTX}"; then
  echo "${OLLAMA_MODEL} already present and clamped to num_ctx ${OLLAMA_NUM_CTX}"
  exit 0
fi

pulled=false
for i in 1 2 3 4 5; do
  ollama pull "${OLLAMA_MODEL}" && {
    pulled=true
    break
  }
  echo "Attempt ${i} to pull ${OLLAMA_MODEL} failed, retrying in 20s..."
  sleep 20
done
# Without this the loop exits 0 after exhausting every attempt, turning an infra miss into a
# test failure 20 minutes later.
${pulled} || {
  echo "::error::failed to pull ${OLLAMA_MODEL} after 5 attempts"
  exit 1
}

echo "Clamping ${OLLAMA_MODEL} to num_ctx ${OLLAMA_NUM_CTX}"
ollama cp "${OLLAMA_MODEL}" "${OLLAMA_MODEL}-128k"
printf 'FROM %s-128k\nPARAMETER num_ctx %s\n' "${OLLAMA_MODEL}" "${OLLAMA_NUM_CTX}" \
  >/tmp/MODELFILE-colab-ci
ollama create "${OLLAMA_MODEL}" -f /tmp/MODELFILE-colab-ci
ollama show "${OLLAMA_MODEL}" --modelfile | grep -qx "PARAMETER num_ctx ${OLLAMA_NUM_CTX}"
ollama list
