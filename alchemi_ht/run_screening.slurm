#!/bin/bash
#SBATCH --job-name=ht-screen
#SBATCH --partition=kempner_dev
#SBATCH --account=kempner_dev
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=logs/screen_%A_%a.out
#SBATCH --error=logs/screen_%A_%a.err

# Usage:
#   sbatch --array=0-<N-1> run_screening.slurm
# Optional env overrides (exported via `sbatch --export=ALL,CHUNK_DIR=...,RESULT_DIR=...`):
#   CHUNK_DIR     path to directory holding chunk_NNNN.csv   (default: data/chunks)
#   RESULT_DIR    path to directory for result_NNNN.csv      (default: data/results)
#   SIF_PATH      path to the alchemi_ht.sif container       (default: ./alchemi_ht.sif)
#   CHUNK_OFFSET  integer added to SLURM_ARRAY_TASK_ID to get the real chunk ID
#                 (default: 0). The orchestrator uses this to work around the
#                 cluster's MaxArraySize cap by splitting submissions into windows.

# When run via `sbatch`, the script is copied to /var/slurmd/spool/jobN/ — so
# BASH_SOURCE resolution is unreliable. The orchestrator passes HT_SCRIPT_DIR
# as an absolute path via --export. Manual callers can rely on BASH_SOURCE.
if [ -n "${HT_SCRIPT_DIR:-}" ]; then
    SCRIPT_DIR="${HT_SCRIPT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
CHUNK_DIR="${CHUNK_DIR:-${SCRIPT_DIR}/data/chunks}"
RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/data/results}"
SIF_PATH="${SIF_PATH:-${SCRIPT_DIR}/alchemi_ht.sif}"
CHUNK_OFFSET="${CHUNK_OFFSET:-0}"

# Writable bind target for AIMNet2's model cache. The container's
# /usr/local/lib/.../aimnet/calculators/assets directory is read-only, so we
# mount a host-side writable path over it. AIMNet downloads on first use (or
# reuses a seeded .pt file) and subsequent workers reuse the cache.
AIMNET_CACHE="${AIMNET_CACHE:-${SCRIPT_DIR}/../container/aimnet_assets}"
mkdir -p "${AIMNET_CACHE}"
AIMNET_ASSETS_IN_CONTAINER="/usr/local/lib/python3.12/dist-packages/aimnet/calculators/assets"

REAL_CHUNK_ID=$((SLURM_ARRAY_TASK_ID + CHUNK_OFFSET))
CHUNK_ID=$(printf "%04d" ${REAL_CHUNK_ID})
CHUNK_FILE="${CHUNK_DIR}/chunk_${CHUNK_ID}.csv"
OUTPUT_FILE="${RESULT_DIR}/result_${CHUNK_ID}.csv"

mkdir -p "${RESULT_DIR}"

export SINGULARITYENV_PYTHONNOUSERSITE=1
export SINGULARITY_CACHEDIR=/n/netscratch/kempner_dev/Lab/bdesinghu/.cache/

if [ ! -f "${CHUNK_FILE}" ]; then
    echo "ERROR: Chunk file not found: ${CHUNK_FILE}"
    exit 1
fi

echo "Processing chunk ${CHUNK_ID}: ${CHUNK_FILE} -> ${OUTPUT_FILE}"
echo "Node: $(hostname), GPU: ${CUDA_VISIBLE_DEVICES}"

singularity exec --nv \
    --bind /n/netscratch/kempner_dev:/data \
    --bind "${SCRIPT_DIR}:${SCRIPT_DIR}" \
    --bind "${CHUNK_DIR}:${CHUNK_DIR}" \
    --bind "${RESULT_DIR}:${RESULT_DIR}" \
    --bind "${AIMNET_CACHE}:${AIMNET_ASSETS_IN_CONTAINER}" \
    "${SIF_PATH}" \
    python "${SCRIPT_DIR}/pipeline.py" \
        --chunk "${CHUNK_FILE}" \
        --output "${OUTPUT_FILE}"

echo "Chunk ${CHUNK_ID} complete, exit code: $?"
