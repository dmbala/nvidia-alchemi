#!/bin/bash
# Full end-to-end Phase 6 pipeline test using alchemi_clip.sif
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIF="${SCRIPT_DIR}/alchemi_clip.sif"

export SINGULARITYENV_PYTHONNOUSERSITE=1
export SINGULARITY_CACHEDIR=/n/netscratch/kempner_dev/Lab/bdesinghu/.cache/

RUN="singularity exec --nv
  --bind /etc/ssl/certs/ca-bundle.crt:/etc/ssl/certs/ca-bundle.crt:ro
  --bind /n/netscratch/kempner_dev:/data
  ${SIF}"

echo "============================================"
echo " Phase 6 – Full Pipeline Test"
echo " Container: ${SIF}"
echo " Node: $(hostname), GPU: ${CUDA_VISIBLE_DEVICES:-none}"
echo "============================================"

echo ""
echo ">>> Step 0: Generate synthetic test data"
${RUN} python "${SCRIPT_DIR}/generate_test_data.py"

echo ""
echo ">>> Step 1a: Train Contrastive Encoder (MolecularCLIP)"
${RUN} python "${SCRIPT_DIR}/train_contrastive.py" \
    --results-dir "${SCRIPT_DIR}/data/results" \
    --epochs 5 \
    --batch-size 8 \
    --latent-dim 128 \
    --lr 1e-3 \
    --output-dir "${SCRIPT_DIR}/models"

echo ""
echo ">>> Step 1b: Train Surrogate Brain (foundation)"
${RUN} python "${SCRIPT_DIR}/train_surrogate.py" \
    --encoder foundation \
    --results-dir "${SCRIPT_DIR}/data/results" \
    --target gap_eV \
    --n-estimators 50 \
    --output "${SCRIPT_DIR}/models/surrogate_foundation.pkl"

echo ""
echo ">>> Step 1b: Train Surrogate Brain (contrastive)"
${RUN} python "${SCRIPT_DIR}/train_surrogate.py" \
    --encoder contrastive \
    --results-dir "${SCRIPT_DIR}/data/results" \
    --target gap_eV \
    --contrastive-encoder-path "${SCRIPT_DIR}/models/contrastive_1d_encoder.pt" \
    --latent-dim 128 \
    --n-estimators 50 \
    --output "${SCRIPT_DIR}/models/surrogate_contrastive.pkl"

echo ""
echo ">>> Step 2: Active Search (foundation)"
${RUN} python "${SCRIPT_DIR}/active_search_latent.py" \
    --encoder foundation \
    --unexplored-csv "${SCRIPT_DIR}/data/unexplored_test.csv" \
    --surrogate-path "${SCRIPT_DIR}/models/surrogate_foundation.pkl" \
    --top-k 10 \
    --output "${SCRIPT_DIR}/data/chunks/active_learning_batch_foundation.csv"

echo ""
echo ">>> Step 2: Active Search (contrastive)"
${RUN} python "${SCRIPT_DIR}/active_search_latent.py" \
    --encoder contrastive \
    --unexplored-csv "${SCRIPT_DIR}/data/unexplored_test.csv" \
    --surrogate-path "${SCRIPT_DIR}/models/surrogate_contrastive.pkl" \
    --contrastive-encoder-path "${SCRIPT_DIR}/models/contrastive_1d_encoder.pt" \
    --latent-dim 128 \
    --top-k 10 \
    --output "${SCRIPT_DIR}/data/chunks/active_learning_batch_contrastive.csv"

echo ""
echo ">>> Step 3: Prepare Verification Chunks"
${RUN} python "${SCRIPT_DIR}/prepare_verification.py" \
    --input "${SCRIPT_DIR}/data/chunks/active_learning_batch_contrastive.csv" \
    --chunk-size 5

echo ""
echo "============================================"
echo " Pipeline Complete — All Steps Passed"
echo "============================================"
echo ""
echo "Models:"
ls -lh "${SCRIPT_DIR}/models/"
echo ""
echo "Data outputs:"
ls -lh "${SCRIPT_DIR}/data/chunks/"
echo ""
echo "Active learning candidates (contrastive):"
cat "${SCRIPT_DIR}/data/chunks/active_learning_batch_contrastive.csv"
