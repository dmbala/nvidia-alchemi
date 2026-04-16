#!/bin/bash
# End-to-end test of Phase 6 pipeline on small synthetic data.
# Uses alchemi_ht.sif with runtime pip install for transformers.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIF="/n/netscratch/kempner_dev/Lab/bdesinghu/Agent/alchemi/alchemi_ht/alchemi_ht.sif"
TMPLIB="/n/netscratch/kempner_dev/Lab/bdesinghu/.cache/pip_runtime"

export SINGULARITYENV_PYTHONNOUSERSITE=0
export SINGULARITYENV_PYTHONPATH="${TMPLIB}/lib/python3.12/site-packages:${TMPLIB}/lib/python3.10/site-packages"
export SINGULARITY_CACHEDIR=/n/netscratch/kempner_dev/Lab/bdesinghu/.cache/

RUN="singularity exec --nv --bind /n/netscratch/kempner_dev:/data --bind ${SCRIPT_DIR}:${SCRIPT_DIR} ${SIF}"

echo "=== Step 0: Install transformers into temp location ==="
${RUN} pip install --target "${TMPLIB}/lib/python3.12/site-packages" transformers tokenizers 2>&1 | tail -5

echo ""
echo "=== Step 1a: Train Contrastive Encoder (5 epochs, small data) ==="
${RUN} python "${SCRIPT_DIR}/train_contrastive.py" \
    --results-dir "${SCRIPT_DIR}/data/results" \
    --epochs 5 \
    --batch-size 8 \
    --latent-dim 128 \
    --lr 1e-3 \
    --output-dir "${SCRIPT_DIR}/models"

echo ""
echo "=== Step 1b: Train Surrogate Brain (foundation) ==="
${RUN} python "${SCRIPT_DIR}/train_surrogate.py" \
    --encoder foundation \
    --results-dir "${SCRIPT_DIR}/data/results" \
    --target gap_eV \
    --n-estimators 50 \
    --output "${SCRIPT_DIR}/models/surrogate_foundation.pkl"

echo ""
echo "=== Step 1b: Train Surrogate Brain (contrastive) ==="
${RUN} python "${SCRIPT_DIR}/train_surrogate.py" \
    --encoder contrastive \
    --results-dir "${SCRIPT_DIR}/data/results" \
    --target gap_eV \
    --contrastive-encoder-path "${SCRIPT_DIR}/models/contrastive_1d_encoder.pt" \
    --n-estimators 50 \
    --output "${SCRIPT_DIR}/models/surrogate_contrastive.pkl"

echo ""
echo "=== Step 2: Active Search (foundation) ==="
${RUN} python "${SCRIPT_DIR}/active_search_latent.py" \
    --encoder foundation \
    --unexplored-csv "${SCRIPT_DIR}/data/unexplored_test.csv" \
    --surrogate-path "${SCRIPT_DIR}/models/surrogate_foundation.pkl" \
    --top-k 10 \
    --output "${SCRIPT_DIR}/data/chunks/active_learning_batch_foundation.csv"

echo ""
echo "=== Step 2: Active Search (contrastive) ==="
${RUN} python "${SCRIPT_DIR}/active_search_latent.py" \
    --encoder contrastive \
    --unexplored-csv "${SCRIPT_DIR}/data/unexplored_test.csv" \
    --surrogate-path "${SCRIPT_DIR}/models/surrogate_contrastive.pkl" \
    --contrastive-encoder-path "${SCRIPT_DIR}/models/contrastive_1d_encoder.pt" \
    --top-k 10 \
    --output "${SCRIPT_DIR}/data/chunks/active_learning_batch_contrastive.csv"

echo ""
echo "=== Step 3: Prepare Verification Chunks ==="
${RUN} python "${SCRIPT_DIR}/prepare_verification.py" \
    --input "${SCRIPT_DIR}/data/chunks/active_learning_batch_contrastive.csv" \
    --chunk-size 5

echo ""
echo "=== All tests passed! ==="
echo "Outputs:"
ls -lh "${SCRIPT_DIR}/models/"
ls -lh "${SCRIPT_DIR}/data/chunks/"
