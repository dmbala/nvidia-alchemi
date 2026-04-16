#!/bin/bash
# Build the alchemi_clip Singularity image from alchemi_ht.sif
# Run interactively on a node with internet access:
#   srun --partition=kempner_dev --account=kempner_dev --time=01:00:00 --mem=32G bash build.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SINGULARITY_CACHEDIR=/n/netscratch/kempner_dev/Lab/bdesinghu/.cache/

singularity build --fakeroot \
    "${SCRIPT_DIR}/alchemi_clip.sif" \
    "${SCRIPT_DIR}/alchemi_clip.def"

echo "Build complete: ${SCRIPT_DIR}/alchemi_clip.sif"
