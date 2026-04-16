#!/bin/bash
# Seed the AIMNet2 asset cache used by alchemi_ht's SLURM jobs.
#
# The container's /usr/local/lib/python3.12/dist-packages/aimnet/calculators/assets
# directory is read-only. run_screening.slurm bind-mounts this host directory
# over it so AIMNet finds a writable path.
#
# Pre-seeding avoids relying on each compute node having outbound HTTPS — FASRC
# compute partitions often restrict egress.
#
# Safe to re-run (idempotent: checks for existing file).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_DIR="${SCRIPT_DIR}/aimnet_assets"
mkdir -p "${CACHE_DIR}"

# Default model used by alchemi_ht/pipeline.py (via AIMNet2ASE("aimnet2") alias).
MODELS=(
    "aimnet2_wb97m_d3_0.pt|https://storage.googleapis.com/aimnetcentral/aimnet2v2/AIMNet2/aimnet2_wb97m_d3_0.pt"
)

for entry in "${MODELS[@]}"; do
    file="${entry%%|*}"
    url="${entry##*|}"
    target="${CACHE_DIR}/${file}"
    if [ -f "${target}" ] && [ -s "${target}" ]; then
        echo "skip: ${file} already cached ($(du -h "${target}" | cut -f1))"
        continue
    fi
    echo "fetch: ${url} -> ${target}"
    curl -sSL --fail -o "${target}.tmp" "${url}"
    mv "${target}.tmp" "${target}"
    echo "  size: $(du -h "${target}" | cut -f1)"
done

echo "Done. Cache at ${CACHE_DIR}"
ls -lh "${CACHE_DIR}"
