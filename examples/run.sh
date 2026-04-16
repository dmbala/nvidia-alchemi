#!/bin/bash
#SBATCH --job-name=alchemi-test
#SBATCH --partition=kempner_dev
#SBATCH --account=kempner_dev
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=run_%j.log

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prevent user site-packages from overriding container packages
# (e.g. numpy 2.x in ~/.local breaks NGC torch which needs numpy <2)
export SINGULARITYENV_PYTHONNOUSERSITE=1

# Use netscratch for singularity cache
export SINGULARITY_CACHEDIR=/n/netscratch/kempner_dev/Lab/bdesinghu/.cache/

# Run the ALCHEMI example inside the container
# --nv passes host NVIDIA drivers into the container
# --bind mounts the scratch filesystem for data access
singularity run --nv \
    --bind /n/netscratch/kempner_dev:/data \
    "${SCRIPT_DIR}/../container/alchemi.sif" "${SCRIPT_DIR}/example_dynamics.py"
