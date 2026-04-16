# Examples

Minimal demos that exercise the base `alchemi.sif` container.

| File | Description |
|------|-------------|
| `test_alchemi.py` | Smoke test: imports `nvalchemi` components and reports PyTorch / CUDA / GPU status. |
| `example_dynamics.py` | Geometry optimization of two small molecules using `DemoDynamics` with convergence detection. |
| `run.sh` | Launcher for `example_dynamics.py` (runs as a SLURM job or via `bash run.sh` on a GPU node). |

## Prerequisites

Build the base container first (see `../container/`):

```bash
cd ../container && sbatch build.sh
```

## Run

```bash
# On a GPU node (interactive or via sbatch)
bash run.sh
# or
sbatch run.sh
```

To run the smoke test instead:

```bash
singularity exec --nv ../container/alchemi.sif python test_alchemi.py
```
