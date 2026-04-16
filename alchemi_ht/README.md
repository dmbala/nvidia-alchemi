# ALCHEMI HT -- High-Throughput Molecular Screening Pipeline

A GPU-accelerated pipeline for screening organic electrolyte candidates from large chemical libraries. It processes millions of molecules through sequential computational stages -- 3D embedding, neural-network geometry relaxation, and DFT electronic-property calculations -- and ranks them by electrochemical stability.

The target application is discovering thermodynamically and electrochemically stable organic molecules for battery electrolyte design.

## Data Source

The input molecule library is **GDB-17**, a database of 166 billion organic molecules with up to 17 heavy atoms. The compressed SMILES files used in this pipeline were downloaded from Zenodo:

> **GDB-17 Database**: <https://zenodo.org/records/5172018>

The following files are used:

| File | Description |
|------|-------------|
| `GDB17.50000000.smi.gz` | Full GDB-17 subset (314 MB compressed) |
| `GDB17.50000000LL.smi.gz` | Low-level subset (75 MB compressed) |
| `GDB17.50000000LLnoSR.smi.gz` | Low-level subset without stereochemistry (55 MB compressed) |

These are converted to `gdb17_subset.csv` via `convert_smi_to_csv.py` before chunking.

## Pipeline Overview

```
SMILES string
    |
    v
[1] 3D Embedding        (RDKit ETKDGv3 + MMFF optimization)
    |
    v
[2] Geometry Relaxation  (AIMNet2 neural network potential, LBFGS optimizer)
    |
    v
[3] Electronic Properties (GPU4PySCF DFT: B3LYP / def2-svp)
    |        Computes HOMO, LUMO, and HOMO-LUMO gap
    v
[4] Aggregation & Ranking (filter by energy + gap, output top candidates)
```

## Directory Structure

```
alchemi_ht/
├── pipeline.py               # Core per-molecule pipeline (steps 1-3)
├── chunk_data.py              # Splits master CSV into chunks for parallel processing
├── aggregate_and_rank.py      # Merges results and ranks candidates (step 4)
├── convert_smi_to_csv.py      # Converts GDB-17 .smi.gz files to pipeline CSV format
├── run_screening.slurm        # SLURM array job script
├── alchemi_ht.def             # Apptainer/Singularity container definition
├── alchemi_ht.sif             # Built container image (~14 GB)
├── gdb17_subset.csv           # Master molecule dataset (8.3M molecules)
├── GDB17.*.smi.gz             # Source GDB-17 SMILES files (compressed)
├── data/
│   ├── chunks/                # Input chunks (chunk_0000.csv ... chunk_1669.csv)
│   └── results/               # Output CSVs from pipeline (result_XXXX.csv)
└── logs/                      # SLURM job stdout/stderr
```

## Prerequisites

- **Cluster access**: FASRC cluster with `kempner_dev` partition and account
- **GPU**: NVIDIA H100 or H200
- **Container runtime**: Apptainer (Singularity) CE 4.4+
- **Container image**: `alchemi_ht.sif` (built from `alchemi_ht.def`)

### Building the container

```bash
apptainer build alchemi_ht.sif alchemi_ht.def
```

The `.def` file bakes the AIMNet2 model (`aimnet2_wb97m_d3_0.pt`) into the container at build time. If you're running against a pre-existing `alchemi_ht.sif` that was built before that change, seed the model cache once:

```bash
bash ../container/fetch_aimnet_assets.sh
```

This populates `container/aimnet_assets/`, which `run_screening.slurm` bind-mounts over the container's read-only `aimnet/calculators/assets/` directory. Compute nodes without outbound HTTPS require this step; nodes with egress can also let AIMNet download on first job run.

The container is built on top of the base ALCHEMI image and adds:

| Package | Purpose |
|---------|---------|
| `aimnet` | AIMNet2 neural network potential for geometry relaxation |
| `ase` | Atomic Simulation Environment (LBFGS optimizer) |
| `rdkit` | SMILES parsing and 3D coordinate embedding |
| `gpu4pyscf-cuda12x` | GPU-accelerated DFT (HOMO-LUMO calculation) |
| `pyscf` | Quantum chemistry framework |
| `pandas` | Data handling |

## Usage

### 1. Prepare the dataset

Convert compressed GDB-17 SMILES files into pipeline-compatible CSV, shuffled for unbiased streaming into downstream active learning:

```bash
python convert_smi_to_csv.py GDB17.50000000.smi.gz gdb17_subset.csv --shuffle-seed 42
```

`--limit N` optionally truncates. `--shuffle-seed -1` disables the shuffle (not recommended — GDB-17 is enumeration-ordered). A `# shuffle_seed=N` comment line is prepended; downstream consumers use `pandas.read_csv(..., comment='#')`.

### 2. Chunk the data

```bash
python chunk_data.py --input-csv gdb17_subset.csv --output-dir data/chunks --chunk-size 5000
```

Creates `chunk_0000.csv` through `chunk_NNNN.csv` in the target directory. Defaults match historical behavior. For orchestrator use, point `--output-dir` at `../runs/screen/chunks`.

### 3. Submit SLURM array jobs

```bash
NUM_CHUNKS=$(ls data/chunks/chunk_*.csv | wc -l)
sbatch --array=0-$((NUM_CHUNKS-1)) run_screening.slurm

# Or point at a non-default layout (used by the /ht-loop orchestrator):
sbatch --array=0-$((NUM_CHUNKS-1)) \
    --export=ALL,CHUNK_DIR=/abs/path/chunks,RESULT_DIR=/abs/path/results \
    run_screening.slurm
```

Each array task processes one chunk on a single GPU. The SLURM script allocates:
- 1 GPU
- 4 CPU cores
- 32 GB memory
- 1 hour wall time

### 4. Monitor progress

```bash
squeue -u $USER --array
tail -f logs/screen_*.err
```

### 5. Aggregate and rank

Once all jobs complete:

```bash
python aggregate_and_rank.py \
    --results-dir data/results \
    --output top_1_percent_candidates.csv \
    --stats-json data/results/stats.json
```

Produces the ranked candidate CSV, plus an optional JSON stats summary (status counts, success rate, thresholds) consumed by the `/ht-loop` orchestrator for decision hooks.

Filtering criteria:
- **Status** = "Success"
- **Energy** <= 50th percentile (thermodynamic stability)
- **HOMO-LUMO gap** >= 90th percentile (electrochemical stability)

Candidates are ranked by gap in descending order.

### Orchestrated mode

For large screens (or when the active-learning loop is consuming results concurrently), use the top-level orchestrator instead of manual steps 3–5:

```bash
python3.12 -m orchestrator.ht_loop --action init
python3.12 -m orchestrator.ht_loop --action tick   # repeat or use /ht-loop
```

See the top-level README for details.

## Output Format

Each result CSV contains these columns:

| Column | Description |
|--------|-------------|
| `id` | Molecule identifier |
| `smiles` | Input SMILES string |
| `name` | Molecule name (if available) |
| `status` | "Success", "Infeasible Geometry", or "Error: ..." |
| `energy_eV` | AIMNet2 potential energy (eV) |
| `n_atoms` | Number of atoms (including hydrogens) |
| `homo_eV` | HOMO energy (eV) |
| `lumo_eV` | LUMO energy (eV) |
| `gap_eV` | HOMO-LUMO gap (eV) |
| `opt_coords` | Optimized 3D coordinates (semicolon-delimited) |

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| "Infeasible Geometry" status | RDKit cannot generate a valid 3D structure | Expected for some molecules; they are skipped |
| GPU out of memory | DFT calculation exceeds GPU VRAM | Reduce basis set (`--basis def2-sto-3g`) or increase `--mem` in SLURM script |
| Convergence failure | LBFGS does not reach `fmax` threshold | Increase `steps` or relax `fmax` in `pipeline.py` |
| WARP_CACHE write error | Container cannot write JIT cache | Ensure `WARP_CACHE_PATH` points to a writable directory (default: `/tmp/warp_cache`) |

## Running a single chunk manually

For testing or debugging, run one chunk directly inside the container:

```bash
singularity exec --nv alchemi_ht.sif \
    python pipeline.py --chunk data/chunks/chunk_0000.csv --output test_result.csv
```

Use `--basis` to change the DFT basis set (default: `def2-svp`).
