# Phase 6: Latent-Space Active Learning for ALCHEMI

Evaluates millions of untested molecules by projecting them into a dense latent space, predicting their properties with a surrogate model, and selecting the highest-value targets for GPU physics verification via the Phase 3 ALCHEMI + GPU4PySCF pipeline.

## Architecture

```
                     +-----------------------+
                     |  HT Screening Results |
                     |  (SMILES + 3D coords  |
                     |   + energy + gap)     |
                     +-----------+-----------+
                                 |
                    +------------+------------+
                    |                         |
          Step 1a: Train              Step 1b: Train
          MolecularCLIP               Surrogate Brain
          (InfoNCE loss)             (RandomForest on
                    |                 latent vectors)
                    |                         |
                    v                         v
        contrastive_1d_encoder.pt    surrogate_{encoder}.pkl
                    |                         |
                    +------------+------------+
                                 |
                           Step 2: Active Search
                           (embed millions of SMILES,
                            predict with Brain,
                            rank by UCB score)
                                 |
                                 v
                     active_learning_batch.csv
                          (top-k candidates)
                                 |
                           Step 3: Verify
                           (submit to Phase 3
                            ALCHEMI + GPU4PySCF)
                                 |
                                 v
                     Append to master dataset,
                     retrain, repeat
```

## Three Encoder Modes

| Mode | Flag | Latent Space | Speed | Best For |
|------|------|-------------|-------|----------|
| **Foundation** | `--encoder foundation` | ChemBERTa 384D | Fast | Broad exploration of large chemical spaces |
| **GNN** | `--encoder gnn` | SchNet 512D (3D coords via RDKit) | Slow | Physics-aware local exploitation |
| **Contrastive** | `--encoder contrastive` | MolecularCLIP 512D | Fast | Best of both: 3D-aware at 1D-encoder speed |

The contrastive encoder is the key contribution of Phase 6. It is a SMILES text encoder that has been contrastively aligned with a 3D GNN (SchNet) using InfoNCE loss, so it produces 3D-aware embeddings without requiring 3D coordinate generation at inference time.

## Surrogate Targets

Single-target (`--target gap_eV`) or multi-target (`--target all`):

| Target | Property | Unit | Selection Direction |
|--------|----------|------|-------------------|
| `gap_eV` | HOMO-LUMO gap | eV | Higher = better (electrochemical stability) |
| `energy_eV` | Total relaxation energy | eV | Lower = better (thermodynamic stability) |
| `homo_eV` | HOMO orbital energy | eV | Higher = better (oxidation resistance) |
| `lumo_eV` | LUMO orbital energy | eV | Higher = better (reduction resistance) |
| `energy_per_atom` | Size-normalized energy | eV/atom | Lower = better (cross-size comparison) |

Multi-target mode trains one RandomForest per property and ranks candidates by a composite UCB score (z-normalized across all targets so each property contributes equally regardless of scale).

## Evaluation Metrics

Each surrogate reports via 5-fold cross-validation:

| Metric | What It Measures | When It Matters |
|--------|-----------------|-----------------|
| **R-squared** | Variance explained (0 = mean baseline, 1 = perfect) | Overall predictive power |
| **MAE** | Average prediction error in eV | Absolute accuracy of predictions |
| **RMSE** | Root mean squared error, penalizes outliers | Cost of badly mispredicted candidates |
| **Spearman rho** | Rank correlation between true and predicted values | Correct ordering for candidate selection |
| **Top-k% recall** | Of the true top k%, how many does the surrogate recover? | Direct measure of selection quality |

For active learning, **Spearman and top-k recall matter most** -- correct ranking drives candidate selection quality, not exact value prediction. A surrogate with MAE = 1.0 eV but Spearman = 0.95 is more useful than one with MAE = 0.3 eV but Spearman = 0.6.

## Container

Built on top of `alchemi_ht.sif` (which includes PyTorch, CUDA, ALCHEMI, AIMNet2, RDKit, pandas, ASE, scikit-learn, joblib).

`alchemi_clip.def` adds:
- `transformers` + `tokenizers` -- HuggingFace Transformers for ChemBERTa foundation model
- `xgboost` -- optional surrogate upgrade over RandomForest

The GNN encoder is implemented in pure PyTorch (SchNet-style continuous-filter convolution), so PyTorch Geometric is not required.

### Build

```bash
# On a node with internet access:
bash build.sh
```

## Workflow

### Step 1a: Train the Contrastive Encoder

Requires HT screening results (from Phase 3) with SMILES + optimized 3D coordinates.

```bash
sbatch run_train_contrastive.slrm
```

Or directly:

```bash
singularity exec --nv alchemi_clip.sif python train_contrastive.py \
    --results-dir ../alchemi_ht/data/results \
    --epochs 100 \
    --batch-size 128 \
    --latent-dim 512 \
    --lr 3e-4 \
    --output-dir models
```

Outputs:
- `models/contrastive_1d_encoder.pt` -- the 3D-aware text encoder (used for fast screening)
- `models/contrastive_3d_encoder.pt` -- the 3D GNN encoder (for reference)
- `models/molecular_clip.pt` -- full MolecularCLIP model state

### Step 1b: Train the Surrogate Brain

```bash
# Single target (HOMO-LUMO gap):
sbatch run_train_surrogate.slrm

# Multi-target (all properties):
sbatch --export=ENCODER=contrastive run_train_surrogate.slrm
```

Or directly:

```bash
# Single target
singularity exec --nv alchemi_clip.sif python train_surrogate.py \
    --encoder contrastive \
    --results-dir ../alchemi_ht/data/results \
    --target gap_eV \
    --output models/surrogate_contrastive.pkl

# Multi-target
singularity exec --nv alchemi_clip.sif python train_surrogate.py \
    --encoder contrastive \
    --results-dir ../alchemi_ht/data/results \
    --target all \
    --output models/surrogate_contrastive_multi.pkl
```

### Step 2: Active Search

```bash
sbatch run_active_search.slrm
```

Or directly:

```bash
singularity exec --nv alchemi_clip.sif python active_search_latent.py \
    --encoder contrastive \
    --unexplored-csv ../alchemi_ht/gdb17_subset.csv \
    --surrogate-path models/surrogate_contrastive.pkl \
    --top-k 5000 \
    --output data/chunks/active_learning_batch.csv
```

### Step 3: Verify Candidates

Submit the top candidates back to the Phase 3 HT pipeline for ground-truth DFT evaluation:

```bash
python prepare_verification.py \
    --input data/chunks/active_learning_batch.csv \
    --chunk-size 500

sbatch --array=0-9 run_verification.slrm
```

Append the verification results to the master dataset and retrain (Steps 1b and 2) for the next iteration.

## File Reference

```
alchemi-clip/
|-- alchemi_clip.def              Singularity definition (extends alchemi_ht.sif)
|-- alchemi_clip.sif              Built container image (~14 GB)
|-- build.sh                      Container build script
|
|-- embed_utils.py                Shared embedding functions (foundation, contrastive, GNN)
|-- train_contrastive.py          Step 1a: MolecularCLIP training (InfoNCE)
|-- train_surrogate.py            Step 1b: Surrogate Brain (RF on latent vectors)
|-- active_search_latent.py       Step 2:  Latent-space active search (UCB)
|-- prepare_verification.py       Step 3:  Chunk candidates for HT verification
|
|-- run_train_contrastive.slrm    Slurm: train contrastive encoder (1 GPU, 4h)
|-- run_train_surrogate.slrm      Slurm: train surrogate brain (1 GPU, 2h)
|-- run_active_search.slrm        Slurm: run active search (1 GPU, 2h)
|-- run_verification.slrm         Slurm: submit candidates to HT pipeline (array job)
|
|-- generate_test_data.py          Generate synthetic data for testing
|-- run_full_test.sh               End-to-end pipeline test script
|
|-- models/                        Trained model weights and surrogates
|   |-- contrastive_1d_encoder.pt
|   |-- contrastive_3d_encoder.pt
|   |-- molecular_clip.pt
|   +-- surrogate_*.pkl
|
|-- data/
|   |-- results/                   HT screening results (input)
|   +-- chunks/                    Active learning batches (output)
|       |-- active_learning_batch.csv
|       +-- verify_*.csv
|
+-- logs/                          Slurm job logs
```

### Module Responsibilities

| Module | Purpose |
|--------|---------|
| `embed_utils.py` | Single implementation of each encoder (foundation, contrastive, GNN). Shared by both `train_surrogate.py` and `active_search_latent.py` to avoid code duplication. |
| `train_contrastive.py` | Defines `MolecularCLIP`, `SmilesEncoder1D` (ChemBERTa + projection), `SchNetEncoder3D` (pure-PyTorch SchNet), and the contrastive training loop. |
| `train_surrogate.py` | Loads HT results, extracts latent vectors via `embed_utils`, trains RandomForest surrogates, and reports 5-fold CV metrics (R2, MAE, RMSE, Spearman, top-k recall). |
| `active_search_latent.py` | Embeds unexplored SMILES via `embed_utils`, loads the surrogate Brain, computes UCB acquisition scores, and selects top-k candidates. Supports both single-target and multi-target surrogates. |
| `prepare_verification.py` | Reformats active learning candidates to match the HT pipeline input format and splits into chunks for Slurm array submission. |

## Testing

Run the full pipeline end-to-end on synthetic data (49 molecules):

```bash
bash run_full_test.sh
```

This generates synthetic HT results, trains the contrastive encoder (5 epochs), trains surrogates for both foundation and contrastive encoders, runs active search with both, and prepares verification chunks. Takes about 2 minutes on a single GPU.

## Dependencies on Phase 3

This pipeline consumes and produces data compatible with the Phase 3 HT screening pipeline (`../alchemi_ht/`):

- **Input**: `result_*.csv` files from `../alchemi_ht/data/results/` with columns: `smiles`, `status`, `energy_eV`, `gap_eV`, `homo_eV`, `lumo_eV`, `n_atoms`, `opt_coords`
- **Output**: `verify_*.csv` files with columns: `id`, `smiles`, `name` (same format as `../alchemi_ht/data/chunks/chunk_*.csv`)
- **Verification** uses `../alchemi_ht/alchemi_ht.sif` and `../alchemi_ht/pipeline.py` directly
