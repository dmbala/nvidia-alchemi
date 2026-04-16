# ALCHEMI Code Architecture

ALCHEMI (NVIDIA's `nvalchemi-toolkit`) is a two-package system for GPU-accelerated molecular dynamics and machine-learned interatomic potential (MLIP) inference. It is built around a clean kernel separation philosophy.

## Two-Package Architecture

| Package | Role |
|---------|------|
| `nvalchemiops` | Low-level GPU kernels written in NVIDIA Warp (JIT-compiled CUDA). Framework-agnostic (has `torch/`, `jax/` bindings). |
| `nvalchemi` | High-level toolkit — model wrappers, dynamics engines, data structures, hooks. Pure PyTorch. |

## How Kernels Are Separated Out

Each physical interaction is isolated into its own kernel module in `nvalchemiops/interactions/`.

### 1. Electrostatics (long-range Coulomb)

- `nvalchemiops/interactions/electrostatics/ewald_kernels.py` — Warp GPU kernels for Ewald summation (real-space erfc-damped + reciprocal-space structure factor). Supports float32/float64 via overload dispatch. The math splits E\_total = E\_real + E\_reciprocal - E\_self - E\_background.
- `nvalchemiops/interactions/electrostatics/pme_kernels.py` — Warp kernels for Particle Mesh Ewald (O(N log N) via FFT). Green's function, B-spline charge assignment, self-energy/background corrections.
- `nvalchemiops/interactions/electrostatics/coulomb.py` — Direct Coulomb summation.
- `nvalchemiops/interactions/electrostatics/dsf.py` — Damped Shifted Force method (alternative to Ewald).
- `nvalchemiops/torch/interactions/electrostatics/ewald.py` — PyTorch bindings wrapping the Warp Ewald kernels with autograd support.
- `nvalchemiops/torch/interactions/electrostatics/pme.py` — PyTorch bindings for PME (handles FFT in PyTorch since Warp doesn't support FFT).
- `nvalchemiops/torch/interactions/electrostatics/parameters.py` — Auto-estimation of alpha, k-vectors, mesh dimensions.

### 2. Lennard-Jones (short-range vdW)

- `nvalchemiops/interactions/lj.py` — Warp kernels for LJ energy, forces, and virial tensor. Uses neighbor-matrix format for O(N) scaling. Supports half-list (Newton's 3rd law) and C2-continuous switching functions.

### 3. Dispersion Correction (DFT-D3)

- `nvalchemiops/interactions/dispersion/_dftd3.py` — Warp kernels for Grimme's DFT-D3(BJ) dispersion correction. Includes coordination number calculation, C6 interpolation, and Becke-Johnson damping.

### 4. Switching Functions

- `nvalchemiops/interactions/switching.py` — C2-continuous quintic switching (`s(x) = 1 - 10x^3 + 15x^4 - 6x^5`) as `@wp.func` callable from any Warp kernel.

### 5. Neighbor Lists

- `nvalchemiops/neighbors/` — Multiple GPU-accelerated algorithms: `batch_naive.py` (< 2000 atoms/system), `batch_cell_list.py` (>= 2000 atoms), `naive_dual_cutoff.py`. Includes Verlet skin-based rebuild detection (`rebuild_detection.py`).

### 6. Math Kernels

- `nvalchemiops/math/` — Spherical harmonics, splines, GTO basis functions.

## Warp Dispatch System

All GPU kernels use a centralized dispatch mechanism (`nvalchemiops/warp_dispatch.py`):

- `register_overloads()` — registers float32/float64 kernel variants at import time.
- `build_dispatch_table()` — maps `(axis_key, dtype)` to overloaded kernels.
- `dispatch()` — looks up and launches the right kernel via `wp.launch()`.

This means zero runtime kernel selection overhead — everything is pre-registered.

## Model Layer (`nvalchemi/models/`)

Each interaction kernel is wrapped in a model wrapper that implements `BaseModelMixin`:

| Wrapper | Kernel Source | Force Computation | Key Optimization |
|---------|--------------|-------------------|-----------------|
| `LennardJonesModelWrapper` | Warp LJ kernel | Analytical (in-kernel) | Pre-allocated output buffers; Newton's 3rd law half-list |
| `EwaldModelWrapper` | Warp Ewald kernels | Analytical (in-kernel) | k-vector/alpha cache per cell; auto-invalidation for NPT |
| `PMEModelWrapper` | Warp PME + PyTorch FFT | Analytical (in-kernel) | O(N log N) via FFT; B-spline interpolation; mesh dimension caching |
| `DFTD3ModelWrapper` | Warp D3 kernel | Analytical (in-kernel) | Auto-download of Fortran reference params; angstrom/Bohr conversion |
| `MACEWrapper` | MACE GNN (PyTorch) | Autograd (`-dE/dr`) | GPU one-hot lookup table for `node_attrs`; cuEquivariance + `torch.compile` support |
| `AIMNet2Wrapper` | AIMNet2 NN (PyTorch) | Autograd (`-dE/dr`) | External neighbor list; affine strain trick for stresses; Coulomb/D3 intentionally disabled for pipeline composition |

The critical design insight: analytical-force models (LJ, Ewald, PME, D3) compute forces inside the GPU kernel itself, avoiding the overhead of PyTorch autograd. The ML models (MACE, AIMNet2) use autograd because their energy functions are defined by neural network weights.

## Pipeline Composition (`nvalchemi/models/pipeline.py`)

Models compose via `PipelineModelWrapper` using groups:

```python
pipe = PipelineModelWrapper(groups=[
    PipelineGroup(steps=[aimnet2, ewald], use_autograd=True),  # shared autograd
    PipelineGroup(steps=[dftd3]),                               # analytical forces
])
```

Or simply via the `+` operator:

```python
combined = mace_model + dftd3_model  # additive energy/force summation
```

Each `PipelineGroup` can choose its own derivative strategy:

- `use_autograd=True` — sum energies from all steps, differentiate once (efficient for ML models).
- `use_autograd=False` — each step provides its own forces (for analytical kernels).

Outputs are summed across groups via `sum_outputs()` for additive keys (energy, forces, stress).

## Where is the Training?

There is no training code in ALCHEMI. The toolkit is an inference and simulation engine, not a training framework. Here is how it works:

- **MACE models** are loaded from pre-trained checkpoints via `MACEWrapper.from_checkpoint("medium-0b2")`. The MACE models themselves are trained externally using the `mace-torch` package.
- **AIMNet2 models** are loaded from pre-trained checkpoints via `AIMNet2Wrapper.from_checkpoint("aimnet2")`. Training happens in the separate `aimnet` package.
- **LJ, Ewald, PME, D3** are physics-based (no learnable parameters beyond known physical constants).

The `MACEWrapper` does have `export_model()` and embedding computation support (`compute_embeddings()`), which suggests it could be used as part of a fine-tuning pipeline, but the training loop itself would be external.

## Dynamics Engine (`nvalchemi/dynamics/`)

The dynamics layer is where simulation runs happen:

- **`BaseDynamics`** — base class coordinating model evaluation with integrator updates. Uses a hook system (`BEFORE_COMPUTE`, `AFTER_COMPUTE`, etc.) for extensibility.
- **Integrators**: `NVE` (microcanonical), `NVTLangevin`, `NVTNoseHoover`, `NPT`, `NPH` — each backed by Warp kernels in `nvalchemiops/dynamics/integrators/`.
- **Optimizers**: `FIRE`, `FIRE2`, `FIRE2VariableCell` — geometry optimization with convergence detection.
- **Hooks**: `NeighborListHook` (auto-rebuilds neighbor lists with Verlet skin), `ConvergenceHook`, `NaNDetectorHook`, `WrapPeriodicHook`, `BiasedPotentialHook`.
- **Data sinks**: `GPUBuffer`, `HostMemory`, `ZarrData` — for trajectory streaming.
- **`FusedStage`** — fuses multiple dynamics stages on a single GPU with shared batch and forward pass.
- **`DistributedPipeline`** — multi-rank pipeline execution.

The `NeighborListHook` is heavily optimized: it pre-allocates staging buffers, auto-selects between naive and cell-list algorithms based on system size, and supports Verlet skin to skip unnecessary rebuilds.

## Package Layout

```
nvalchemiops (GPU kernels, framework-agnostic)
├── interactions/
│   ├── electrostatics/  ← Ewald, PME, Coulomb, DSF (Warp kernels)
│   ├── lj.py            ← Lennard-Jones (Warp kernel)
│   ├── dispersion/      ← DFT-D3 (Warp kernel)
│   └── switching.py     ← C2 switching function
├── neighbors/           ← GPU neighbor list algorithms
├── dynamics/            ← Integrator/optimizer Warp kernels
├── torch/               ← PyTorch bindings with autograd
└── jax/                 ← JAX bindings

nvalchemi (high-level toolkit, PyTorch)
├── models/
│   ├── base.py          ← BaseModelMixin + ModelConfig protocol
│   ├── pipeline.py      ← Composable model pipeline (+ operator)
│   ├── lj.py, ewald.py, pme.py, dftd3.py  ← Physics wrappers
│   └── mace.py, aimnet2.py               ← ML model wrappers
├── dynamics/
│   ├── base.py          ← BaseDynamics engine
│   ├── integrators/     ← NVE, NVT, NPT, NPH
│   └── optimizers/      ← FIRE, FIRE2
├── hooks/               ← NeighborList, PBC wrap, bias potential
└── data/                ← AtomicData, Batch, DataPipes
```

The design is cleanly layered: physics kernels are written once in Warp, bound to multiple frameworks, and composed at the model wrapper level via a uniform `BaseModelMixin` interface and the pipeline system.
