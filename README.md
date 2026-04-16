# ALCHEMI — GPU-accelerated molecular screening on FASRC

An end-to-end stack for discovering organic electrolyte candidates from large chemical libraries (GDB-17, ~10^6–10^9 molecules). Three layers:

1. **Base container** (`container/`, `examples/`) — NVIDIA ALCHEMI Toolkit + PyTorch, packaged as a Singularity image. Used for generic molecular-dynamics demos.
2. **HT screening pipeline** (`alchemi_ht/`) — SMILES → 3D embed (RDKit) → AIMNet2 relaxation → HOMO/LUMO gap (GPU4PySCF DFT). Runs as a SLURM array over chunked CSVs.
3. **Active-learning loop** (`alchemi-clip/`) — MolecularCLIP contrastive encoder + RandomForest surrogate, snapshot-driven active search over the unexplored library, top-k candidates verified back through the HT pipeline.

A Python orchestrator (`orchestrator/`, driven by the `/ht-loop` slash command) ties the two pipelines together: HT screen runs as a continuous stream, AL iterations fire on a threshold trigger without blocking the screen.

## Prerequisites

- FASRC cluster access with `kempner_dev` partition/account
- GPU node (tested on NVIDIA H200)
- Singularity / Apptainer CE 4.4+ at `/usr/bin/singularity`
- Python 3.12 on the submit node for the orchestrator (`module load python/3.12.11-fasrc01` or use `/usr/bin/python3.12`)

## Layout

| Path | Description |
|------|-------------|
| `container/` | Base image: `alchemi.def` (NGC `pytorch:25.06-py3`, Python 3.12, PyTorch 2.8, CUDA 12.9), `build.sh`, and the built `alchemi.sif` (~13 GB, not checked in). Also `fetch_aimnet_assets.sh` + `aimnet_assets/` cache. |
| `examples/` | Minimal demos of the base container: `test_alchemi.py` (smoke test), `example_dynamics.py` (geometry optimization), `run.sh` (launcher). |
| `alchemi_ht/` | Phase 3: high-throughput screening pipeline. Extends `alchemi.sif` into `alchemi_ht.sif`. See [`alchemi_ht/README.md`](alchemi_ht/README.md). |
| `alchemi-clip/` | Phase 6: latent-space active learning. Extends `alchemi_ht.sif` into `alchemi_clip.sif`. See [`alchemi-clip/README.md`](alchemi-clip/README.md). |
| `orchestrator/` | `/ht-loop` orchestrator: submits + polls SLURM arrays, handles retries, drives the AL state machine. Includes `run_loop.sh` (tmux-based automation wrapper, agent-free or agent-driven). |
| `runs/` | Runtime state (screen stream + AL iterations). Gitignored except for `.gitkeep`. |
| `.claude/commands/ht-loop.md` | Slash command spec for the orchestrator. |
| `docs/`, `ARCHITECTURE.md` | Supplementary notes and top-level design. |

## Quick start

### 1. Build the base container (one-time, ~10–15 min)

```bash
cd container
sbatch build.sh                          # or: singularity build --fakeroot alchemi.sif alchemi.def
```

Then build the HT container on top:

```bash
cd ../alchemi_ht
apptainer build alchemi_ht.sif alchemi_ht.def
```

And the AL container (optional, only for research mode):

```bash
cd ../alchemi-clip
bash build.sh
```

### 2. Run the base-container demo

```bash
cd examples
bash run.sh                              # direct, must be on a GPU node
sbatch run.sh                            # or submit
```

Expected tail:

```
Dynamics run complete.
Final positions shape: torch.Size([7, 3])
```

### 3. Run the HT screen (orchestrator, v1)

The orchestrator drives the HT screen as a SLURM-array stream. It auto-classifies failures (timeout / OOM / error / infra), resubmits with resource bumps (2× mem for OOMs, 2× time for timeouts, 3-attempt cap), splits submissions when `MaxArraySize` would be exceeded, and transitions `sacct-reports-success + no-CSV` chunks to `anomaly` so pipeline bugs surface cleanly.

```bash
# Prep: convert + shuffle + chunk (one-time)
python3.12 alchemi_ht/convert_smi_to_csv.py \
    alchemi_ht/GDB17.50000000LL.smi.gz alchemi_ht/gdb17_subset.csv \
    --shuffle-seed 42
python3.12 alchemi_ht/chunk_data.py \
    --input-csv alchemi_ht/gdb17_subset.csv \
    --output-dir runs/screen/chunks

# Seed the AIMNet2 model cache so compute nodes don't need outbound HTTPS
bash container/fetch_aimnet_assets.sh

# Orchestrator
python3.12 -m orchestrator.ht_loop --mode screen --action init --shuffle-seed 42
python3.12 -m orchestrator.ht_loop --mode screen --action tick      # submit / poll / retry
python3.12 -m orchestrator.ht_loop --mode screen --action status
```

State: `runs/screen/screen_state.json`. Per-chunk results: `runs/screen/results/result_NNNN.csv`.

### 4. Run the research loop (orchestrator, v2)

Research mode runs both pipelines concurrently: the HT screen streams results, and AL iterations fire whenever enough new results accumulate (default: 100 new CSVs since the last AL iter). Each AL iter snapshots the current screen stream, trains a contrastive encoder + surrogate on it, runs active search over the unexplored library, and verifies the top-k candidates back through the HT pipeline. Verified results are merged back into the stream for the next iter.

```bash
# After screen init (above):
python3.12 -m orchestrator.ht_loop --mode research --action init --trigger-threshold 100

# One tick advances screen + AL by at most one phase each.
python3.12 -m orchestrator.ht_loop --mode research --action tick
python3.12 -m orchestrator.ht_loop --mode research --action status

# Force-start an AL iter below the threshold (debugging):
python3.12 -m orchestrator.ht_loop --mode research --action advance-al
```

State: `runs/al/al_state.json`. Per-iter artifacts: `runs/al/iter_NNN/` (snapshot symlinks, trained models, active-search batch, verify chunks, metrics).

AL phases (each is one tick):

```
idle → snapshot_building → training_contrastive → training_surrogate
     → active_search → preparing_verification → verification
     → merging → idle (iter++)
```

### 5. Automate ticking

Ticking is just `python3.12 -m orchestrator.ht_loop --action tick` on a schedule. Three ways to automate it, pick what fits:

**a. tmux loop (agent-free) — `orchestrator/run_loop.sh`**

Zero external dependencies beyond tmux + python. Deterministic, auditable, and free. Each tick runs the python command directly in a loop inside a tmux session with three panes (tick loop | `squeue` watch | result + AL-iter counts).

```bash
bash orchestrator/run_loop.sh start      # create session 'alchemi-loop'
bash orchestrator/run_loop.sh attach     # view live (Ctrl-b d to detach)
bash orchestrator/run_loop.sh status     # check from outside
bash orchestrator/run_loop.sh stop       # kill the session
bash orchestrator/run_loop.sh tick       # one-off tick, no tmux
bash orchestrator/run_loop.sh logs       # tail latest log
```

Defaults: 900-second interval, research mode, `runs/prod/` state. Override via `ALCHEMI_*` env vars; see `bash orchestrator/run_loop.sh help`.

**b. tmux loop (agent-orchestrated) — same script, `ALCHEMI_DRIVER=agent`**

Each tick invokes headless Claude Code (`claude -p "<prompt>"`) instead of running python directly. The agent runs the tick, reads state files, triages `pending_decisions` by kind (`anomalies`, `permanent_failures`, `al_phase_failed`), and reports in ≤4 lines of natural-language summary per tick. It cannot edit state files, cancel jobs, or modify code — bounded by the prompt + `--max-turns 8`.

```bash
ALCHEMI_DRIVER=agent bash orchestrator/run_loop.sh start
```

Agent-mode extras:
- `ALCHEMI_CLAUDE_MODEL` (default `sonnet`)
- `ALCHEMI_CLAUDE_PERM` (default `auto`; use `acceptEdits` for tighter control)
- `ALCHEMI_CLAUDE_MAX_TURNS` (default `8`)

Cost: ≈100k–200k tokens per day at 15-min intervals. Trivial for short runs, non-free for multi-day ones.

**c. `/ht-loop` slash command (inside a Claude Code session)**

See [`.claude/commands/ht-loop.md`](.claude/commands/ht-loop.md). Uses the `loop` skill or `CronCreate` to re-fire ticks within an active REPL session. Default cadence: 10 min running / 1 hour idle. Best when you're actively working in Claude Code and want tick summaries inline.

**Which driver?**

| | `bash` | `agent` | `/ht-loop` |
|---|---|---|---|
| Dependencies | tmux, python | + `claude` CLI | active Claude session |
| API cost | none | per-tick | per-tick |
| Runs unattended | yes | yes | only while session is open |
| Decision triage | accumulates | per-tick by agent | per-tick by agent |
| Natural-language summaries | no | yes | yes |
| Good for | long unattended production runs | runs where narrative progress matters | interactive work |

## Environment variables (manual runs)

```bash
export SINGULARITYENV_PYTHONNOUSERSITE=1                          # required
export SINGULARITY_CACHEDIR=/n/netscratch/kempner_dev/Lab/bdesinghu/.cache/   # recommended
```

The `examples/run.sh` and all `alchemi_ht/`, `alchemi-clip/` SLURM scripts set these automatically.

## Known issues

**`Batch.batch_idx` is int32 but PyTorch scatter ops require int64.** Bug in `nvalchemi-toolkit` v0.1.0. `examples/example_dynamics.py` monkey-patches it.

**NumPy version conflict.** NGC's PyTorch is compiled against NumPy 1.x; user-installed 2.x in `~/.local` can override it. All run scripts set `SINGULARITYENV_PYTHONNOUSERSITE=1` to prevent this.

**AIMNet2 assets dir is read-only inside the container.** `alchemi_ht/run_screening.slurm` bind-mounts `container/aimnet_assets/` over the container's `.../aimnet/calculators/assets` path. Seed with `bash container/fetch_aimnet_assets.sh` once (compute nodes typically lack outbound HTTPS). The `alchemi_ht.def` now also bakes the model in at build time.

**`dftd3` signature mismatch.** AIMNet 0.3.x expects the high-level torch wrapper at `nvalchemiops.torch.interactions.dispersion.dftd3`; the shim in earlier container builds re-exported the low-level warp kernel by mistake. `alchemi_ht/pipeline.py` installs a runtime `sys.modules` shim as a safety net; the fix is also baked into `alchemi_ht.def` for future rebuilds.

**`kempner_dev` partition is oversubscribed.** Expect minutes to hours of queue time. For small debugging runs on an interactive GPU session, invoke the container directly (see the example loop in the conversation history — run `singularity exec --nv ... pipeline.py` per chunk sequentially).

**SLURM copies batch scripts to `/var/slurmd/spool/`.** `$(dirname "${BASH_SOURCE[0]}")` inside a `#SBATCH` script resolves to the spool dir, not the source. The orchestrator passes absolute `HT_SCRIPT_DIR` / `AL_SCRIPT_DIR` via `--export` so the SLURM scripts find their siblings. Manual `sbatch` invocations still work via the `BASH_SOURCE` fallback.

## Troubleshooting

| Symptom | First place to look |
|---|---|
| Chunk stuck in `running` forever | `squeue -j <job_id>`; if cleared, check `last_sacct_state` field in `screen_state.json`. If `completed` but no CSV, it's an `anomaly` — look at `logs/screen_*.err`. |
| All chunks fail with "file not found: .sif" | Orchestrator's absolute-path conversion for `HT_SCRIPT_DIR` is missing — check `job_history[].overrides` and re-init if needed. |
| AL `training_contrastive` fails fast | Snapshot size too small — bump `--min-results`, or wait for more screen results. |
| Can't download AIMNet2 model from compute node | Run `bash container/fetch_aimnet_assets.sh` on the submit/login node (has egress); compute nodes typically don't. |
| Array submit rejected with "invalid array specification" | Chunk IDs ≥ `MaxArraySize`. The orchestrator auto-splits; if you're invoking `sbatch` manually, use `CHUNK_OFFSET`. |

## References

- [`alchemi_ht/README.md`](alchemi_ht/README.md) — HT pipeline internals, DFT tuning, per-column output schema.
- [`alchemi-clip/README.md`](alchemi-clip/README.md) — AL encoders, surrogate targets, evaluation metrics.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — top-level design decisions.
- [`.claude/commands/ht-loop.md`](.claude/commands/ht-loop.md) — orchestrator slash-command spec.
