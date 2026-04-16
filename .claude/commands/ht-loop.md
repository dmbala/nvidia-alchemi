---
description: Orchestrate the HT screening stream (v1) and optionally the active-learning loop (v2). Submits SLURM arrays, polls sacct, resubmits failed chunks, and (in research mode) fires AL iterations on a snapshot trigger.
---

# /ht-loop — HT + active-learning orchestrator

Driver module: `orchestrator.ht_loop`. State files:
- `runs/screen/screen_state.json` — v1 producer state
- `runs/al/al_state.json` — v2 consumer state (only in research mode)

## Each tick

Run one of the following and parse the output:

```bash
# v1: HT screening only
python3.12 -m orchestrator.ht_loop --mode screen --action status
python3.12 -m orchestrator.ht_loop --mode screen --action tick

# v2: both pipelines
python3.12 -m orchestrator.ht_loop --mode research --action status
python3.12 -m orchestrator.ht_loop --mode research --action tick
```

`tick` advances the screen state machine and (in research mode) advances the AL state machine by **at most one phase**. It's safe to fire repeatedly — no double-submits, no race conditions; idempotent in `pending` / `running` states.

## Bootstrap

If the user asks to start fresh:

```bash
# Convert + chunk the master CSV (one-time)
python3.12 alchemi_ht/convert_smi_to_csv.py \
    alchemi_ht/GDB17.50000000LL.smi.gz alchemi_ht/gdb17_subset.csv \
    --shuffle-seed 42
python3.12 alchemi_ht/chunk_data.py \
    --input-csv alchemi_ht/gdb17_subset.csv \
    --output-dir runs/screen/chunks

# Init screen state
python3.12 -m orchestrator.ht_loop --mode screen --action init --shuffle-seed 42

# v2 only: init AL state (after screen init)
python3.12 -m orchestrator.ht_loop --mode research --action init \
    --trigger-threshold 100
```

## Cadence

Use the `loop` skill:
- **10 min** while any SLURM jobs are running.
- **1 hour** when everything is idle / waiting for data.
- **Stop** when all chunks are `done` AND (screen mode: no pending_decisions) OR (research mode: target number of AL iterations reached).

## AL phases (research mode)

```
idle → snapshot_building → training_contrastive → training_surrogate
     → active_search → preparing_verification → verification
     → merging → idle (iter++)
```

Each phase advances on its own tick. `snapshot_building`, `preparing_verification`, and `merging` are local (fast); the others submit a SLURM job and wait.

## Decisions to surface to the user (don't auto-resolve)

- `screen_state.json.pending_decisions`: permanent chunk failures (3 retries exhausted), or anomalies (sacct success but no CSV).
- `al_state.json.pending_decisions`: AL phase failed (encoder/surrogate/search didn't complete), or low snapshot size.
- `/ht-loop --mode research --action advance-al` forces an AL iter to start even below the trigger threshold (useful for debugging).

## Don't

- Don't resubmit a chunk already at the 3-attempt cap; it's in `failed_permanent` for a reason.
- Don't edit state files by hand.
- Don't delete `result_*.csv` files; they're the stream output AND the input to subsequent AL iterations.

## Configuration knobs (passed to `--action init`)

Screen mode:
- `--chunk-dir`, `--result-dir`, `--slurm-script`, `--shuffle-seed`

Research mode additionally:
- `--trigger-threshold N` (default 100) — fire new AL iter when N new screen results have landed since the last snapshot.
- `--min-results N` (default 10) — minimum screen results before the first AL iter.
- `--al-clip-dir`, `--ht-dir`, `--al-sif`, `--ht-sif`, `--aimnet-cache`, `--unexplored-csv`
