"""ht_loop.py — orchestrator driver for /ht-loop.

Modes:
  --mode screen   v1: Pipeline A (HT screening) as a stream.
  --mode research v2: both pipelines — screen keeps running, AL iterations fire
                     on a snapshot trigger (default: 100 new results).

Each invocation is one tick: read state, advance, write state.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import al_monitor as am
from . import screen_monitor as sm


DEFAULT_RUNS_DIR = Path("runs")
DEFAULT_CHUNK_DIR = DEFAULT_RUNS_DIR / "screen" / "chunks"
DEFAULT_RESULT_DIR = DEFAULT_RUNS_DIR / "screen" / "results"
DEFAULT_SCREEN_STATE = DEFAULT_RUNS_DIR / "screen" / "screen_state.json"
DEFAULT_SLURM_SCRIPT = Path("alchemi_ht") / "run_screening.slurm"
DEFAULT_AL_ROOT = DEFAULT_RUNS_DIR / "al"
DEFAULT_AL_STATE = DEFAULT_AL_ROOT / "al_state.json"
DEFAULT_AL_CLIP_DIR = Path("alchemi-clip")
DEFAULT_HT_DIR = Path("alchemi_ht")
DEFAULT_AL_SIF = DEFAULT_AL_CLIP_DIR / "alchemi_clip.sif"
DEFAULT_HT_SIF = DEFAULT_HT_DIR / "alchemi_ht.sif"
DEFAULT_AIMNET_CACHE = Path("container") / "aimnet_assets"
DEFAULT_UNEXPLORED_CSV = DEFAULT_HT_DIR / "gdb17_subset.csv"


# ---------- screen mode (v1) ----------

def _screen_state_path(args: argparse.Namespace) -> Path:
    return Path(args.state) if args.state else DEFAULT_SCREEN_STATE


def cmd_screen_init(args: argparse.Namespace) -> int:
    state_path = _screen_state_path(args)
    chunk_dir = Path(args.chunk_dir)
    result_dir = Path(args.result_dir)
    slurm_script = Path(args.slurm_script)

    if not chunk_dir.exists():
        print(f"ERROR: chunk dir {chunk_dir} does not exist", file=sys.stderr)
        return 2
    if not slurm_script.exists():
        print(f"ERROR: slurm script {slurm_script} does not exist", file=sys.stderr)
        return 2

    state = sm.init_state(
        state_path=state_path,
        chunk_dir=chunk_dir,
        result_dir=result_dir,
        slurm_script=slurm_script,
        shuffle_seed=args.shuffle_seed,
    )
    print(f"Initialized screen state at {state_path}")
    print(f"  chunks: {len(state.chunks)}")
    print(f"  chunk_dir: {state.chunk_dir}")
    print(f"  result_dir: {state.result_dir}")
    print(f"  MaxArraySize: {state.max_array_size}")
    if len(state.chunks) > state.max_array_size:
        n_windows = (len(state.chunks) + state.max_array_size - 1) // state.max_array_size
        print(f"  (will be split into ~{n_windows} windowed submissions)")
    return 0


def cmd_screen_tick(args: argparse.Namespace) -> int:
    state_path = _screen_state_path(args)
    if not state_path.exists():
        print(f"ERROR: no state at {state_path}; run `--action init` first", file=sys.stderr)
        return 2
    state = sm.ScreenState.load(state_path)

    if not state.job_history:
        if not sm.ensure_sbatch_available():
            print("ERROR: sbatch not on PATH; cannot submit", file=sys.stderr)
            return 3
        job_ids = sm.submit_initial(state, state_path)
        if len(job_ids) == 1:
            print(f"Submitted initial array job: {job_ids[0]}")
        else:
            print(f"Submitted {len(job_ids)} windowed array jobs: {job_ids}")
        summary = sm.summarize(state)
        _print_screen_summary(summary)
        return 0

    summary = sm.poll(state, state_path)
    reloaded = sm.ScreenState.load(state_path)
    retriable = sum(
        1 for cs in reloaded.chunks.values()
        if cs.status in {"timeout", "oom", "error", "infra"} and cs.attempts < sm.MAX_ATTEMPTS
    )
    if retriable and sm.ensure_sbatch_available():
        new_jobs = sm.resubmit_failed(reloaded, state_path)
        if new_jobs:
            print(f"Resubmitted failed chunks in {len(new_jobs)} new job(s): {new_jobs}")
        summary = sm.summarize(reloaded)

    _print_screen_summary(summary)
    return 0


def cmd_screen_status(args: argparse.Namespace) -> int:
    state_path = _screen_state_path(args)
    if not state_path.exists():
        print(f"No screen state at {state_path}")
        return 1
    state = sm.ScreenState.load(state_path)
    summary = sm.summarize(state)
    _print_screen_summary(summary)
    _print_decisions(state.pending_decisions)
    return 0


def _print_screen_summary(summary: dict) -> None:
    print("--- screen state ---")
    print(f"total chunks: {summary['total']}")
    for status, count in sorted(summary["by_status"].items()):
        print(f"  {status}: {count}")
    if summary.get("last_poll"):
        print(f"last poll: {summary['last_poll']}")
    if summary.get("pending_decisions"):
        print(f"pending decisions: {summary['pending_decisions']}")


# ---------- research mode (v2) ----------

def _al_state_path(args: argparse.Namespace) -> Path:
    return Path(args.al_state) if args.al_state else DEFAULT_AL_STATE


def cmd_research_init(args: argparse.Namespace) -> int:
    """Initialize the AL state file. Screen state must exist first (use --mode screen)."""
    al_path = _al_state_path(args)
    screen_path = _screen_state_path(args)
    if not screen_path.exists():
        print(f"ERROR: screen state missing at {screen_path}.", file=sys.stderr)
        print("Run `--mode screen --action init` first.", file=sys.stderr)
        return 2

    screen = sm.ScreenState.load(screen_path)
    state = am.init_state(
        state_path=al_path,
        al_root=Path(args.al_root),
        screen_results_dir=Path(screen.result_dir),
        al_clip_dir=Path(args.al_clip_dir),
        ht_dir=Path(args.ht_dir),
        al_sif=Path(args.al_sif),
        ht_sif=Path(args.ht_sif),
        aimnet_cache=Path(args.aimnet_cache),
        unexplored_csv=Path(args.unexplored_csv),
        trigger_threshold=args.trigger_threshold,
    )
    print(f"Initialized AL state at {al_path}")
    print(f"  al_root: {state.al_root}")
    print(f"  trigger: {state.trigger}")
    print(f"  screen_results_dir: {state.screen_results_dir}")
    return 0


def cmd_research_tick(args: argparse.Namespace) -> int:
    """One tick: advance screen side, then advance AL side by at most one phase."""
    # Step 1: screen tick (reuses v1 logic).
    screen_rc = cmd_screen_tick(args)
    if screen_rc not in (0,):
        return screen_rc

    # Step 2: AL tick.
    al_path = _al_state_path(args)
    if not al_path.exists():
        print(f"\nAL state missing at {al_path}. Skipping AL tick.")
        print("Run `--mode research --action init` to initialize.")
        return 0

    state = am.ALState.load(al_path)
    summary = am.tick(state, al_path, min_results=args.min_results)
    _print_al_summary(am.summarize(am.ALState.load(al_path)))
    _print_decisions(state.pending_decisions)
    return 0


def cmd_research_status(args: argparse.Namespace) -> int:
    screen_rc = cmd_screen_status(args)

    al_path = _al_state_path(args)
    if not al_path.exists():
        print(f"\nAL state missing at {al_path}")
        return screen_rc
    state = am.ALState.load(al_path)
    _print_al_summary(am.summarize(state))
    _print_decisions(state.pending_decisions)
    return 0


def cmd_research_advance_al(args: argparse.Namespace) -> int:
    """Force an AL iteration to start (bypasses the trigger threshold)."""
    al_path = _al_state_path(args)
    if not al_path.exists():
        print(f"ERROR: AL state missing at {al_path}", file=sys.stderr)
        return 2
    state = am.ALState.load(al_path)
    if state.current_iter is not None:
        print(f"Iter {state.current_iter.iter} already in progress; nothing to do.")
        return 0
    state.trigger = {"type": "new_results", "threshold": 0}
    state.save(al_path)
    print("Trigger threshold lowered to 0. Next --action tick will start a new iter.")
    return 0


def _print_al_summary(summary: dict) -> None:
    print("\n--- AL state ---")
    if summary.get("current_iter") is None:
        print(f"  idle (completed iters: {summary['completed_iters']})")
        print(f"  screen results available: {summary.get('screen_available', 0)}")
        new = summary.get("new_since_last", 0)
        thr = summary.get("threshold")
        print(f"  new since last iter: {new} (trigger: ≥{thr})")
    else:
        print(f"  iter {summary['current_iter']} — phase: {summary['phase']}")
        print(f"  snapshot size: {summary['snapshot_size']}")
        if summary.get("jobs"):
            for phase, jid in summary["jobs"].items():
                print(f"  {phase}: job {jid}")


def _print_decisions(decisions: list[dict]) -> None:
    if not decisions:
        return
    print("\nPending decisions:")
    for d in decisions:
        print(f"  [{d.get('ts','?')}] {d.get('kind','?')}: {d.get('question','')}")
        for k in ("chunks", "iter", "phase", "job_id"):
            if k in d:
                print(f"    {k}: {d[k]}")


# ---------- parser ----------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ht_loop", description="Orchestrator for HT + active learning")
    p.add_argument("--mode", choices=["screen", "research"], default="screen")
    p.add_argument("--action",
                   choices=["init", "tick", "status", "advance-al"],
                   default="tick")

    # Screen-side
    p.add_argument("--state", default=None, help=f"screen state path (default: {DEFAULT_SCREEN_STATE})")
    p.add_argument("--chunk-dir", default=str(DEFAULT_CHUNK_DIR))
    p.add_argument("--result-dir", default=str(DEFAULT_RESULT_DIR))
    p.add_argument("--slurm-script", default=str(DEFAULT_SLURM_SCRIPT))
    p.add_argument("--shuffle-seed", type=int, default=None)

    # AL-side
    p.add_argument("--al-state", default=None, help=f"AL state path (default: {DEFAULT_AL_STATE})")
    p.add_argument("--al-root", default=str(DEFAULT_AL_ROOT))
    p.add_argument("--al-clip-dir", default=str(DEFAULT_AL_CLIP_DIR))
    p.add_argument("--ht-dir", default=str(DEFAULT_HT_DIR))
    p.add_argument("--al-sif", default=str(DEFAULT_AL_SIF))
    p.add_argument("--ht-sif", default=str(DEFAULT_HT_SIF))
    p.add_argument("--aimnet-cache", default=str(DEFAULT_AIMNET_CACHE))
    p.add_argument("--unexplored-csv", default=str(DEFAULT_UNEXPLORED_CSV))
    p.add_argument("--trigger-threshold", type=int, default=100,
                   help="Fire new AL iter when N new screen results have landed")
    p.add_argument("--min-results", type=int, default=am.MIN_RESULTS_DEFAULT,
                   help="Minimum screen results before first AL iter")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "screen":
        if args.action == "init":   return cmd_screen_init(args)
        if args.action == "tick":   return cmd_screen_tick(args)
        if args.action == "status": return cmd_screen_status(args)
        print(f"action {args.action!r} not valid for --mode screen", file=sys.stderr)
        return 1
    if args.mode == "research":
        if args.action == "init":        return cmd_research_init(args)
        if args.action == "tick":        return cmd_research_tick(args)
        if args.action == "status":      return cmd_research_status(args)
        if args.action == "advance-al":  return cmd_research_advance_al(args)
        print(f"action {args.action!r} not valid for --mode research", file=sys.stderr)
        return 1
    print(f"unknown --mode {args.mode!r}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
