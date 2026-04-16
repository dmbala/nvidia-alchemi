"""Active-learning state machine (v2 `/ht-loop --mode research`).

Per-iteration phases:
    idle
        → snapshot_building           (local, fast)
        → training_contrastive        (SLURM job)
        → training_surrogate          (SLURM job)
        → active_search               (SLURM job)
        → preparing_verification      (local)
        → verification                (SLURM array)
        → merging                     (local)
        → idle                        (iter += 1)

Triggers:
    new_results:N   — fire when screen has N+ new results since last snapshot
    manual          — only on explicit `--advance-al`

Each tick reads state, inspects filesystem + sacct, advances at most one
phase. The AL state file lives at `runs/al/al_state.json`.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import merge_verify, snapshot


# --- phase constants ------------------------------------------------------

PHASE_IDLE = "idle"
PHASE_SNAPSHOT = "snapshot_building"
PHASE_TRAIN_ENCODER = "training_contrastive"
PHASE_TRAIN_SURROGATE = "training_surrogate"
PHASE_ACTIVE_SEARCH = "active_search"
PHASE_PREP_VERIFY = "preparing_verification"
PHASE_VERIFY = "verification"
PHASE_MERGE = "merging"

# Minimum results to seed an AL iter — prevents training on a degenerate set.
MIN_RESULTS_DEFAULT = 10

# Fallbacks when sacct isn't informative yet.
STATE_RUNNING = {"RUNNING", "PENDING", "CONFIGURING", "REQUEUED"}
STATE_SUCCESS = {"COMPLETED"}
STATE_FAIL = {"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "CANCELLED",
              "CANCELLED+", "BOOT_FAIL", "DEADLINE", "PREEMPTED"}


# --- state schema ---------------------------------------------------------

@dataclass
class IterState:
    iter: int
    phase: str = PHASE_SNAPSHOT
    snapshot_dir: str | None = None
    snapshot_size: int = 0
    models_dir: str | None = None
    batch_csv: str | None = None
    verify_chunk_dir: str | None = None
    verify_result_dir: str | None = None
    jobs: dict[str, str] = field(default_factory=dict)          # phase -> job_id
    job_state: dict[str, str] = field(default_factory=dict)     # job_id -> sacct state
    metrics: dict = field(default_factory=dict)
    started: str | None = None
    finished: str | None = None


@dataclass
class ALState:
    al_root: str
    screen_results_dir: str
    al_clip_dir: str
    ht_dir: str
    al_sif: str
    ht_sif: str
    aimnet_cache: str
    unexplored_csv: str
    trigger: dict = field(default_factory=lambda: {"type": "new_results", "threshold": 100})
    current_iter: IterState | None = None
    iter_history: list[IterState] = field(default_factory=list)
    last_snapshot_size: int = 0
    pending_decisions: list[dict] = field(default_factory=list)
    last_poll: str | None = None

    @classmethod
    def load(cls, path: Path) -> "ALState":
        if not path.exists():
            raise FileNotFoundError(f"No AL state at {path}")
        d = json.loads(path.read_text())
        current = IterState(**d["current_iter"]) if d.get("current_iter") else None
        history = [IterState(**x) for x in d.get("iter_history", [])]
        return cls(
            al_root=d["al_root"],
            screen_results_dir=d["screen_results_dir"],
            al_clip_dir=d["al_clip_dir"],
            ht_dir=d["ht_dir"],
            al_sif=d["al_sif"],
            ht_sif=d["ht_sif"],
            aimnet_cache=d["aimnet_cache"],
            unexplored_csv=d["unexplored_csv"],
            trigger=d.get("trigger", {"type": "new_results", "threshold": 100}),
            current_iter=current,
            iter_history=history,
            last_snapshot_size=d.get("last_snapshot_size", 0),
            pending_decisions=d.get("pending_decisions", []),
            last_poll=d.get("last_poll"),
        )

    def save(self, path: Path) -> None:
        payload = {
            "al_root": self.al_root,
            "screen_results_dir": self.screen_results_dir,
            "al_clip_dir": self.al_clip_dir,
            "ht_dir": self.ht_dir,
            "al_sif": self.al_sif,
            "ht_sif": self.ht_sif,
            "aimnet_cache": self.aimnet_cache,
            "unexplored_csv": self.unexplored_csv,
            "trigger": self.trigger,
            "current_iter": asdict(self.current_iter) if self.current_iter else None,
            "iter_history": [asdict(x) for x in self.iter_history],
            "last_snapshot_size": self.last_snapshot_size,
            "pending_decisions": self.pending_decisions,
            "last_poll": self.last_poll,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str))


# --- init -----------------------------------------------------------------

def init_state(
    state_path: Path,
    al_root: Path,
    screen_results_dir: Path,
    al_clip_dir: Path,
    ht_dir: Path,
    al_sif: Path,
    ht_sif: Path,
    aimnet_cache: Path,
    unexplored_csv: Path,
    trigger_threshold: int = 100,
) -> ALState:
    state = ALState(
        al_root=str(al_root.resolve()),
        screen_results_dir=str(screen_results_dir.resolve()),
        al_clip_dir=str(al_clip_dir.resolve()),
        ht_dir=str(ht_dir.resolve()),
        al_sif=str(al_sif.resolve()),
        ht_sif=str(ht_sif.resolve()),
        aimnet_cache=str(aimnet_cache.resolve()),
        unexplored_csv=str(unexplored_csv.resolve()),
        trigger={"type": "new_results", "threshold": trigger_threshold},
    )
    state.save(state_path)
    return state


# --- trigger --------------------------------------------------------------

def trigger_ready(state: ALState, min_results: int = MIN_RESULTS_DEFAULT) -> tuple[bool, str]:
    """Return (ready, reason). Cheap — just an ls."""
    if state.current_iter is not None:
        return False, f"iter {state.current_iter.iter} already in progress"
    available = snapshot.count_available_results(Path(state.screen_results_dir))
    if state.trigger.get("type") == "manual":
        return False, "manual trigger; use --advance-al"
    threshold = int(state.trigger.get("threshold", 100))
    new_results = available - state.last_snapshot_size
    if available < min_results:
        return False, f"only {available} result CSVs available (need ≥{min_results})"
    if new_results < threshold and state.last_snapshot_size > 0:
        return False, f"only {new_results} new since last iter (need ≥{threshold})"
    return True, f"{available} results total, {new_results} new since last iter"


# --- tick -----------------------------------------------------------------

def tick(state: ALState, state_path: Path, min_results: int = MIN_RESULTS_DEFAULT) -> dict:
    """Advance the AL state machine by at most one phase. Returns summary."""
    state.last_poll = _now()

    if state.current_iter is None:
        ready, reason = trigger_ready(state, min_results=min_results)
        if ready:
            _start_iter(state)
        else:
            state.save(state_path)
            return {"phase": "idle", "reason": reason}

    it = state.current_iter
    assert it is not None

    phase = it.phase
    if phase == PHASE_SNAPSHOT:
        _phase_snapshot(state, it, min_results=min_results)
    elif phase == PHASE_TRAIN_ENCODER:
        _advance_slurm_phase(state, it, PHASE_TRAIN_ENCODER, PHASE_TRAIN_SURROGATE,
                             submit_fn=_submit_train_contrastive)
    elif phase == PHASE_TRAIN_SURROGATE:
        _advance_slurm_phase(state, it, PHASE_TRAIN_SURROGATE, PHASE_ACTIVE_SEARCH,
                             submit_fn=_submit_train_surrogate)
    elif phase == PHASE_ACTIVE_SEARCH:
        _advance_slurm_phase(state, it, PHASE_ACTIVE_SEARCH, PHASE_PREP_VERIFY,
                             submit_fn=_submit_active_search)
    elif phase == PHASE_PREP_VERIFY:
        _phase_prep_verify(state, it)
    elif phase == PHASE_VERIFY:
        _advance_slurm_phase(state, it, PHASE_VERIFY, PHASE_MERGE,
                             submit_fn=_submit_verification)
    elif phase == PHASE_MERGE:
        _phase_merge(state, it)

    state.save(state_path)
    return {
        "iter": it.iter,
        "phase": it.phase,
        "snapshot_size": it.snapshot_size,
        "jobs": it.jobs,
    }


# --- phase handlers -------------------------------------------------------

def _start_iter(state: ALState) -> None:
    iter_no = (state.iter_history[-1].iter + 1) if state.iter_history else 1
    iter_dir = Path(state.al_root) / f"iter_{iter_no:03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    state.current_iter = IterState(iter=iter_no, started=_now())


def _phase_snapshot(state: ALState, it: IterState, min_results: int) -> None:
    iter_dir = Path(state.al_root) / f"iter_{it.iter:03d}"
    snap_dir, manifest = snapshot.build_snapshot(
        iter_dir=iter_dir,
        screen_results_dir=Path(state.screen_results_dir),
        min_results=min_results,
    )
    it.snapshot_dir = str(snap_dir)
    it.snapshot_size = len(manifest)
    it.models_dir = str(iter_dir / "models")
    it.batch_csv = str(iter_dir / "active_learning_batch.csv")
    it.verify_chunk_dir = str(iter_dir / "verify_chunks")
    it.verify_result_dir = str(iter_dir / "verify_results")
    it.phase = PHASE_TRAIN_ENCODER


def _advance_slurm_phase(
    state: ALState,
    it: IterState,
    phase: str,
    next_phase: str,
    submit_fn,
) -> None:
    """Generic slurm-job phase: submit → poll → transition on success."""
    job_id = it.jobs.get(phase)
    if job_id is None:
        job_id = submit_fn(state, it)
        it.jobs[phase] = job_id
        return  # will poll next tick

    slurm_state = _sacct_state(job_id)
    it.job_state[job_id] = slurm_state or "unknown"
    if slurm_state in STATE_SUCCESS:
        it.phase = next_phase
    elif slurm_state in STATE_FAIL:
        state.pending_decisions.append({
            "ts": _now(),
            "kind": "al_phase_failed",
            "iter": it.iter,
            "phase": phase,
            "job_id": job_id,
            "slurm_state": slurm_state,
            "question": f"AL phase {phase!r} (iter {it.iter}) ended in {slurm_state}. Retry, skip, or abort?",
        })
        it.phase = "failed"
    # else: still running; stay in phase for next tick.


def _phase_prep_verify(state: ALState, it: IterState) -> None:
    """Run prepare_verification.py locally (quick, CPU-only) to chunk candidates."""
    chunk_dir = Path(it.verify_chunk_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python3.12",
        str(Path(state.al_clip_dir) / "prepare_verification.py"),
        "--input", it.batch_csv,
        "--output-dir", str(chunk_dir),
        "--chunk-size", "500",
    ]
    subprocess.check_call(cmd)
    chunks = sorted(chunk_dir.glob("verify_*.csv"))
    if not chunks:
        state.pending_decisions.append({
            "ts": _now(),
            "kind": "al_no_verify_chunks",
            "iter": it.iter,
            "question": "prepare_verification.py produced no chunks. Investigate.",
        })
        it.phase = "failed"
        return
    it.metrics["n_verify_chunks"] = len(chunks)
    it.phase = PHASE_VERIFY


def _phase_merge(state: ALState, it: IterState) -> None:
    """Fold verify_result_*.csv into the screen stream, finish iter."""
    added = merge_verify.merge_verify_into_stream(
        verify_results_dir=Path(it.verify_result_dir),
        screen_results_dir=Path(state.screen_results_dir),
    )
    it.metrics["n_merged_into_stream"] = len(added)
    it.finished = _now()
    it.phase = PHASE_IDLE
    state.iter_history.append(it)
    state.last_snapshot_size = it.snapshot_size
    state.current_iter = None


# --- submitters -----------------------------------------------------------

def _submit_train_contrastive(state: ALState, it: IterState) -> str:
    slurm = Path(state.al_clip_dir) / "run_train_contrastive.slrm"
    exports = {
        "AL_SCRIPT_DIR": state.al_clip_dir,
        "AL_SNAPSHOT_DIR": it.snapshot_dir,
        "AL_MODELS_DIR": it.models_dir,
        "AL_SIF": state.al_sif,
    }
    return _sbatch(slurm, exports)


def _submit_train_surrogate(state: ALState, it: IterState) -> str:
    slurm = Path(state.al_clip_dir) / "run_train_surrogate.slrm"
    exports = {
        "AL_SCRIPT_DIR": state.al_clip_dir,
        "AL_SNAPSHOT_DIR": it.snapshot_dir,
        "AL_MODELS_DIR": it.models_dir,
        "AL_SIF": state.al_sif,
    }
    return _sbatch(slurm, exports)


def _submit_active_search(state: ALState, it: IterState) -> str:
    slurm = Path(state.al_clip_dir) / "run_active_search.slrm"
    exports = {
        "AL_SCRIPT_DIR": state.al_clip_dir,
        "AL_MODELS_DIR": it.models_dir,
        "AL_SIF": state.al_sif,
        "AL_UNEXPLORED_CSV": state.unexplored_csv,
        "AL_OUTPUT_CSV": it.batch_csv,
    }
    return _sbatch(slurm, exports)


def _submit_verification(state: ALState, it: IterState) -> str:
    slurm = Path(state.al_clip_dir) / "run_verification.slrm"
    chunks = sorted(Path(it.verify_chunk_dir).glob("verify_*.csv"))
    ids = [_extract_index(c.name) for c in chunks]
    ids = [x for x in ids if x is not None]
    if not ids:
        raise RuntimeError(f"No verify_*.csv chunks in {it.verify_chunk_dir}")
    array_spec = _compress_to_array(ids)
    exports = {
        "AL_SCRIPT_DIR": state.al_clip_dir,
        "HT_SCRIPT_DIR": state.ht_dir,
        "AL_CHUNK_DIR": it.verify_chunk_dir,
        "AL_RESULT_DIR": it.verify_result_dir,
        "AL_HT_SIF": state.ht_sif,
        "AIMNET_CACHE": state.aimnet_cache,
    }
    return _sbatch(slurm, exports, array_spec=array_spec)


# --- SLURM helpers --------------------------------------------------------

def _sbatch(slurm_script: Path, exports: dict[str, str], array_spec: str | None = None) -> str:
    slurm_script = slurm_script.resolve()
    cmd = ["sbatch", "--parsable"]
    if array_spec is not None:
        cmd += [f"--array={array_spec}"]
    export_str = "ALL," + ",".join(f"{k}={v}" for k, v in exports.items())
    cmd += [f"--export={export_str}", str(slurm_script)]
    out = subprocess.check_output(cmd, text=True).strip()
    return out.split(";")[0]


def _sacct_state(job_id: str) -> str:
    """Return the dominant sacct state for a job. Empty string if unknown."""
    try:
        out = subprocess.check_output(
            ["sacct", "-j", job_id, "--format=State", "--noheader", "--parsable2"],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""
    lines = [l.split("|")[0].split()[0].strip().upper()
             for l in out.strip().splitlines() if l]
    if not lines:
        return ""
    # For array jobs, lines include both the aggregate "JOB_[0-N]" entry and
    # per-task entries. Prioritize: if anything is running/pending, still running.
    # If all ended, check for failures.
    if any(s in STATE_RUNNING for s in lines):
        return "RUNNING"
    if any(s in STATE_FAIL for s in lines):
        return next(s for s in lines if s in STATE_FAIL)
    if all(s in STATE_SUCCESS for s in lines):
        return "COMPLETED"
    return lines[0]


def _extract_index(filename: str) -> int | None:
    m = re.search(r"(\d+)\.csv$", filename)
    return int(m.group(1)) if m else None


def _compress_to_array(ids: list[int]) -> str:
    if not ids:
        return ""
    nums = sorted(set(ids))
    ranges: list[str] = []
    start = prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = n
    ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ",".join(ranges)


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def summarize(state: ALState) -> dict:
    it = state.current_iter
    if it is None:
        available = snapshot.count_available_results(Path(state.screen_results_dir))
        new = available - state.last_snapshot_size
        return {
            "current_iter": None,
            "completed_iters": len(state.iter_history),
            "screen_available": available,
            "new_since_last": new,
            "threshold": state.trigger.get("threshold"),
            "pending_decisions": len(state.pending_decisions),
        }
    return {
        "current_iter": it.iter,
        "phase": it.phase,
        "snapshot_size": it.snapshot_size,
        "jobs": it.jobs,
        "metrics": it.metrics,
        "pending_decisions": len(state.pending_decisions),
    }
