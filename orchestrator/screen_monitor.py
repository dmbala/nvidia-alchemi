"""HT screening producer: submits SLURM array jobs, polls sacct, classifies failures.

Writes per-chunk status into runs/screen/screen_state.json. The orchestrator driver
(ht_loop.py) calls into this module on each tick.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


MAX_ATTEMPTS = 3
DEFAULT_MAX_ARRAY_SIZE = 10000  # fallback if scontrol is unavailable

# Exit-code → failure classification. See plan's "Job failure handling" section.
EXIT_TIMEOUT = {124, 9}       # SIGTERM from SLURM wall-time kill, GNU timeout convention
EXIT_OOM = {137}              # SIGKILL from oom-killer
STATE_INFRA = {"NODE_FAIL", "PREEMPTED", "BOOT_FAIL"}
STATE_TIMEOUT = {"TIMEOUT", "DEADLINE"}
STATE_CANCELLED = {"CANCELLED", "CANCELLED+"}
STATE_SUCCESS = {"COMPLETED"}
STATE_RUNNING = {"RUNNING", "PENDING", "CONFIGURING", "REQUEUED", "RESIZING"}


@dataclass
class ChunkStatus:
    status: str = "pending"           # pending | running | done | timeout | oom | error | infra | cancelled | failed_permanent | anomaly
    attempts: int = 0
    last_job_id: str | None = None
    last_failure: str | None = None
    failures: list[str] = field(default_factory=list)
    csv: str | None = None
    last_sacct_state: str | None = None  # tracks SLURM's verdict independently of CSV presence


@dataclass
class ScreenState:
    chunk_dir: str
    result_dir: str
    slurm_script: str
    shuffle_seed: int | None = None
    max_array_size: int = DEFAULT_MAX_ARRAY_SIZE
    job_history: list[dict] = field(default_factory=list)
    chunks: dict[str, ChunkStatus] = field(default_factory=dict)
    last_poll: str | None = None
    pending_decisions: list[dict] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "ScreenState":
        if not path.exists():
            raise FileNotFoundError(f"No screen state at {path}")
        data = json.loads(path.read_text())
        chunks = {k: ChunkStatus(**v) for k, v in data.get("chunks", {}).items()}
        return cls(
            chunk_dir=data["chunk_dir"],
            result_dir=data["result_dir"],
            slurm_script=data["slurm_script"],
            shuffle_seed=data.get("shuffle_seed"),
            max_array_size=data.get("max_array_size", DEFAULT_MAX_ARRAY_SIZE),
            job_history=data.get("job_history", []),
            chunks=chunks,
            last_poll=data.get("last_poll"),
            pending_decisions=data.get("pending_decisions", []),
        )

    def save(self, path: Path) -> None:
        payload = {
            "chunk_dir": self.chunk_dir,
            "result_dir": self.result_dir,
            "slurm_script": self.slurm_script,
            "shuffle_seed": self.shuffle_seed,
            "max_array_size": self.max_array_size,
            "job_history": self.job_history,
            "chunks": {k: asdict(v) for k, v in self.chunks.items()},
            "last_poll": self.last_poll,
            "pending_decisions": self.pending_decisions,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))


def discover_chunks(chunk_dir: Path) -> list[str]:
    pattern = re.compile(r"chunk_(\d+)\.csv$")
    ids = []
    for p in sorted(chunk_dir.glob("chunk_*.csv")):
        m = pattern.search(p.name)
        if m:
            ids.append(str(int(m.group(1))))
    return ids


def init_state(
    state_path: Path,
    chunk_dir: Path,
    result_dir: Path,
    slurm_script: Path,
    shuffle_seed: int | None,
    max_array_size: int | None = None,
) -> ScreenState:
    ids = discover_chunks(chunk_dir)
    if not ids:
        raise RuntimeError(f"No chunk_*.csv files in {chunk_dir}")
    result_dir.mkdir(parents=True, exist_ok=True)
    size = max_array_size if max_array_size is not None else detect_max_array_size()
    state = ScreenState(
        chunk_dir=str(chunk_dir.resolve()),
        result_dir=str(result_dir.resolve()),
        slurm_script=str(slurm_script.resolve()),
        shuffle_seed=shuffle_seed,
        max_array_size=size,
        chunks={cid: ChunkStatus() for cid in ids},
    )
    state.save(state_path)
    return state


def detect_max_array_size() -> int:
    """Query `scontrol show config` for MaxArraySize. Falls back to default on failure."""
    try:
        out = subprocess.check_output(["scontrol", "show", "config"], text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return DEFAULT_MAX_ARRAY_SIZE
    m = re.search(r"MaxArraySize\s*=\s*(\d+)", out)
    return int(m.group(1)) if m else DEFAULT_MAX_ARRAY_SIZE


def _sbatch(
    slurm_script: Path,
    array_spec: str,
    chunk_dir: Path,
    result_dir: Path,
    chunk_offset: int = 0,
    time_min: int | None = None,
    mem_gb: int | None = None,
) -> str:
    """Submit an array job and return the job ID as a string."""
    # Always absolute — sbatch copies the script to /var/slurmd/spool/, so
    # BASH_SOURCE inside the job resolves wrong and any relative paths break.
    slurm_script = slurm_script.resolve()
    chunk_dir = chunk_dir.resolve()
    result_dir = result_dir.resolve()
    script_dir = slurm_script.parent

    cmd = ["sbatch", "--parsable"]
    if time_min is not None:
        cmd += [f"--time={time_min}"]
    if mem_gb is not None:
        cmd += [f"--mem={mem_gb}G"]
    exports = [
        f"HT_SCRIPT_DIR={script_dir}",
        f"CHUNK_DIR={chunk_dir}",
        f"RESULT_DIR={result_dir}",
        f"CHUNK_OFFSET={chunk_offset}",
    ]
    cmd += [f"--array={array_spec}", f"--export=ALL,{','.join(exports)}", str(slurm_script)]
    out = subprocess.check_output(cmd, text=True).strip()
    # --parsable may return "JOBID" or "JOBID;CLUSTER"
    return out.split(";")[0]


def _window_chunks(chunk_ids: list[str], max_array_size: int) -> dict[int, list[str]]:
    """Group chunk IDs into windows by floor(chunk_id / max_array_size).

    SLURM rejects array indices >= MaxArraySize, so we submit each window separately
    with CHUNK_OFFSET = window_start, and use SLURM indices in [0, max_array_size).
    """
    windows: dict[int, list[str]] = {}
    for cid in chunk_ids:
        n = int(cid)
        window_start = (n // max_array_size) * max_array_size
        windows.setdefault(window_start, []).append(cid)
    return windows


def _submit_windowed(
    state: ScreenState,
    chunk_ids: list[str],
    time_min: int | None = None,
    mem_gb: int | None = None,
) -> list[tuple[str, int, str]]:
    """Submit chunks as one or more windowed arrays. Returns [(job_id, offset, array_spec)]."""
    if not chunk_ids:
        return []
    windows = _window_chunks(chunk_ids, state.max_array_size)
    submissions: list[tuple[str, int, str]] = []
    for offset, cids in sorted(windows.items()):
        # SLURM array indices within this window are (chunk_id - offset).
        local_ids = [str(int(c) - offset) for c in cids]
        array_spec = _compress_to_array(local_ids)
        jid = _sbatch(
            Path(state.slurm_script),
            array_spec,
            Path(state.chunk_dir),
            Path(state.result_dir),
            chunk_offset=offset,
            time_min=time_min,
            mem_gb=mem_gb,
        )
        submissions.append((jid, offset, array_spec))
    return submissions


def submit_initial(state: ScreenState, state_path: Path) -> list[str]:
    """Submit all pending chunks as one or more windowed arrays. Returns job IDs."""
    pending = [cid for cid, s in state.chunks.items() if s.status == "pending"]
    if not pending:
        return []
    submissions = _submit_windowed(state, pending)
    window_by_chunk = {
        cid: offset
        for offset, cids in _window_chunks(pending, state.max_array_size).items()
        for cid in cids
    }
    job_by_window = {offset: jid for jid, offset, _ in submissions}
    for cid in pending:
        offset = window_by_chunk[cid]
        state.chunks[cid].status = "running"
        state.chunks[cid].attempts += 1
        state.chunks[cid].last_job_id = job_by_window[offset]
    for jid, offset, spec in submissions:
        state.job_history.append(
            {"job_id": jid, "array": spec, "offset": offset, "ts": _now()}
        )
    state.save(state_path)
    return [jid for jid, _, _ in submissions]


def _compress_to_array(ids: list[str]) -> str:
    """Compress ['0','1','2','5','7','8'] to '0-2,5,7-8'."""
    if not ids:
        return ""
    nums = sorted(set(int(x) for x in ids))
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


def poll(state: ScreenState, state_path: Path) -> dict:
    """Poll sacct + filesystem, update per-chunk statuses. Returns tick summary."""
    # Gather all job IDs we've submitted.
    job_ids = [h["job_id"] for h in state.job_history]
    if job_ids:
        _update_from_sacct(state, job_ids)

    # Cross-check filesystem: even if sacct is laggy, a CSV on disk means done.
    result_dir = Path(state.result_dir)
    for cid, cs in state.chunks.items():
        expected = result_dir / f"result_{int(cid):04d}.csv"
        if expected.exists():
            if cs.status != "done":
                cs.status = "done"
                cs.csv = expected.name
        else:
            if cs.status == "done":
                # Defensive: don't auto-revert, but flag an anomaly.
                cs.status = "anomaly"
                cs.last_failure = "csv_disappeared"
            elif cs.status == "running" and cs.last_sacct_state == "completed":
                # SLURM claims success but nothing was written. Pipeline bug — do not retry.
                cs.status = "anomaly"
                cs.last_failure = "completed_no_csv"

    _surface_decisions(state)
    state.last_poll = _now()
    state.save(state_path)

    return summarize(state)


def _surface_decisions(state: ScreenState) -> None:
    """Idempotent: add pending_decisions for anomaly chunks (and permanent failures)."""
    anomalies = sorted(
        (cid for cid, cs in state.chunks.items() if cs.status == "anomaly"),
        key=lambda x: int(x),
    )
    if anomalies and not any(
        d.get("kind") == "anomalies" and d.get("chunks") == anomalies
        for d in state.pending_decisions
    ):
        state.pending_decisions.append(
            {
                "ts": _now(),
                "kind": "anomalies",
                "chunks": anomalies,
                "question": "SLURM reported success but no result CSVs were written. "
                "Likely a pipeline bug; check logs/. Retry anyway, or stop?",
            }
        )

    permanent = sorted(
        (cid for cid, cs in state.chunks.items() if cs.status == "failed_permanent"),
        key=lambda x: int(x),
    )
    if permanent and not any(
        d.get("kind") == "permanent_failures" and d.get("chunks") == permanent
        for d in state.pending_decisions
    ):
        state.pending_decisions.append(
            {
                "ts": _now(),
                "kind": "permanent_failures",
                "chunks": permanent,
                "question": "Investigate these chunks manually, or continue without them?",
            }
        )


def _update_from_sacct(state: ScreenState, job_ids: Iterable[str]) -> None:
    fmt = "JobID,State,ExitCode"
    jobs_arg = ",".join(job_ids)
    try:
        out = subprocess.check_output(
            ["sacct", "-j", jobs_arg, f"--format={fmt}", "--noheader", "--parsable2"],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return  # sacct unavailable or job too new; retry next tick

    # Map parent_job_id -> chunk_offset from job_history so we can recover
    # the real chunk ID from the SLURM array task ID.
    offset_by_job = {h["job_id"]: int(h.get("offset", 0)) for h in state.job_history}

    for line in out.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 3:
            continue
        jobid_field, slurm_state, exit_code = parts[0], parts[1], parts[2]
        # jobid_field looks like "6123456_41" or "6123456_41.batch" — we want the array task.
        m = re.match(r"^(\d+)_(\d+)(?:\..*)?$", jobid_field)
        if not m:
            continue
        parent_job = m.group(1)
        task_id = int(m.group(2))
        offset = offset_by_job.get(parent_job, 0)
        chunk_id = str(task_id + offset)
        cs = state.chunks.get(chunk_id)
        if cs is None or cs.last_job_id != parent_job:
            continue
        _apply_sacct(cs, slurm_state, exit_code)


def _apply_sacct(cs: ChunkStatus, slurm_state: str, exit_code: str) -> None:
    slurm_state = slurm_state.split()[0].strip().upper()
    code_parts = exit_code.split(":")
    numeric = int(code_parts[0]) if code_parts and code_parts[0].isdigit() else -1
    signal = int(code_parts[1]) if len(code_parts) > 1 and code_parts[1].isdigit() else 0

    if slurm_state in STATE_RUNNING:
        cs.status = "running"
        cs.last_sacct_state = "running"
        return
    if slurm_state in STATE_SUCCESS and numeric == 0:
        # Record completion; filesystem poll is authoritative for done-vs-anomaly.
        cs.last_sacct_state = "completed"
        return
    if slurm_state in STATE_TIMEOUT or numeric in EXIT_TIMEOUT or signal == 15:
        cs.last_sacct_state = slurm_state.lower()
        _mark_failure(cs, "timeout")
        return
    if numeric in EXIT_OOM or signal == 9:
        cs.last_sacct_state = slurm_state.lower() or "oom"
        _mark_failure(cs, "oom")
        return
    if slurm_state in STATE_INFRA:
        cs.last_sacct_state = slurm_state.lower()
        _mark_failure(cs, "infra")
        return
    if slurm_state in STATE_CANCELLED:
        cs.status = "cancelled"
        cs.last_failure = "cancelled"
        cs.last_sacct_state = "cancelled"
        return
    if slurm_state == "FAILED" or numeric != 0:
        cs.last_sacct_state = slurm_state.lower() or "failed"
        _mark_failure(cs, "error")


def _mark_failure(cs: ChunkStatus, kind: str) -> None:
    cs.status = kind
    cs.last_failure = kind
    cs.failures.append(kind)


def resubmit_failed(state: ScreenState, state_path: Path) -> list[str]:
    """Group retriable failures by resource bump, window by MaxArraySize, resubmit."""
    retriable = {"timeout", "oom", "error", "infra"}
    groups: dict[tuple[int | None, int | None], list[str]] = {}
    for cid, cs in state.chunks.items():
        if cs.status not in retriable:
            continue
        if cs.attempts >= MAX_ATTEMPTS:
            cs.status = "failed_permanent"
            continue
        bump = _resource_bump(cs)
        groups.setdefault(bump, []).append(cid)

    all_job_ids: list[str] = []
    for (time_min, mem_gb), cids in groups.items():
        if not cids:
            continue
        submissions = _submit_windowed(state, cids, time_min=time_min, mem_gb=mem_gb)
        window_by_chunk = {
            cid: offset
            for offset, window_cids in _window_chunks(cids, state.max_array_size).items()
            for cid in window_cids
        }
        job_by_window = {offset: jid for jid, offset, _ in submissions}
        for cid in cids:
            offset = window_by_chunk[cid]
            cs = state.chunks[cid]
            cs.status = "running"
            cs.attempts += 1
            cs.last_job_id = job_by_window[offset]
        for jid, offset, spec in submissions:
            all_job_ids.append(jid)
            state.job_history.append(
                {
                    "job_id": jid,
                    "array": spec,
                    "offset": offset,
                    "ts": _now(),
                    "overrides": {"time_min": time_min, "mem_gb": mem_gb},
                }
            )

    _surface_decisions(state)
    state.save(state_path)
    return all_job_ids


def _resource_bump(cs: ChunkStatus) -> tuple[int | None, int | None]:
    """Return (time_minutes, mem_gb) overrides for a retry based on last failure."""
    base_time_min = 60     # matches run_screening.slurm default
    base_mem_gb = 32
    factor = 2 ** max(0, cs.attempts)  # attempts already incremented before retry
    if cs.last_failure == "timeout":
        return (base_time_min * factor, None)
    if cs.last_failure == "oom":
        return (None, base_mem_gb * factor)
    if cs.last_failure in {"error", "infra"}:
        return (None, None)
    return (None, None)


def summarize(state: ScreenState) -> dict:
    counts: dict[str, int] = {}
    for cs in state.chunks.values():
        counts[cs.status] = counts.get(cs.status, 0) + 1
    return {
        "total": len(state.chunks),
        "by_status": counts,
        "pending_decisions": len(state.pending_decisions),
        "last_poll": state.last_poll,
    }


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def ensure_sbatch_available() -> bool:
    return shutil.which("sbatch") is not None
