"""Snapshot builder: materialize a point-in-time view of the HT screen stream.

Each AL iteration consumes a *snapshot* — a directory of symlinks into
`runs/screen/results/` (and any merged verify results). The symlink form lets
the existing alchemi-clip training scripts consume the snapshot via their
`--results-dir` flag without modification.

The `snapshot.txt` manifest alongside the symlink dir records exact filenames
for reproducibility: an iteration can be retrained months later from the same
inputs (assuming the screen results haven't been deleted).
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path


def build_snapshot(
    iter_dir: Path,
    screen_results_dir: Path,
    min_results: int = 1,
) -> tuple[Path, list[str]]:
    """Materialize `iter_dir/screen_snapshot/` with symlinks to current results.

    Returns (snapshot_dir, manifest_entries). Raises if fewer than `min_results`
    CSVs exist.
    """
    screen_results_dir = Path(screen_results_dir).resolve()
    iter_dir = Path(iter_dir).resolve()
    snapshot_dir = iter_dir / "screen_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    available = sorted(screen_results_dir.glob("result_*.csv"))
    if len(available) < min_results:
        raise RuntimeError(
            f"Only {len(available)} result CSVs available; need at least {min_results}"
        )

    manifest: list[str] = []
    for src in available:
        link = snapshot_dir / src.name
        if link.is_symlink() or link.exists():
            # Idempotent: overwrite if existing but pointing elsewhere.
            try:
                link.unlink()
            except FileNotFoundError:
                pass
        link.symlink_to(src)
        manifest.append(src.name)

    manifest_path = iter_dir / "snapshot.txt"
    with manifest_path.open("w") as f:
        f.write(f"# built={_dt.datetime.now(_dt.timezone.utc).isoformat()}\n")
        f.write(f"# source={screen_results_dir}\n")
        f.write(f"# n_files={len(manifest)}\n")
        for name in manifest:
            f.write(f"{name}\n")

    return snapshot_dir, manifest


def count_available_results(screen_results_dir: Path) -> int:
    """Cheap: how many result_*.csv are in the screen stream right now?"""
    screen_results_dir = Path(screen_results_dir)
    if not screen_results_dir.exists():
        return 0
    return sum(1 for _ in screen_results_dir.glob("result_*.csv"))


def load_manifest(iter_dir: Path) -> list[str]:
    """Read snapshot.txt, return list of result CSV filenames (comments stripped)."""
    path = Path(iter_dir) / "snapshot.txt"
    if not path.exists():
        return []
    names: list[str] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            names.append(line)
    return names
