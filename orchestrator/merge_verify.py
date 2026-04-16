"""Fold AL verification results back into the shared screen stream.

After an AL iteration's verification phase completes, `verify_result_*.csv`
files contain ground-truth DFT values for the top-k candidates. We want those
to flow into the next AL iteration's training snapshot — the easiest way is to
symlink them into `runs/screen/results/` with a distinguishing prefix.

We do NOT rewrite them as regular `result_NNNN.csv` (which would collide with
the screen stream's chunk numbering). Instead we use the `verify_result_*.csv`
prefix intact — snapshot builders that glob `result_*.csv` will NOT pick these
up, so we also emit a manifest addendum. Training scripts that use
`--results-dir` glob `result_*.csv` by default, which excludes verify files.

To make them visible to training, the merge function also symlinks them into
the screen results dir under a normalized name: `result_verify_NNNN.csv`.
These DO match the `result_*.csv` glob used by `train_contrastive.py` /
`train_surrogate.py`, so they fold in naturally.
"""

from __future__ import annotations

import re
from pathlib import Path


def merge_verify_into_stream(
    verify_results_dir: Path,
    screen_results_dir: Path,
) -> list[str]:
    """Symlink verify_result_*.csv files into screen_results_dir as result_verify_*.csv.

    Returns list of names added (new symlinks only). Idempotent.
    """
    verify_results_dir = Path(verify_results_dir).resolve()
    screen_results_dir = Path(screen_results_dir).resolve()
    screen_results_dir.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(r"verify_result_(\d+)\.csv$")
    added: list[str] = []

    for src in sorted(verify_results_dir.glob("verify_result_*.csv")):
        m = pattern.search(src.name)
        if not m:
            continue
        idx = m.group(1)
        # Namespace the symlink so it can't collide with a screen chunk ID.
        target_name = f"result_verify_{idx}.csv"
        target = screen_results_dir / target_name
        if target.is_symlink() or target.exists():
            continue
        target.symlink_to(src)
        added.append(target_name)

    return added


def count_verify_results(verify_results_dir: Path) -> int:
    verify_results_dir = Path(verify_results_dir)
    if not verify_results_dir.exists():
        return 0
    return sum(1 for _ in verify_results_dir.glob("verify_result_*.csv"))
