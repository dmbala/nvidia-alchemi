"""
Aggregate results from all chunk CSVs and rank candidates.

Filtering criteria (from instructions):
  - Status == "Success"
  - Energy below the 50th percentile (stable molecules)
  - HOMO-LUMO gap above the 90th percentile (electrochemical stability)

Outputs a CSV of ranked candidates (default: top_1_percent_candidates.csv).
Also emits a JSON stats blob (via --stats-json) consumable by the orchestrator.
"""

import argparse
import glob
import json
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Aggregate and rank screening results")
    parser.add_argument(
        "--results-dir",
        default="data/results",
        help="Directory containing result_*.csv files",
    )
    parser.add_argument(
        "--output",
        default="top_1_percent_candidates.csv",
        help="Output CSV for ranked candidates",
    )
    parser.add_argument(
        "--stats-json",
        default=None,
        help="Optional path to emit a JSON stats summary for downstream consumers",
    )
    args = parser.parse_args()

    result_files = sorted(glob.glob(os.path.join(args.results_dir, "result_*.csv")))
    if not result_files:
        print(f"No result files found in {args.results_dir}/")
        return

    dfs = [pd.read_csv(f) for f in result_files]
    all_results = pd.concat(dfs, ignore_index=True)
    print(f"Aggregated {len(all_results)} molecules from {len(result_files)} chunk(s)")

    status_counts = all_results["status"].value_counts()
    print("\nStatus summary:")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")

    success = all_results[all_results["status"] == "Success"].copy()
    stats = {
        "n_chunks": len(result_files),
        "n_total": int(len(all_results)),
        "n_success": int(len(success)),
        "status_counts": {k: int(v) for k, v in status_counts.items()},
    }

    if success.empty:
        print("\nNo successful results to rank.")
        if args.stats_json:
            with open(args.stats_json, "w") as f:
                json.dump(stats, f, indent=2)
        return

    print(f"\nSuccessful molecules: {len(success)}")

    energy_threshold = success["energy_eV"].quantile(0.50)
    stable = success[success["energy_eV"] <= energy_threshold]
    print(f"Energy <= {energy_threshold:.4f} eV (50th pct): {len(stable)} molecules")

    gap_threshold = success["gap_eV"].quantile(0.90)
    candidates = stable[stable["gap_eV"] >= gap_threshold].copy()
    print(f"Gap >= {gap_threshold:.4f} eV (90th pct): {len(candidates)} candidates")

    candidates = candidates.sort_values("gap_eV", ascending=False)
    candidates.to_csv(args.output, index=False)
    print(f"\nWrote {len(candidates)} top candidates to {args.output}")

    stats.update(
        {
            "energy_threshold_50pct": float(energy_threshold),
            "gap_threshold_90pct": float(gap_threshold),
            "n_candidates": int(len(candidates)),
            "energy_eV_mean": float(success["energy_eV"].mean()),
            "gap_eV_mean": float(success["gap_eV"].mean()),
            "success_rate": float(len(success) / len(all_results)) if len(all_results) else 0.0,
        }
    )
    if args.stats_json:
        with open(args.stats_json, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Wrote stats summary to {args.stats_json}")


if __name__ == "__main__":
    main()
