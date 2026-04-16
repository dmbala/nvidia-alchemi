"""
Prepare active learning candidates for verification via the HT pipeline.

Takes the active_learning_batch.csv output from active_search_latent.py,
reformats it to match the HT pipeline's expected input format (id, smiles, name),
and splits it into chunks for Slurm array processing.

Usage:
    python prepare_verification.py \
        --input data/chunks/active_learning_batch.csv \
        --chunk-size 500
"""

import argparse
import math
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Prepare active learning batch for HT verification"
    )
    parser.add_argument(
        "--input", default="data/chunks/active_learning_batch.csv",
        help="Active learning batch CSV from active_search_latent.py",
    )
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--output-dir", default="data/chunks")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    # Reformat to match HT pipeline input: id, smiles, name
    smiles_col = "SMILES" if "SMILES" in df.columns else "smiles"
    verify_df = pd.DataFrame({
        "id": range(1, len(df) + 1),
        "smiles": df[smiles_col],
        "name": "",
    })

    os.makedirs(args.output_dir, exist_ok=True)
    n_chunks = math.ceil(len(verify_df) / args.chunk_size)

    for i in range(n_chunks):
        start = i * args.chunk_size
        end = min(start + args.chunk_size, len(verify_df))
        chunk = verify_df.iloc[start:end]
        out_path = os.path.join(args.output_dir, f"verify_{i:04d}.csv")
        chunk.to_csv(out_path, index=False)

    print(f"Split {len(verify_df)} candidates into {n_chunks} verification chunks")
    print(f"Submit with: sbatch --array=0-{n_chunks - 1} run_verification.slrm")


if __name__ == "__main__":
    main()
