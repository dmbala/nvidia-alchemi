"""Split the master molecule CSV into chunks for Slurm array processing."""

import argparse
import math
import os

import pandas as pd

DEFAULT_CHUNK_SIZE = 5000


def main():
    parser = argparse.ArgumentParser(description="Chunk master CSV for array processing")
    parser.add_argument(
        "--input-csv",
        default="gdb17_subset.csv",
        help="Path to master CSV produced by convert_smi_to_csv.py",
    )
    parser.add_argument(
        "--output-dir",
        default="data/chunks",
        help="Directory to write chunk_NNNN.csv files into",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Molecules per chunk",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv, comment="#")
    n_chunks = math.ceil(len(df) / args.chunk_size)

    for i in range(n_chunks):
        start = i * args.chunk_size
        end = min(start + args.chunk_size, len(df))
        chunk = df.iloc[start:end]
        out_path = os.path.join(args.output_dir, f"chunk_{i:04d}.csv")
        chunk.to_csv(out_path, index=False)
        print(f"Wrote {len(chunk)} molecules to {out_path}")

    print(f"\nTotal: {len(df)} molecules -> {n_chunks} chunk(s)")


if __name__ == "__main__":
    main()
