"""Convert GDB17 .smi.gz files to the CSV format expected by the pipeline."""

import argparse
import csv
import gzip
import random


def main():
    parser = argparse.ArgumentParser(description="Convert .smi.gz to pipeline CSV")
    parser.add_argument("input", help="Input .smi.gz file")
    parser.add_argument("output", help="Output .csv file")
    parser.add_argument("--limit", type=int, default=0, help="Max molecules (0 = all)")
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=42,
        help="Random seed for shuffling rows (use -1 to disable shuffle)",
    )
    args = parser.parse_args()

    with gzip.open(args.input, "rt") as fin:
        smiles_list = [line.strip() for line in fin if line.strip()]

    if args.limit and len(smiles_list) > args.limit:
        smiles_list = smiles_list[: args.limit]

    if args.shuffle_seed >= 0:
        rng = random.Random(args.shuffle_seed)
        rng.shuffle(smiles_list)

    with open(args.output, "w", newline="") as fout:
        if args.shuffle_seed >= 0:
            fout.write(f"# shuffle_seed={args.shuffle_seed}\n")
        writer = csv.writer(fout)
        writer.writerow(["id", "smiles", "name"])
        for idx, smiles in enumerate(smiles_list, start=1):
            writer.writerow([idx, smiles, ""])

    print(f"Wrote {len(smiles_list)} molecules to {args.output}")


if __name__ == "__main__":
    main()
