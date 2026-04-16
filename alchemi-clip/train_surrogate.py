"""
Phase 6 – Step 1b: Extract Latent Vectors & Train the Surrogate Brain

Takes HT screening ground-truth results, encodes them into a latent space
using one of the three encoders, and trains an ensemble surrogate model
(Random Forest or XGBoost) to predict Energy and Gap from latent vectors.

Usage:
    # Train surrogate on contrastive embeddings (recommended)
    python train_surrogate.py \
        --encoder contrastive \
        --results-dir ../alchemi_ht/data/results \
        --target gap_eV \
        --output models/surrogate_contrastive.pkl

    # Train surrogate on foundation model embeddings
    python train_surrogate.py \
        --encoder foundation \
        --results-dir ../alchemi_ht/data/results \
        --target gap_eV \
        --output models/surrogate_foundation.pkl
"""

import argparse
import glob
import os

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import cross_val_predict


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ground_truth(results_dir: str) -> pd.DataFrame:
    """Load and filter successful HT screening results."""
    result_files = sorted(glob.glob(os.path.join(results_dir, "result_*.csv")))
    if not result_files:
        raise FileNotFoundError(f"No result files found in {results_dir}")

    dfs = [pd.read_csv(f) for f in result_files]
    all_results = pd.concat(dfs, ignore_index=True)

    # Keep only successful molecules with valid targets
    mask = (
        (all_results["status"] == "Success") &
        all_results["smiles"].notna() &
        all_results["energy_eV"].notna() &
        all_results["gap_eV"].notna()
    )
    data = all_results[mask].reset_index(drop=True)

    # Derive energy_per_atom (size-normalized stability)
    data["energy_per_atom"] = data["energy_eV"] / data["n_atoms"]

    print(f"Loaded {len(data)} ground-truth molecules from {len(result_files)} files")
    return data


from embed_utils import embed_contrastive, embed_foundation, embed_gnn


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

ALL_TARGETS = ["gap_eV", "energy_eV", "homo_eV", "lumo_eV", "energy_per_atom"]


def evaluate_surrogate(brain, X, y, target_name, n_folds=5):
    """Compute and print all evaluation metrics for a single target.

    Uses a single cross_val_predict pass (5 RF fits) instead of separate
    cross_val_predict + cross_val_score (which would be 10 fits).
    """
    n_folds = min(n_folds, len(y))
    y_pred_cv = cross_val_predict(brain, X, y, cv=n_folds)

    mae = mean_absolute_error(y, y_pred_cv)
    rmse = root_mean_squared_error(y, y_pred_cv)
    r2 = r2_score(y, y_pred_cv)
    rho, rho_pval = spearmanr(y, y_pred_cv)

    print(f"\n--- {target_name}: {n_folds}-fold CV Metrics ---")
    for k_frac in [0.01, 0.05, 0.10]:
        k = max(1, int(len(y) * k_frac))
        true_top_k = set(np.argsort(y)[::-1][:k])
        pred_top_k = set(np.argsort(y_pred_cv)[::-1][:k])
        recall = len(true_top_k & pred_top_k) / k
        print(f"  Top-{k_frac:.0%} recall (k={k}): {recall:.2%}")

    unit = "eV" if "eV" in target_name else "eV/atom"
    print(f"  R²:       {r2:.4f}")
    print(f"  MAE:      {mae:.4f} {unit}")
    print(f"  RMSE:     {rmse:.4f} {unit}")
    print(f"  Spearman: {rho:.4f}  (p={rho_pval:.2e})")


def train_surrogate(args):
    # 1. Load ground truth
    data = load_ground_truth(args.results_dir)
    smiles_list = data["smiles"].tolist()

    # 2. Extract latent embeddings (once, shared across all targets)
    if args.encoder == "foundation":
        X, _ = embed_foundation(smiles_list)
    elif args.encoder == "contrastive":
        X, _ = embed_contrastive(
            smiles_list, encoder_path=args.contrastive_encoder_path,
            latent_dim=args.latent_dim,
        )
    elif args.encoder == "gnn":
        X, _ = embed_gnn(smiles_list, gnn_model_path=args.gnn_model_path)

    # Resolve target list
    targets = ALL_TARGETS if args.target == "all" else [args.target]

    print(f"\nTraining data: X={X.shape}, targets={targets}")

    # 3. Train one surrogate per target
    bundle = {}
    for target_name in targets:
        y = data[target_name].values
        print(f"\n{'='*50}")
        print(f"Target '{target_name}': mean={y.mean():.4f}, std={y.std():.4f}")

        brain = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
        )

        evaluate_surrogate(brain, X, y, target_name)

        brain.fit(X, y)
        bundle[target_name] = brain
        print(f"  feature_importances top-5: "
              f"{np.argsort(brain.feature_importances_)[::-1][:5].tolist()}")

    # 4. Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if len(bundle) == 1:
        # Single target: save the model directly (backward-compatible)
        joblib.dump(bundle[targets[0]], args.output)
    else:
        # Multi-target: save the dict of models
        joblib.dump(bundle, args.output)

    print(f"\nSaved surrogate model to: {args.output}")
    print(f"  targets: {list(bundle.keys())}")


def main():
    parser = argparse.ArgumentParser(
        description="Train surrogate Brain on latent embeddings"
    )
    parser.add_argument(
        "--encoder", choices=["foundation", "gnn", "contrastive"],
        required=True,
        help="Encoder used to generate latent vectors.",
    )
    parser.add_argument(
        "--results-dir", required=True,
        help="Directory with HT screening result CSVs.",
    )
    parser.add_argument(
        "--target", default="gap_eV",
        choices=ALL_TARGETS + ["all"],
        help="Target property to predict. Use 'all' to train a multi-target "
             "surrogate bundle (default: gap_eV).",
    )
    parser.add_argument(
        "--contrastive-encoder-path",
        default="models/contrastive_1d_encoder.pt",
        help="Path to the contrastive 1D encoder weights.",
    )
    parser.add_argument(
        "--gnn-model-path", default="models/custom_gnn.pt",
        help="Path to the GNN model weights.",
    )
    parser.add_argument("--latent-dim", type=int, default=512,
                        help="Latent dimension of the contrastive encoder.")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument(
        "--output", default=None,
        help="Output path for the surrogate model. "
             "Defaults to models/surrogate_{encoder}.pkl",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = f"models/surrogate_{args.encoder}.pkl"

    train_surrogate(args)


if __name__ == "__main__":
    main()
