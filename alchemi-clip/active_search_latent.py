"""
Phase 6 – Step 2: Latent-Space Active Search

Embeds millions of unexplored SMILES into a latent space, evaluates them with
the surrogate Brain, and selects the top UCB candidates for GPU verification.

Three encoder modes:
  --encoder foundation  : ChemBERTa pre-trained transformer (fast, broad exploration)
  --encoder gnn         : Custom 3D GNN trained on ALCHEMI data (slower, physics-aware)
  --encoder contrastive : MolecularCLIP 1D encoder (3D-aware text encoder, best of both)

Usage:
    python active_search_latent.py \
        --encoder contrastive \
        --unexplored-csv data/unexplored_gdb17_sample.csv \
        --surrogate-path models/surrogate_contrastive.pkl \
        --top-k 5000 \
        --output data/chunks/active_learning_batch.csv
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd

from embed_utils import embed_contrastive, embed_foundation, embed_gnn


# ---------------------------------------------------------------------------
# UCB Acquisition & Candidate Selection
# ---------------------------------------------------------------------------

def _ucb_from_forest(brain, X_latent, kappa):
    """Compute mean, std, and UCB from a single RandomForest."""
    tree_preds = np.array([t.predict(X_latent) for t in brain.estimators_])
    mu = np.mean(tree_preds, axis=0)
    sigma = np.std(tree_preds, axis=0)
    return mu, sigma, mu + kappa * sigma


# Sign convention: UCB *maximises*. For properties where *lower is better*
# (energy, energy_per_atom), we negate predictions so that maximising UCB
# still selects the most desirable molecules.
_NEGATE_TARGETS = {"energy_eV", "energy_per_atom"}


def evaluate_and_select(X_latent: np.ndarray, valid_smiles: list[str],
                        brain, kappa: float = 2.0,
                        top_k: int = 5000) -> pd.DataFrame:
    """
    Use the surrogate Brain to predict scores and select top UCB candidates.

    `brain` can be:
      - A single RandomForest  (single-target, backward-compatible)
      - A dict {target_name: RandomForest}  (multi-target bundle)

    For multi-target bundles the composite UCB is the mean of per-target
    UCB scores after z-score normalisation, so each property contributes
    equally regardless of scale.
    """
    print(f"Evaluating {len(valid_smiles)} candidates...")

    results_df = pd.DataFrame({"SMILES": valid_smiles})

    if isinstance(brain, dict):
        # Multi-target: one surrogate per property
        ucb_components = []
        for target_name, model in brain.items():
            mu, sigma, ucb = _ucb_from_forest(model, X_latent, kappa)

            # Negate "lower is better" targets so maximising UCB is always good
            sign = -1.0 if target_name in _NEGATE_TARGETS else 1.0
            ucb_signed = sign * ucb

            results_df[f"pred_{target_name}"] = sign * mu
            results_df[f"unc_{target_name}"] = sigma
            results_df[f"ucb_{target_name}"] = ucb_signed

            # Z-score normalise UCB so each property contributes equally
            ucb_z = (ucb_signed - ucb_signed.mean()) / (ucb_signed.std() + 1e-8)
            ucb_components.append(ucb_z)

        # Composite score: mean of z-scored per-target UCBs
        composite = np.mean(ucb_components, axis=0)
        results_df["UCB_Composite"] = composite
        sort_col = "UCB_Composite"

    else:
        # Single-target (backward-compatible)
        mu, sigma, ucb = _ucb_from_forest(brain, X_latent, kappa)
        results_df["Predicted_Score"] = mu
        results_df["Uncertainty"] = sigma
        results_df["UCB_Score"] = ucb
        sort_col = "UCB_Score"

    top_candidates = results_df.sort_values(
        by=sort_col, ascending=False
    ).head(top_k)

    return top_candidates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Latent-Space Active Learning Search"
    )
    parser.add_argument(
        "--encoder", choices=["foundation", "gnn", "contrastive"],
        required=True,
        help="Choose the latent space encoder.",
    )
    parser.add_argument(
        "--unexplored-csv", default="data/unexplored_gdb17_sample.csv",
        help="CSV with SMILES column of unexplored molecules.",
    )
    parser.add_argument(
        "--surrogate-path", default=None,
        help="Path to the trained surrogate model (.pkl). "
             "Defaults to models/surrogate_{encoder}.pkl",
    )
    parser.add_argument(
        "--gnn-model-path", default="models/custom_gnn.pt",
        help="Path to the GNN model weights (for --encoder gnn).",
    )
    parser.add_argument(
        "--contrastive-encoder-path", default="models/contrastive_1d_encoder.pt",
        help="Path to the contrastive 1D encoder (for --encoder contrastive).",
    )
    parser.add_argument("--latent-dim", type=int, default=512,
                        help="Latent dimension of the contrastive encoder.")
    parser.add_argument("--chunk-size", type=int, default=100000,
                        help="Number of SMILES to process per chunk.")
    parser.add_argument("--kappa", type=float, default=2.0,
                        help="UCB exploration-exploitation trade-off.")
    parser.add_argument("--top-k", type=int, default=5000,
                        help="Number of top candidates to select.")
    parser.add_argument(
        "--output", default="data/chunks/active_learning_batch.csv",
        help="Output CSV path for selected candidates.",
    )
    args = parser.parse_args()

    # Resolve surrogate path
    if args.surrogate_path is None:
        args.surrogate_path = f"models/surrogate_{args.encoder}.pkl"

    # 1. Load unexplored data
    print(f"Loading unexplored database: {args.unexplored_csv}")
    df_unexplored = pd.read_csv(args.unexplored_csv)

    # Accept either 'SMILES' or 'smiles' column
    smiles_col = "SMILES" if "SMILES" in df_unexplored.columns else "smiles"
    smiles_subset = df_unexplored[smiles_col].head(args.chunk_size).tolist()
    print(f"Processing {len(smiles_subset)} molecules with '{args.encoder}' encoder")

    # 2. Map to latent space
    if args.encoder == "foundation":
        X_latent, valid_smiles = embed_foundation(smiles_subset, validate=True)
    elif args.encoder == "gnn":
        X_latent, valid_smiles = embed_gnn(
            smiles_subset, gnn_model_path=args.gnn_model_path, validate=True,
        )
    elif args.encoder == "contrastive":
        X_latent, valid_smiles = embed_contrastive(
            smiles_subset, encoder_path=args.contrastive_encoder_path,
            latent_dim=args.latent_dim, validate=True,
        )

    print(f"Embedded {len(valid_smiles)} valid molecules "
          f"into {X_latent.shape[1]}D latent space")

    # 3. Load the surrogate Brain
    print(f"Loading surrogate model: {args.surrogate_path}")
    brain = joblib.load(args.surrogate_path)

    # 4. Predict and select top UCB candidates
    top_candidates = evaluate_and_select(
        X_latent, valid_smiles, brain,
        kappa=args.kappa, top_k=args.top_k,
    )

    # 5. Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    top_candidates.to_csv(args.output, index=False)
    print(f"\nSelected top {len(top_candidates)} candidates via "
          f"'{args.encoder}' latent space.")
    print(f"Saved to: {args.output}")

    # Print summary statistics
    print(f"\n--- Candidate Statistics ---")
    if "UCB_Composite" in top_candidates.columns:
        # Multi-target output
        for col in top_candidates.columns:
            if col.startswith("pred_") or col.startswith("ucb_"):
                print(f"  {col}: mean={top_candidates[col].mean():.4f}, "
                      f"std={top_candidates[col].std():.4f}")
        print(f"  UCB_Composite:   mean={top_candidates['UCB_Composite'].mean():.4f}, "
              f"std={top_candidates['UCB_Composite'].std():.4f}")
    else:
        # Single-target output
        print(f"  Predicted Score: mean={top_candidates['Predicted_Score'].mean():.4f}, "
              f"std={top_candidates['Predicted_Score'].std():.4f}")
        print(f"  Uncertainty:     mean={top_candidates['Uncertainty'].mean():.4f}, "
              f"std={top_candidates['Uncertainty'].std():.4f}")
        print(f"  UCB Score:       mean={top_candidates['UCB_Score'].mean():.4f}, "
              f"std={top_candidates['UCB_Score'].std():.4f}")


if __name__ == "__main__":
    main()
