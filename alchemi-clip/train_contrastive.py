"""
Phase 6 – Step 1a: Train the Contrastive Molecular Encoder (MolecularCLIP)

Aligns a 1D Transformer (SMILES via ChemBERTa) and a 3D GNN (relaxed
coordinates via SchNet) in a shared latent space using InfoNCE loss.

Inputs:
    - HT screening results with SMILES + optimized 3D coordinates
      (produced by Phase 3 pipeline.py)

Outputs:
    - models/contrastive_1d_encoder.pt  (the 3D-aware text encoder for fast screening)
    - models/contrastive_3d_encoder.pt  (the 3D GNN encoder, for reference)

Usage:
    python train_contrastive.py \
        --results-dir ../alchemi_ht/data/results \
        --epochs 100 \
        --batch-size 128 \
        --latent-dim 512 \
        --lr 3e-4 \
        --output-dir models
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from rdkit import Chem
from rdkit.Chem import AllChem


# ---------------------------------------------------------------------------
# 1D Encoder: ChemBERTa + projection head
# ---------------------------------------------------------------------------

class SmilesEncoder1D(nn.Module):
    """Wraps a frozen ChemBERTa backbone with a trainable projection head."""

    def __init__(self, latent_dim: int = 512, model_name: str = "DeepChem/ChemBERTa-77M-MTR"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)

        # Freeze the backbone — we only train the projection head
        for param in self.backbone.parameters():
            param.requires_grad = False

        hidden_dim = self.backbone.config.hidden_size  # 768 for ChemBERTa
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, smiles_list: list[str]) -> torch.Tensor:
        """Encode a list of SMILES strings into latent vectors."""
        tokens = self.tokenizer(
            smiles_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        tokens = {k: v.to(next(self.projection.parameters()).device) for k, v in tokens.items()}

        with torch.no_grad():
            backbone_out = self.backbone(**tokens)

        # [CLS] token embedding
        cls_embedding = backbone_out.last_hidden_state[:, 0, :]
        return self.projection(cls_embedding)


# ---------------------------------------------------------------------------
# 3D Encoder: Lightweight SchNet-style GNN
# ---------------------------------------------------------------------------

class SchNetEncoder3D(nn.Module):
    """
    A lightweight SchNet-inspired continuous-filter GNN that operates on
    atomic numbers + 3D coordinates. Produces a fixed-size molecular embedding
    via sum pooling.
    """

    def __init__(self, latent_dim: int = 512, hidden_dim: int = 256,
                 n_interactions: int = 3, cutoff: float = 5.0, max_z: int = 100):
        super().__init__()
        self.cutoff = cutoff
        self.embedding = nn.Embedding(max_z, hidden_dim)

        # Interaction blocks
        self.interactions = nn.ModuleList()
        for _ in range(n_interactions):
            self.interactions.append(
                SchNetInteraction(hidden_dim, cutoff)
            )

        # Projection to shared latent space
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, z: torch.Tensor, pos: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z:     Atomic numbers, shape [N_atoms]
            pos:   3D coordinates, shape [N_atoms, 3]
            batch: Batch assignment, shape [N_atoms]
        Returns:
            Molecular embeddings, shape [B, latent_dim]
        """
        h = self.embedding(z)

        for interaction in self.interactions:
            h = h + interaction(h, pos, batch)

        # Sum pooling per molecule
        n_molecules = batch.max().item() + 1
        out = torch.zeros(n_molecules, h.size(1), device=h.device)
        out.scatter_add_(0, batch.unsqueeze(1).expand_as(h), h)

        return self.projection(out)


class SchNetInteraction(nn.Module):
    """One SchNet interaction block with continuous-filter convolution."""

    def __init__(self, hidden_dim: int, cutoff: float, n_gaussians: int = 50):
        super().__init__()
        self.cutoff = cutoff

        # Radial basis functions (Gaussian expansion of distances)
        self.register_buffer(
            "centers", torch.linspace(0.0, cutoff, n_gaussians)
        )
        self.register_buffer(
            "widths", torch.full((n_gaussians,), 0.5 * (cutoff / n_gaussians))
        )

        # Continuous filter network: distance -> filter weight
        self.filter_net = nn.Sequential(
            nn.Linear(n_gaussians, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Atom-wise transformation
        self.atom_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h: torch.Tensor, pos: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        # Compute pairwise distances
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # [N, N, 3]
        dist = diff.norm(dim=-1)  # [N, N]

        # Apply cutoff mask
        mask = (dist < self.cutoff) & (dist > 1e-6)  # exclude self-loops

        # Gaussian expansion of distances
        rbf = torch.exp(-((dist.unsqueeze(-1) - self.centers) ** 2) /
                        (2 * self.widths ** 2))  # [N, N, n_gaussians]

        # Continuous filter
        W = self.filter_net(rbf)  # [N, N, hidden_dim]

        # Filter-weighted message passing
        messages = W * h.unsqueeze(0)  # [N, N, hidden_dim]
        messages = messages * mask.unsqueeze(-1).float()

        # Aggregate messages
        agg = messages.sum(dim=1)  # [N, hidden_dim]

        return self.atom_net(agg)


# ---------------------------------------------------------------------------
# MolecularCLIP: the contrastive model
# ---------------------------------------------------------------------------

class MolecularCLIP(nn.Module):
    """
    Aligns 1D (SMILES) and 3D (coordinates) encoders via symmetric InfoNCE loss.
    """

    def __init__(self, latent_dim: int = 512, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.encoder_1d = SmilesEncoder1D(latent_dim=latent_dim)
        self.encoder_3d = SchNetEncoder3D(latent_dim=latent_dim)
        # Learnable temperature (log-parameterized for stability)
        self.log_temperature = nn.Parameter(torch.tensor(np.log(1.0 / temperature)))

    def forward(self, smiles_list, z, pos, batch):
        """Compute symmetric InfoNCE loss for a batch of molecules."""
        z_1d = self.encoder_1d(smiles_list)
        z_3d = self.encoder_3d(z, pos, batch)

        # L2 normalize (critical for cosine similarity)
        z_1d = F.normalize(z_1d, p=2, dim=1)
        z_3d = F.normalize(z_3d, p=2, dim=1)

        # Scaled cosine similarity matrix
        temperature = self.log_temperature.exp().clamp(min=0.01, max=100.0)
        logits = torch.matmul(z_1d, z_3d.T) / temperature

        # Labels: diagonal (molecule i's 1D should match molecule i's 3D)
        batch_size = z_1d.size(0)
        labels = torch.arange(batch_size, device=z_1d.device)

        # Symmetric InfoNCE loss
        loss_1d_to_3d = F.cross_entropy(logits, labels)
        loss_3d_to_1d = F.cross_entropy(logits.T, labels)

        return (loss_1d_to_3d + loss_3d_to_1d) / 2.0


# ---------------------------------------------------------------------------
# Dataset: loads HT screening results
# ---------------------------------------------------------------------------

class MolecularPairDataset(Dataset):
    """
    Loads HT screening results and produces (SMILES, atomic_numbers, coords) pairs.

    Expects CSVs from pipeline.py with columns:
        smiles, status, opt_coords, n_atoms
    where opt_coords is "x,y,z;x,y,z;..." from the ALCHEMI relaxation.
    """

    def __init__(self, results_dir: str):
        result_files = sorted(glob.glob(os.path.join(results_dir, "result_*.csv")))
        if not result_files:
            raise FileNotFoundError(f"No result files found in {results_dir}")

        dfs = [pd.read_csv(f) for f in result_files]
        all_results = pd.concat(dfs, ignore_index=True)

        # Filter to successful runs with valid coordinates
        mask = (
            (all_results["status"] == "Success") &
            all_results["opt_coords"].notna() &
            all_results["smiles"].notna()
        )
        self.data = all_results[mask].reset_index(drop=True)
        print(f"Loaded {len(self.data)} molecules with valid 3D coordinates")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        smiles = row["smiles"]

        # Parse optimized coordinates: "x,y,z;x,y,z;..."
        coords_str = row["opt_coords"]
        coords = np.array(
            [[float(v) for v in atom.split(",")] for atom in coords_str.split(";")],
            dtype=np.float32,
        )

        # Get atomic numbers from SMILES (including implicit H from the 3D structure)
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mol = Chem.AddHs(mol)
            atomic_nums = np.array(
                [atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int64
            )
        else:
            # Fallback: infer from number of coordinates
            atomic_nums = np.full(len(coords), 6, dtype=np.int64)  # assume carbon

        # Ensure coords and atomic_nums have same length
        min_len = min(len(coords), len(atomic_nums))
        coords = coords[:min_len]
        atomic_nums = atomic_nums[:min_len]

        return smiles, atomic_nums, coords


def collate_molecular_pairs(batch):
    """Custom collate function to handle variable-size molecules."""
    smiles_list = []
    all_z = []
    all_pos = []
    all_batch_idx = []

    for i, (smiles, atomic_nums, coords) in enumerate(batch):
        smiles_list.append(smiles)
        all_z.append(torch.from_numpy(atomic_nums))
        all_pos.append(torch.from_numpy(coords))
        all_batch_idx.append(torch.full((len(atomic_nums),), i, dtype=torch.long))

    z = torch.cat(all_z, dim=0)
    pos = torch.cat(all_pos, dim=0)
    batch_idx = torch.cat(all_batch_idx, dim=0)

    return smiles_list, z, pos, batch_idx


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset and dataloader
    dataset = MolecularPairDataset(args.results_dir)
    n_workers = min(4, os.cpu_count() or 1)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=n_workers,
        collate_fn=collate_molecular_pairs,
        drop_last=True,
        pin_memory=True,
    )

    # Model
    model = MolecularCLIP(latent_dim=args.latent_dim, temperature=0.07).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    os.makedirs(args.output_dir, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for smiles_list, z, pos, batch_idx in dataloader:
            z = z.to(device)
            pos = pos.to(device)
            batch_idx = batch_idx.to(device)

            loss = model(smiles_list, z, pos, batch_idx)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        lr = optimizer.param_groups[0]["lr"]
        temp = model.log_temperature.exp().item()
        print(f"Epoch {epoch:3d}/{args.epochs} | Loss: {avg_loss:.4f} | "
              f"LR: {lr:.2e} | Temp: {temp:.4f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save the full model
            torch.save(model.state_dict(), os.path.join(args.output_dir, "molecular_clip.pt"))
            # Save ONLY the 1D encoder for ultra-fast screening
            torch.save(
                model.encoder_1d.state_dict(),
                os.path.join(args.output_dir, "contrastive_1d_encoder.pt"),
            )
            # Save the 3D encoder for reference
            torch.save(
                model.encoder_3d.state_dict(),
                os.path.join(args.output_dir, "contrastive_3d_encoder.pt"),
            )
            print(f"  -> Saved best model (loss={best_loss:.4f})")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Saved to: {args.output_dir}/contrastive_1d_encoder.pt")


def main():
    parser = argparse.ArgumentParser(
        description="Train MolecularCLIP: contrastive 1D-3D molecular encoder"
    )
    parser.add_argument(
        "--results-dir", required=True,
        help="Directory with HT screening result CSVs (result_*.csv)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output-dir", default="models")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
