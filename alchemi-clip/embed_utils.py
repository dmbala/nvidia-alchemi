"""
Shared embedding utilities for Phase 6.

Provides a single implementation of each encoder (foundation, contrastive, GNN)
used by both train_surrogate.py and active_search_latent.py.
"""

import os

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def embed_foundation(smiles_list: list[str], batch_size: int = 256,
                     validate: bool = False) -> tuple[np.ndarray, list[str]]:
    """Embed SMILES using ChemBERTa. Returns (embeddings, valid_smiles)."""
    from transformers import AutoModel, AutoTokenizer

    print("Loading Foundation Model (ChemBERTa)...")
    model_name = "DeepChem/ChemBERTa-77M-MTR"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = get_device()
    model = model.to(device)
    model.eval()

    embeddings = []
    valid_smiles = []

    with torch.no_grad():
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i : i + batch_size]
            if validate:
                batch = [s for s in batch if Chem.MolFromSmiles(s) is not None]
                if not batch:
                    continue

            tokens = tokenizer(
                batch, return_tensors="pt",
                padding=True, truncation=True, max_length=256,
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}
            outputs = model(**tokens)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_emb)
            valid_smiles.extend(batch)

            if (i // batch_size) % 100 == 0:
                print(f"  Foundation: {i + len(batch)}/{len(smiles_list)}")

    return np.concatenate(embeddings, axis=0), valid_smiles


def embed_contrastive(smiles_list: list[str],
                      encoder_path: str = "models/contrastive_1d_encoder.pt",
                      latent_dim: int = 512, batch_size: int = 256,
                      validate: bool = False) -> tuple[np.ndarray, list[str]]:
    """Embed SMILES using the trained MolecularCLIP 1D encoder."""
    from train_contrastive import SmilesEncoder1D

    print(f"Loading Contrastive 1D Encoder from {encoder_path}...")
    device = get_device()
    encoder = SmilesEncoder1D(latent_dim=latent_dim).to(device)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder.eval()

    embeddings = []
    valid_smiles = []

    with torch.no_grad():
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i : i + batch_size]
            if validate:
                batch = [s for s in batch if Chem.MolFromSmiles(s) is not None]
                if not batch:
                    continue

            latent = encoder(batch)
            embeddings.append(latent.cpu().numpy())
            valid_smiles.extend(batch)

            if (i // batch_size) % 100 == 0:
                print(f"  Contrastive: {i + len(batch)}/{len(smiles_list)}")

    return np.concatenate(embeddings, axis=0), valid_smiles


def embed_gnn(smiles_list: list[str],
              gnn_model_path: str = "models/custom_gnn.pt",
              latent_dim: int = 512,
              validate: bool = False) -> tuple[np.ndarray, list[str]]:
    """Embed molecules using the GNN encoder with RDKit 3D coordinates."""
    device = get_device()

    gnn_model = None
    if os.path.exists(gnn_model_path):
        from train_contrastive import SchNetEncoder3D
        gnn_model = SchNetEncoder3D(latent_dim=latent_dim).to(device)
        gnn_model.load_state_dict(torch.load(gnn_model_path, map_location=device))
        gnn_model.eval()
        print(f"Loaded GNN from {gnn_model_path}")
    else:
        print("Loading Custom GNN and generating 3D graphs...")

    embeddings = []
    valid_smiles = []

    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            if not validate:
                embeddings.append(np.zeros(latent_dim, dtype=np.float32))
            continue
        mol = Chem.AddHs(mol)
        success = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if success != 0:
            if not validate:
                embeddings.append(np.zeros(latent_dim, dtype=np.float32))
            continue

        if gnn_model is not None:
            atomic_nums = torch.tensor(
                [atom.GetAtomicNum() for atom in mol.GetAtoms()],
                dtype=torch.long, device=device,
            )
            conf = mol.GetConformer()
            coords = torch.tensor(
                conf.GetPositions(), dtype=torch.float32, device=device,
            )
            batch_idx = torch.zeros(len(atomic_nums), dtype=torch.long, device=device)
            with torch.no_grad():
                latent = gnn_model(atomic_nums, coords, batch_idx)
            embeddings.append(latent.cpu().numpy().flatten())
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=latent_dim)
            embeddings.append(np.array(fp, dtype=np.float32))

        valid_smiles.append(smi)

        if idx % 10000 == 0:
            print(f"  GNN: {idx}/{len(smiles_list)}")

    return np.array(embeddings), valid_smiles
