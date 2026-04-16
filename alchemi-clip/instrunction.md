The current workflow is built on ../alchemi_ht. Find out the relevant content from alchemi_ht. See if additional packages needs to be installed. In that case, build the packages from the existing alchemi_ht.sif

## Phase 6: Latent-Space Active Learning Search Loop

The Active Learning loop evaluates millions of untested molecules by projecting them into a dense latent space, predicting their properties, and selecting the highest-value targets for GPU physics verification.

### 1a. Train the Contrastive Molecular Encoder
- **Action:** Execute `train_contrastive.py` using your database of successful ALCHEMI relaxations.
- **Logic:** Aligns a 1D Transformer (SMILES) and a 3D GNN (Relaxed Coordinates) in a shared latent space using InfoNCE loss.
- **Output:** Saves `contrastive_1d_encoder.pt` — a 3D-aware text encoder.

### 1b. Extract 3D-Aware Latent Vectors & Train Brain
- **Action:** Pass all ground-truth SMILES through `contrastive_1d_encoder.pt` to get their latent vectors. Train the XGBoost/Random Forest "Brain" on these vectors to predict Energy and Gap.

### 2. The Active Search
- **Action:** Execute `active_search_latent.py --encoder contrastive`.
- **Logic:** Instantly embed millions of unexplored SMILES using the Contrastive 1D Encoder. The Brain evaluates them and selects the top UCB candidates.


### 1. Generate Latent Embeddings & Predict
- **Action:** Execute `active_search_latent.py --encoder [foundation | gnn]`.
- **Mode `foundation`:** Uses a pre-trained Transformer (e.g., ChemBERTa or MoLFormer) to embed 1D SMILES directly into a 768D latent space. Ultra-fast, great for broad exploration.
- **Mode `gnn`:** Uses a custom Graph Neural Network trained on previous ALCHEMI 3D structural data. Converts SMILES to rough 3D graphs via RDKit, then embeds them into a physics-aware latent space. Slower, but highly accurate for local exploitation.
- **Output:** Selects the top 5,000 candidates based on the Upper Confidence Bound (UCB) score and saves `active_learning_batch.csv`.

### 1a. Train the Contrastive Molecular Encoder
- **Action:** Execute `train_contrastive.py` using your database of successful ALCHEMI relaxations.
- **Logic:** Aligns a 1D Transformer (SMILES) and a 3D GNN (Relaxed Coordinates) in a shared latent space using InfoNCE loss.
- **Output:** Saves `contrastive_1d_encoder.pt` — a 3D-aware text encoder.

### 1b. Extract 3D-Aware Latent Vectors & Train Brain
- **Action:** Pass all ground-truth SMILES through `contrastive_1d_encoder.pt` to get their latent vectors. Train the XGBoost/Random Forest "Brain" on these vectors to predict Energy and Gap.

### 2. The Active Search
- **Action:** Execute `active_search_latent.py --encoder contrastive`.
- **Logic:** Instantly embed millions of unexplored SMILES using the Contrastive 1D Encoder. The Brain evaluates them and selects the top UCB candidates.

### 2. The Verification Loop
- **Action:** Submit `active_learning_batch.csv` to the Phase 3 ALCHEMI + GPU4PySCF Slurm array.
- **Update:** Append the ground-truth results to the master dataset. If using the `gnn` mode, trigger a retraining step to update the GNN weights with the new ALCHEMI ground-truth geometries before the next search iteration.


The Latent Search Script (active_search_latent.py)
This script gives the orchestrator the ability to build the latent space dynamically based on the chosen flag.

Python
import argparse
import pandas as pd
import numpy as np
import torch
import joblib
from transformers import AutoModel, AutoTokenizer
from rdkit import Chem
from rdkit.Chem import AllChem

# --- FOUNDATION MODEL (ChemBERTa / MoLFormer) ---
def get_foundation_embeddings(smiles_list):
    """Embeds SMILES using a pre-trained HuggingFace Transformer."""
    print("Loading Foundation Model (ChemBERTa)...")
    tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
    model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
    model.eval()
    
    embeddings = []
    valid_smiles = []
    
    with torch.no_grad():
        for smi in smiles_list:
            try:
                inputs = tokenizer(smi, return_tensors="pt", padding=True, truncation=True)
                outputs = model(**inputs)
                # Take the [CLS] token embedding as the latent representation
                latent_vector = outputs.last_hidden_state[0, 0, :].numpy()
                embeddings.append(latent_vector)
                valid_smiles.append(smi)
            except Exception:
                continue
                
    return np.array(embeddings), valid_smiles

# --- CUSTOM GNN (Placeholder for PyTorch Geometric) ---
def get_gnn_embeddings(smiles_list, gnn_model_path="models/custom_gnn.pt"):
    """Embeds molecules using a custom-trained Graph Neural Network."""
    print("Loading Custom GNN and generating 3D graphs...")
    # NOTE: In a real implementation, you would load your PyG model here
    # model = torch.load(gnn_model_path)
    # model.eval()
    
    embeddings = []
    valid_smiles = []
    
    # Dummy embedding size to simulate GNN output
    LATENT_DIM = 512 
    
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mol = Chem.AddHs(mol)
            # GNNs often require 3D coordinates to build the spatial graph
            success = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            if success == 0:
                # 1. Convert RDKit Mol to PyTorch Geometric Data object
                # 2. graph_data = mol_to_graph(mol)
                # 3. latent_vector = model.encode(graph_data).detach().numpy()
                
                # Simulating the latent vector output for the pipeline
                latent_vector = np.random.rand(LATENT_DIM) 
                embeddings.append(latent_vector)
                valid_smiles.append(smi)
                
    return np.array(embeddings), valid_smiles

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', choices=['foundation', 'gnn'], required=True, 
                        help='Choose the latent space encoder.')
    args = parser.parse_args()

    # 1. Load Unexplored Data
    print("Loading unexplored database chunk...")
    df_unexplored = pd.read_csv("data/unexplored_gdb17_sample.csv")
    smiles_subset = df_unexplored['SMILES'].head(100000).tolist() # Process a chunk
    
    # 2. Map to Latent Space
    if args.encoder == 'foundation':
        X_latent, valid_smiles = get_foundation_embeddings(smiles_subset)
    elif args.encoder == 'gnn':
        X_latent, valid_smiles = get_gnn_embeddings(smiles_subset)
        
    # 3. Load the Surrogate "Brain" (trained on this specific latent space)
    # Ensure you load the correct surrogate model corresponding to the encoder
    surrogate_path = f"models/surrogate_{args.encoder}.pkl"
    brain = joblib.load(surrogate_path)
    
    # 4. Predict and Calculate Acquisition Score (UCB)
    print(f"Evaluating candidates in {args.encoder} latent space...")
    tree_predictions = np.array([tree.predict(X_latent) for tree in brain.estimators_])
    
    mu = np.mean(tree_predictions, axis=0)
    sigma = np.std(tree_predictions, axis=0)
    
    kappa = 2.0 
    ucb_scores = mu + (kappa * sigma)
    
    # 5. Extract Candidates
    results_df = pd.DataFrame({
        'SMILES': valid_smiles,
        'Predicted_Score': mu,
        'Uncertainty': sigma,
        'UCB_Score': ucb_scores
    })
    
    top_candidates = results_df.sort_values(by='UCB_Score', ascending=False).head(5000)
    top_candidates[['SMILES']].to_csv("data/chunks/active_learning_batch.csv", index=False)
    print(f"Selected Top 5000 candidates via {args.encoder} latent space.")

if __name__ == "__main__":
    main()


The PyTorch Implementation (train_contrastive.py)
Here is the mathematical core of the Contrastive model. Your orchestrator agent can use this to build the training loop.

Python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MolecularCLIP(nn.Module):
    def __init__(self, latent_dim=512, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
        # 1. The 1D SMILES Encoder (e.g., a lightweight Transformer)
        # In reality, load a pretrained huggingface model and add a projection head
        self.encoder_1d = self._build_transformer_encoder(latent_dim)
        
        # 2. The 3D Geometry Encoder (e.g., a GNN like SchNet)
        # In reality, load a PyG model that reads ALCHEMI atomic coordinates
        self.encoder_3d = self._build_gnn_encoder(latent_dim)

    def _build_transformer_encoder(self, dim):
        # Placeholder for ChemBERTa + Linear Projection
        return nn.Sequential(nn.Linear(768, dim), nn.LayerNorm(dim))

    def _build_gnn_encoder(self, dim):
        # Placeholder for 3D GNN + Linear Projection
        return nn.Sequential(nn.Linear(256, dim), nn.LayerNorm(dim))

    def forward(self, smiles_batch, coords_batch):
        """
        Takes a batch of SMILES and their corresponding ALCHEMI 3D coordinates.
        """
        # Embed both modalities
        z_1d = self.encoder_1d(smiles_batch) # Shape: [Batch, Latent_Dim]
        z_3d = self.encoder_3d(coords_batch) # Shape: [Batch, Latent_Dim]
        
        # Normalize the vectors (critical for Cosine Similarity)
        z_1d = F.normalize(z_1d, p=2, dim=1)
        z_3d = F.normalize(z_3d, p=2, dim=1)
        
        # Calculate Cosine Similarity Matrix
        # Shape: [Batch, Batch]
        logits = torch.matmul(z_1d, z_3d.T) / self.temperature
        
        # The labels are the diagonal (Molecule A's 1D should match Molecule A's 3D)
        batch_size = z_1d.size(0)
        labels = torch.arange(batch_size, device=z_1d.device)
        
        # Contrastive Loss (InfoNCE)
        # Symmetrical loss: 1D to 3D, and 3D to 1D
        loss_1d_to_3d = F.cross_entropy(logits, labels)
        loss_3d_to_1d = F.cross_entropy(logits.T, labels)
        
        total_loss = (loss_1d_to_3d + loss_3d_to_1d) / 2.0
        return total_loss

# --- How  Orchestrator uses this ---
# 1. Train the model using the loss above on your H100s.
# 2. Save ONLY the 1D encoder for ultra-fast screening.
# torch.save(model.encoder_1d.state_dict(), "models/contrastive_1d_encoder.pt")
