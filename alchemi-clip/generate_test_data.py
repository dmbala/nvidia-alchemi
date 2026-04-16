"""
Generate a small synthetic test dataset that mimics HT screening results.
Produces:
  - data/results/result_0000.csv  (fake HT ground-truth with opt_coords)
  - data/unexplored_test.csv      (small set of unexplored SMILES)
"""

import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# Small diverse SMILES set for testing
TEST_SMILES = [
    "c1ccccc1",            # benzene
    "CC(=O)O",             # acetic acid
    "CCO",                 # ethanol
    "CC=O",                # acetaldehyde
    "c1ccc(O)cc1",         # phenol
    "CC(C)O",              # isopropanol
    "c1ccncc1",            # pyridine
    "CC(=O)N",             # acetamide
    "c1ccc(N)cc1",         # aniline
    "OC(=O)CC(O)=O",      # malonic acid
    "c1ccc2ccccc2c1",      # naphthalene
    "CCN",                 # ethylamine
    "CC#N",                # acetonitrile
    "c1ccc(F)cc1",         # fluorobenzene
    "c1ccc(Cl)cc1",        # chlorobenzene
    "CCOC(=O)C",           # ethyl acetate
    "CC(C)=O",             # acetone
    "c1ccoc1",             # furan
    "c1ccsc1",             # thiophene
    "c1cc[nH]c1",          # pyrrole
    "OC(=O)c1ccccc1",     # benzoic acid
    "CC(O)=O",             # acetic acid (alternate)
    "c1ccc(C)cc1",         # toluene
    "CCN(CC)CC",           # triethylamine
    "CCCC",                # butane
    "CCCCC",               # pentane
    "c1ccc(-c2ccccc2)cc1", # biphenyl
    "Oc1ccccc1O",          # catechol
    "Nc1ccccc1N",          # o-phenylenediamine
    "c1ccc(C=O)cc1",       # benzaldehyde
    "CC(=O)c1ccccc1",      # acetophenone
    "OC(=O)/C=C/c1ccccc1", # cinnamic acid
    "c1cnc2ccccc2n1",      # quinoxaline
    "c1ccc2[nH]ccc2c1",    # indole
    "c1ccc2c(c1)ccn2",     # indoline-like
    "CC1CCCCC1",            # methylcyclohexane
    "C1CCNCC1",             # piperidine
    "C1CCOC1",              # THF
    "c1cc(O)c(O)cc1",      # hydroquinone variant
    "Cc1ccc(O)cc1",         # p-cresol
    "O=Cc1ccco1",           # furfural
    "c1ccc(S)cc1",          # thiophenol
    "CC(=O)OC",             # methyl acetate
    "CCCCCC",               # hexane
    "c1ccccc1O",            # phenol (canonical)
    "c1ccc(CC)cc1",         # ethylbenzene
    "c1ccc(OC)cc1",         # anisole
    "NC(=O)c1ccccc1",       # benzamide
    "Oc1cccc(O)c1",         # resorcinol
    "c1ccnc(N)c1",          # 2-aminopyridine
]


def generate_test_results(smiles_list, output_path):
    """Generate fake HT results with real 3D coordinates from RDKit."""
    np.random.seed(42)
    records = []

    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mol = Chem.AddHs(mol)
        result = AllChem.EmbedMolecule(mol, maxAttempts=5)
        if result != 0:
            continue
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)

        conf = mol.GetConformer()
        positions = conf.GetPositions()
        coords_str = ";".join(f"{x:.4f},{y:.4f},{z:.4f}" for x, y, z in positions)

        # Generate plausible fake energy and gap values
        n_atoms = mol.GetNumAtoms()
        energy = -10.0 * n_atoms + np.random.normal(0, 2.0)
        gap = np.random.uniform(1.0, 8.0)

        records.append({
            "id": idx + 1,
            "smiles": smi,
            "name": f"test_mol_{idx+1}",
            "status": "Success",
            "energy_eV": round(energy, 4),
            "n_atoms": n_atoms,
            "homo_eV": round(-gap / 2 - 3.0 + np.random.normal(0, 0.1), 4),
            "lumo_eV": round(gap / 2 - 3.0 + np.random.normal(0, 0.1), 4),
            "gap_eV": round(gap, 4),
            "opt_coords": coords_str,
        })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} test molecules to {output_path}")
    return df


def generate_unexplored(smiles_list, output_path, n_extra=100):
    """Generate a small CSV of 'unexplored' SMILES for active search testing."""
    # Use the same molecules plus some duplicates/variants
    all_smiles = list(smiles_list)
    # Add simple variants
    extras = [
        "CCCCCCC", "CCCCCCCC", "c1ccc(Br)cc1", "c1ccc(I)cc1",
        "CC(=O)CC", "CCOCC", "c1ccncc1O", "Cc1ccncc1",
        "c1ccc(C(=O)O)cc1", "OC(c1ccccc1)c1ccccc1",
        "c1ccc2c(c1)ccc1ccccc12",  # phenanthrene
        "CCc1ccccc1CC", "c1ccc(NC=O)cc1",
    ]
    all_smiles.extend(extras[:min(n_extra, len(extras))])

    df = pd.DataFrame({"smiles": all_smiles})
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} unexplored SMILES to {output_path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    results_path = os.path.join(script_dir, "data", "results", "result_0000.csv")
    generate_test_results(TEST_SMILES, results_path)

    unexplored_path = os.path.join(script_dir, "data", "unexplored_test.csv")
    generate_unexplored(TEST_SMILES, unexplored_path)
