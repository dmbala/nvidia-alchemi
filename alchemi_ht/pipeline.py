"""
HT Screening Pipeline: SMILES -> 3D -> AIMNet2 Relaxation -> HOMO-LUMO (GPU4PySCF)

Processes a single chunk CSV. Intended to be run as a Slurm array task:
    python pipeline.py --chunk data/chunks/chunk_0000.csv --output data/results/result_0000.csv
"""

# Runtime shim: the container's alchemi_ht.def installed a dftd3.py shim that
# re-exports the low-level warp kernel. AIMNet 0.3.x expects the high-level
# torch wrapper (which accepts `cell`, `neighbor_matrix`, etc.). Until the
# container is rebuilt from the fixed .def, monkey-patch sys.modules BEFORE
# aimnet is imported.
def _install_dftd3_shim() -> None:
    import sys
    import types
    try:
        from nvalchemiops.torch.interactions.dispersion import dftd3 as _dftd3_torch
    except ImportError:
        return
    shim = types.ModuleType("nvalchemiops.interactions.dispersion.dftd3")
    shim.dftd3 = _dftd3_torch
    sys.modules["nvalchemiops.interactions.dispersion.dftd3"] = shim


_install_dftd3_shim()

import argparse
import sys
import traceback

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from ase import Atoms
from ase.optimize import LBFGS

from aimnet.calculators.aimnet2ase import AIMNet2ASE

from pyscf import gto, dft


# ---------------------------------------------------------------------------
# Step 1: SMILES -> 3D coordinates via RDKit
# ---------------------------------------------------------------------------

def embed_3d(smiles: str, max_attempts: int = 10, random_seed: int = 42) -> Atoms | None:
    """Convert SMILES to an ASE Atoms object with 3D coordinates.

    Uses the kwargs overload of EmbedMolecule (the ETKDGv3 params object in this
    RDKit version rejects `maxAttempts` as a settable attribute). The kwargs
    below reproduce ETKDGv3 defaults.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    result = AllChem.EmbedMolecule(
        mol,
        maxAttempts=max_attempts,
        randomSeed=random_seed,
        useRandomCoords=True,
        useExpTorsionAnglePrefs=True,
        useBasicKnowledge=True,
        useSmallRingTorsions=True,
    )
    if result != 0:
        return None
    AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    conf = mol.GetConformer()
    positions = np.array(conf.GetPositions())
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    return Atoms(symbols=symbols, positions=positions)


# ---------------------------------------------------------------------------
# Step 2: Geometry relaxation via AIMNet2
# ---------------------------------------------------------------------------

def relax_aimnet2(atoms: Atoms, calc: AIMNet2ASE, fmax: float = 0.05, steps: int = 200) -> tuple[Atoms, float]:
    """Relax geometry with AIMNet2 and return (relaxed_atoms, energy)."""
    atoms.calc = calc
    opt = LBFGS(atoms, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    energy = atoms.get_potential_energy()
    return atoms, energy


# ---------------------------------------------------------------------------
# Step 3: HOMO-LUMO gap via GPU4PySCF (DFT)
# ---------------------------------------------------------------------------

def compute_homo_lumo(atoms: Atoms, basis: str = "def2-svp") -> tuple[float, float, float]:
    """Run DFT with GPU4PySCF and return (homo, lumo, gap) in eV."""
    atom_str = "; ".join(
        f"{s} {x:.6f} {y:.6f} {z:.6f}"
        for s, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.get_positions())
    )
    mol = gto.M(atom=atom_str, basis=basis, verbose=0)
    mf = dft.RKS(mol)
    mf.xc = "b3lyp"
    try:
        mf = mf.to_gpu()
    except Exception:
        pass  # fall back to CPU PySCF if GPU not available
    mf.kernel()

    mo_occ = mf.mo_occ
    mo_energy = mf.mo_energy
    occupied = mo_energy[mo_occ > 0]
    virtual = mo_energy[mo_occ == 0]
    if len(occupied) == 0 or len(virtual) == 0:
        return float("nan"), float("nan"), float("nan")

    ha_to_ev = 27.2114
    homo = float(occupied[-1]) * ha_to_ev
    lumo = float(virtual[0]) * ha_to_ev
    gap = lumo - homo
    return homo, lumo, gap


# ---------------------------------------------------------------------------
# Main: process one chunk
# ---------------------------------------------------------------------------

def process_chunk(chunk_path: str, output_path: str, basis: str = "def2-svp"):
    df = pd.read_csv(chunk_path)
    calc = AIMNet2ASE("aimnet2")

    results = []
    for _, row in df.iterrows():
        mol_id = row["id"]
        smiles = row["smiles"]
        name = row.get("name", "")
        record = {"id": mol_id, "smiles": smiles, "name": name, "status": "Pending"}

        try:
            # Step 1: 3D embedding
            atoms = embed_3d(smiles)
            if atoms is None:
                record["status"] = "Infeasible Geometry"
                results.append(record)
                continue

            # Step 2: AIMNet2 relaxation
            atoms, energy = relax_aimnet2(atoms, calc)
            record["energy_eV"] = energy
            record["n_atoms"] = len(atoms)

            # Step 3: HOMO-LUMO
            homo, lumo, gap = compute_homo_lumo(atoms, basis=basis)
            record["homo_eV"] = homo
            record["lumo_eV"] = lumo
            record["gap_eV"] = gap
            record["status"] = "Success"

            # Store optimized coordinates as a compact string
            coords = atoms.get_positions()
            record["opt_coords"] = ";".join(f"{x:.4f},{y:.4f},{z:.4f}" for x, y, z in coords)

        except Exception as e:
            record["status"] = f"Error: {e}"
            traceback.print_exc()

        results.append(record)
        print(f"  [{record['status']}] {mol_id}: {smiles}")

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False)
    print(f"\nWrote {len(out_df)} results to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="HT Screening Pipeline")
    parser.add_argument("--chunk", required=True, help="Path to input chunk CSV")
    parser.add_argument("--output", required=True, help="Path to output result CSV")
    parser.add_argument("--basis", default="def2-svp", help="DFT basis set (default: def2-svp)")
    args = parser.parse_args()
    process_chunk(args.chunk, args.output, basis=args.basis)


if __name__ == "__main__":
    main()
