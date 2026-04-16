import torch
from periodictable import elements
from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.demo import DemoModel, DemoModelWrapper
from nvalchemi.dynamics import DemoDynamics, ConvergenceHook
from nvalchemi.dynamics.hooks import NaNDetectorHook

# Workaround: Batch.batch_idx is int32 but torch scatter ops require int64.
# Patch the batch_idx property to always return int64.
_orig_batch_idx = Batch.batch_idx.fget

@property
def _int64_batch_idx(self):
    idx = _orig_batch_idx(self)
    return idx.to(torch.long) if idx is not None else idx

Batch.batch_idx = _int64_batch_idx


def get_mass(z):
    """Get atomic mass from atomic number."""
    return elements[z].mass


# Create two small molecules with all required fields for dynamics
atomic_nums_a = [6, 6, 1, 1]
atomic_nums_b = [8, 1, 1]

mol_a = AtomicData(
    positions=torch.randn(4, 3),
    atomic_numbers=torch.tensor(atomic_nums_a, dtype=torch.long),
    atomic_masses=torch.tensor([get_mass(z) for z in atomic_nums_a], dtype=torch.float32),
    velocities=torch.zeros(4, 3),
    forces=torch.zeros(4, 3),
)
mol_b = AtomicData(
    positions=torch.randn(3, 3),
    atomic_numbers=torch.tensor(atomic_nums_b, dtype=torch.long),
    atomic_masses=torch.tensor([get_mass(z) for z in atomic_nums_b], dtype=torch.float32),
    velocities=torch.zeros(3, 3),
    forces=torch.zeros(3, 3),
)

# Batch, move to GPU, wrap model
batch = Batch.from_data_list([mol_a, mol_b]).cuda()
model = DemoModelWrapper(model=DemoModel()).cuda()

# Run geometry optimization with convergence detection
dynamics = DemoDynamics(
    model=model,
    n_steps=1_000,
    dt=0.5,
    convergence_hook=ConvergenceHook.from_fmax(0.05),
    hooks=[NaNDetectorHook()],
)

print(f"Batch: {batch.batch_size} molecules, {batch.num_nodes} atoms")
print("Starting dynamics run...")
with dynamics:
    result = dynamics.run(batch)

print("Dynamics run complete.")
print(f"Final positions shape: {result.positions.shape}")
print(f"Final forces shape: {result.forces.shape}")
