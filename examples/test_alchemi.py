import torch
from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.demo import DemoModel, DemoModelWrapper
import nvalchemiops

# Create simple atomic structures
mol_a = AtomicData(
    positions=torch.randn(4, 3),
    atomic_numbers=torch.tensor([6, 6, 1, 1], dtype=torch.long),
)
mol_b = AtomicData(
    positions=torch.randn(3, 3),
    atomic_numbers=torch.tensor([8, 1, 1], dtype=torch.long),
)

# Batch structures
batch = Batch.from_data_list([mol_a, mol_b])

# Initialize model components
model = DemoModelWrapper(model=DemoModel())

print("Setup successful: ALCHEMI components initialized.")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Batch size: {batch.batch_size}, total atoms: {batch.num_nodes}")
print(f"nvalchemiops available: {nvalchemiops.__name__}")
