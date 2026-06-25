"""Phase 2 smoke test — one batch through all three models.

Task 2.4: build a single real batch of molecules and pass it through the
Vanilla GIN, the EGNN, and the e3nn SE(3) network. Confirm every model returns
``[batch_size, out_dim]`` with no errors, where ``out_dim`` matches the number
of prediction targets (``CONFIG.model.out_dim`` == ``len(CONFIG.data.targets)``).

This is an integration check across Phase 2: it uses the SAME data path the app
will use (``molecule_utils.smiles_to_data`` -> PyG ``Batch``), so the 11-dim
node features, 4-dim bond features, and 3D ``pos`` all have to line up with
each model's expectations. No QM9 download and no GPU required.

Run:

    python -m src.models.smoke_test
"""

from __future__ import annotations

import os
import sys

import torch
from torch_geometric.data import Batch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.train.config import CONFIG  # noqa: E402
from src.data.molecule_utils import smiles_to_data  # noqa: E402
from src.models.vanilla_gnn import build_vanilla_gnn  # noqa: E402
from src.models.egnn import build_egnn  # noqa: E402
from src.models.se3_transformer import build_se3_transformer  # noqa: E402

# A few small, valid QM9-element molecules. Kept small so the e3nn tensor
# products stay fast on CPU.
SMILES_BATCH = ["CCO", "CC#N", "C", "CO"]  # ethanol, acetonitrile, methane, methanol


def build_batch(smiles_list: list[str]) -> Batch:
    """Turn SMILES into a single PyG ``Batch`` (shared forward inputs)."""
    data_list = [smiles_to_data(smi) for smi in smiles_list]
    return Batch.from_data_list(data_list)


def main() -> None:
    torch.manual_seed(CONFIG.seed)
    out_dim = CONFIG.model.out_dim
    batch = build_batch(SMILES_BATCH)
    n_graphs = int(batch.batch.max().item()) + 1
    expected = (n_graphs, out_dim)

    print(f"Batch: {len(SMILES_BATCH)} molecules -> {batch.num_nodes} atoms, "
          f"{batch.edge_index.shape[1]} directed edges")
    print(f"  x={tuple(batch.x.shape)}  pos={tuple(batch.pos.shape)}  "
          f"edge_attr={tuple(batch.edge_attr.shape)}")
    print(f"Expecting every model to output {expected} "
          f"(targets={CONFIG.data.targets})\n")

    # Each model has its own forward signature; the Vanilla GIN is geometry-blind
    # (no pos), the other two consume pos. Keep them small/eval for a fast check.
    vanilla = build_vanilla_gnn().eval()
    egnn = build_egnn().eval()
    se3 = build_se3_transformer(mul=16, num_layers=2, lmax=2).eval()

    results: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        results["VanillaGNN"] = vanilla(batch.x, batch.edge_index, batch.batch)
        results["EGNN"] = egnn(
            batch.x, batch.pos, batch.edge_index, batch.edge_attr, batch.batch
        )
        results["SE3Transformer"] = se3(
            batch.x, batch.pos, batch.edge_index, batch.edge_attr, batch.batch
        )

    all_ok = True
    for name, out in results.items():
        ok = tuple(out.shape) == expected and torch.isfinite(out).all()
        all_ok &= ok
        print(f"  {name:16s} -> {tuple(out.shape)}  {'OK' if ok else 'FAIL'}")
        assert tuple(out.shape) == expected, f"{name} output shape {tuple(out.shape)} != {expected}"
        assert torch.isfinite(out).all(), f"{name} produced non-finite values"

    assert all_ok
    print("\nPhase 2 smoke test passed: all three models output "
          f"{expected} with finite values.")


if __name__ == "__main__":
    main()
