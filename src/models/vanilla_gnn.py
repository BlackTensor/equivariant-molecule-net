"""Vanilla GNN baseline: a GIN that uses node features only.

Task 2.1 — the *intentionally weak* model. This is the "bad example" the whole
project is built to expose: a graph neural network that knows WHICH atoms are
bonded to which (topology) and WHAT each atom is (node features), but is
completely **blind to 3D geometry**.

Why blind on purpose
--------------------
It reads only ``data.x`` (the [N, 11] QM9 atom features: element one-hot +
atomic number, aromaticity, hybridization, bonded-H count) and the bond graph
``data.edge_index``. It NEVER touches ``data.pos`` (the 3D coordinates). So it
cannot tell apart two conformers, and — crucially for our rotation stress test
(Phase 4) — rotating a molecule does not change ``x`` or ``edge_index`` at all,
so this model's prediction is trivially constant under rotation... *for a fixed
input graph*.

The interesting failure shows up the other way around: because it ignores
geometry entirely, it has no access to the very information (bond angles,
interatomic distances, 3D shape) that actually determines properties like the
dipole moment. The EGNN and SE(3)-Transformer do use ``pos`` equivariantly and
should therefore be both more accurate AND provably stable. This baseline is
the foil that makes that contrast measurable.

GIN (Graph Isomorphism Network, Xu et al. 2019) is chosen because it is a
strong, standard, purely topological message-passing model — so when it loses,
it loses on the merits of ignoring geometry, not because it is a toy.

Architecture (sizes shared via ``CONFIG.model`` for a fair comparison):
    x [N, 11]
      -> Linear encoder            -> [N, hidden]
      -> num_layers x GINConv      -> [N, hidden]   (topology only)
      -> global_add_pool           -> [B, hidden]   (graph-level readout)
      -> MLP head                  -> [B, out_dim]  (the predicted targets)

Run a standalone smoke test (CPU, no dataset download needed):

    python -m src.models.vanilla_gnn
"""

from __future__ import annotations

import os
import sys

import torch
from torch import nn
from torch_geometric.nn import GINConv, global_add_pool

# Make `from src.train.config import CONFIG` work whether run as a module
# (`python -m src.models.vanilla_gnn`) or imported elsewhere (mirrors the
# data-pipeline modules).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.train.config import CONFIG  # noqa: E402

# QM9 / molecule_utils node-feature width (see src/data/molecule_utils.py).
NODE_FEATURE_DIM: int = 11


def _gin_mlp(in_dim: int, hidden_dim: int) -> nn.Sequential:
    """The 2-layer MLP that GIN applies after summing neighbor messages.

    GINConv aggregates neighbors by SUM and then transforms with this MLP; a
    2-layer MLP gives the conv enough capacity to be an injective (and thus
    maximally discriminative) aggregator, which is GIN's whole point.
    """
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
    )


class VanillaGNN(nn.Module):
    """Topology-only GIN regressor (the geometry-blind baseline).

    Args:
        in_dim:     node-feature width (defaults to QM9's 11).
        hidden_dim: width of every hidden layer (default ``CONFIG.model.hidden_dim``).
        num_layers: number of GINConv message-passing layers.
        out_dim:    number of regression targets (default ``CONFIG.model.out_dim``).
        dropout:    dropout applied in the prediction head.

    forward(x, edge_index, batch) -> [B, out_dim]
        Deliberately takes NO ``pos`` argument — there is no way to feed this
        model 3D coordinates, which is the point.
    """

    def __init__(
        self,
        in_dim: int = NODE_FEATURE_DIM,
        hidden_dim: int | None = None,
        num_layers: int | None = None,
        out_dim: int | None = None,
        dropout: float | None = None,
    ) -> None:
        super().__init__()
        mc = CONFIG.model
        hidden_dim = mc.hidden_dim if hidden_dim is None else hidden_dim
        num_layers = mc.num_layers if num_layers is None else num_layers
        out_dim = mc.out_dim if out_dim is None else out_dim
        dropout = mc.dropout if dropout is None else dropout

        # Project raw node features up to the hidden width before message passing.
        self.encoder = nn.Linear(in_dim, hidden_dim)

        # Stack of GIN convolutions. Each one mixes a node with its bonded
        # neighbors (edge_index) only — no coordinates ever enter here.
        self.convs = nn.ModuleList(
            GINConv(_gin_mlp(hidden_dim, hidden_dim), train_eps=True)
            for _ in range(num_layers)
        )

        # Graph-level prediction head applied after pooling node embeddings.
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict graph-level targets from node features + bond topology.

        Args:
            x:          [N, in_dim] node features.
            edge_index: [2, E] bond graph (both directions).
            batch:      [N] graph-id per node for batched pooling; if ``None``,
                        all nodes are treated as one graph.

        Returns:
            [B, out_dim] predictions (normalized space — denormalize for units).
        """
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        h = self.encoder(x)
        for conv in self.convs:
            # Residual connection keeps gradients healthy across the stack.
            h = h + conv(h, edge_index)

        # SUM pooling matches GIN's injective-aggregation philosophy at the
        # graph level: collapse [N, hidden] node embeddings -> [B, hidden].
        h = global_add_pool(h, batch)
        return self.head(h)


def build_vanilla_gnn(**overrides) -> VanillaGNN:
    """Convenience factory mirroring ``CONFIG`` defaults; pass kwargs to override."""
    return VanillaGNN(**overrides)


if __name__ == "__main__":
    # CPU smoke test: build the model and push two tiny fake molecules through
    # it. No dataset download required — we just need the shapes to line up and
    # to confirm the model genuinely ignores coordinates.
    torch.manual_seed(CONFIG.seed)

    model = build_vanilla_gnn()
    n_params = sum(p.numel() for p in model.parameters())
    print("VanillaGNN (GIN baseline):")
    print(f"  hidden_dim={CONFIG.model.hidden_dim}  num_layers={CONFIG.model.num_layers}"
          f"  out_dim={CONFIG.model.out_dim}  params={n_params:,}")

    # Fake batch: molecule A has 3 atoms, molecule B has 2 atoms (5 nodes total).
    x = torch.randn(5, NODE_FEATURE_DIM)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 3, 4],
         [1, 0, 2, 1, 4, 3]], dtype=torch.long
    )
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)

    model.eval()
    with torch.no_grad():
        out = model(x, edge_index, batch)
    assert out.shape == (2, CONFIG.model.out_dim), f"bad output shape: {out.shape}"
    print(f"  forward OK: input 5 nodes / 2 graphs -> output {tuple(out.shape)}")

    # Single-graph path (batch=None) should also work and give [1, out_dim].
    with torch.no_grad():
        out1 = model(x[:3], edge_index[:, :4])
    assert out1.shape == (1, CONFIG.model.out_dim)

    # KEY PROPERTY: the model is geometry-blind. Its forward signature has no
    # `pos` argument, so 3D coordinates (and any rotation of them) cannot affect
    # the output. We demonstrate that the output depends ONLY on (x, edge_index)
    # by re-running with identical inputs and getting identical results.
    with torch.no_grad():
        out_again = model(x, edge_index, batch)
    assert torch.allclose(out, out_again), "model should be deterministic on fixed inputs"
    print("  geometry-blind: forward() takes no `pos`; output is a function of (x, edge_index) only")

    print("\nvanilla_gnn self-check passed.")
