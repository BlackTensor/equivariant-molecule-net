"""Naive geometry GNN: a GIN that sees 3D coordinates the WRONG way.

The deliberately-broken counterexample for the rotation stress test. It exists
to make the "wobbly vs. flat" contrast real:

  * VanillaGNN (vanilla_gnn.py) ignores ``pos`` entirely. It is flat under
    rotation, but only because it is BLIND to geometry — not because it is
    robust. That is a misleading kind of stability.
  * This model FIXES the blindness in the laziest possible way: it concatenates
    each atom's raw (x, y, z) coordinates onto its node-feature vector and then
    runs an ordinary GIN. Now the network genuinely uses geometry...
  * ...but it uses it in a way that is NOT equivariant. Raw coordinates are not
    rotation-invariant: rotating the molecule changes every (x, y, z) triple,
    which changes the node features, which changes the prediction. So this
    model WOBBLES under rotation. That is the failure the project is built to
    expose, and it cannot be shown with the geometry-blind baseline alone.

Why it is NOT equivariant (the one-line version)
------------------------------------------------
The EGNN earns equivariance by feeding geometry in ONLY through invariant
distances ``||x_i - x_j||``. Here we feed in the raw absolute coordinates
``x_i`` directly. Under a rotation ``x -> R x`` the input features change
(``[h_i, x_i] -> [h_i, R x_i]``), and a plain MLP has no reason to map those
to the same output. Equivariance is broken at the very first layer. (It is also
not even translation-invariant — shifting the molecule moves it too.)

This model is intentionally NEVER TRAINED. Random weights are enough to
demonstrate instability: an untrained non-equivariant function of the
coordinates already produces rotation-dependent outputs. We just need to show
the wobble, not predict the property well.

Architecture (same GIN spine as VanillaGNN, just a wider input):
    x [N, 11] , pos [N, 3]
      -> concat                    -> [N, 14]   (THE non-equivariant step)
      -> Linear encoder            -> [N, hidden]
      -> num_layers x GINConv      -> [N, hidden]
      -> global_add_pool           -> [B, hidden]
      -> MLP head                  -> [B, out_dim]

Run a standalone smoke test that PROVES it wobbles (CPU, no dataset):

    python -m src.models.naive_geo_gnn
"""

from __future__ import annotations

import os
import sys

import torch
from torch import nn
from torch_geometric.nn import GINConv, global_add_pool

# Make `from src.train.config import CONFIG` work whether run as a module
# (`python -m src.models.naive_geo_gnn`) or imported elsewhere.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.train.config import CONFIG  # noqa: E402

# QM9 / molecule_utils node-feature width (see src/data/molecule_utils.py).
NODE_FEATURE_DIM: int = 11
COORD_DIM: int = 3                          # raw (x, y, z) appended per atom
INPUT_DIM: int = NODE_FEATURE_DIM + COORD_DIM  # 14: the geometry-naive input


def _gin_mlp(in_dim: int, hidden_dim: int) -> nn.Sequential:
    """The 2-layer MLP GIN applies after summing neighbor messages.

    Identical to the one in ``vanilla_gnn.py`` — the ONLY difference between the
    two models is whether raw coordinates are concatenated onto the input.
    """
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
    )


class NaiveGeoGNN(nn.Module):
    """GIN that consumes raw xyz as node features — geometry-aware, NOT equivariant.

    Args:
        in_dim:     node-feature width BEFORE coordinates (defaults to QM9's 11).
        hidden_dim: width of every hidden layer (default ``CONFIG.model.hidden_dim``).
        num_layers: number of GINConv message-passing layers.
        out_dim:    number of regression targets (default ``CONFIG.model.out_dim``).
        dropout:    dropout applied in the prediction head.

    forward(x, pos, edge_index, batch) -> [B, out_dim]
        Takes ``pos`` and concatenates it onto ``x``, so the output DOES change
        when the molecule is rotated. That non-equivariance is the whole point.
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

        # Encoder accepts the WIDENED input: original features + 3 coords. This
        # extra +3 is exactly where rotation-sensitivity enters the model.
        self.encoder = nn.Linear(in_dim + COORD_DIM, hidden_dim)

        # Same GIN spine as the vanilla baseline (topological message passing).
        self.convs = nn.ModuleList(
            GINConv(_gin_mlp(hidden_dim, hidden_dim), train_eps=True)
            for _ in range(num_layers)
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict graph-level targets from node features + RAW coordinates.

        Args:
            x:          [N, in_dim] node features.
            pos:        [N, 3] raw coordinates — fed in directly (non-invariant).
            edge_index: [2, E] bond graph (both directions).
            batch:      [N] graph-id per node; ``None`` -> single graph.

        Returns:
            [B, out_dim] predictions. NOT invariant to rotation/translation.
        """
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        # *** The non-equivariant step ***
        # Appending absolute coordinates makes the input depend on the molecule's
        # orientation, so any rotation of `pos` changes `h` and the prediction.
        h = torch.cat([x, pos], dim=-1)   # [N, 14]
        h = self.encoder(h)
        for conv in self.convs:
            h = h + conv(h, edge_index)   # residual, same as VanillaGNN

        h = global_add_pool(h, batch)
        return self.head(h)


def build_naive_geo_gnn(**overrides) -> NaiveGeoGNN:
    """Convenience factory mirroring ``CONFIG`` defaults; pass kwargs to override."""
    return NaiveGeoGNN(**overrides)


def _random_rotation(seed: int = 0) -> torch.Tensor:
    """A random proper rotation matrix (det = +1) via QR (matches egnn/se3 test)."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(3, 3, generator=g)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    if torch.det(q) < 0:        # ensure a rotation, not a reflection
        q[:, 0] = -q[:, 0]
    return q


if __name__ == "__main__":
    # CPU smoke test that PROVES the opposite property from egnn/se3: this model
    # is NOT rotation-invariant. No dataset download needed.
    torch.manual_seed(CONFIG.seed)

    model = build_naive_geo_gnn().eval()
    n_params = sum(p.numel() for p in model.parameters())
    print("NaiveGeoGNN (GIN + raw xyz features, NOT equivariant):")
    print(f"  input_dim={INPUT_DIM} (= {NODE_FEATURE_DIM} feats + {COORD_DIM} coords)"
          f"  hidden_dim={CONFIG.model.hidden_dim}  out_dim={CONFIG.model.out_dim}"
          f"  params={n_params:,}")

    # One fake molecule (6 atoms) so BatchNorm sees >1 node in eval is irrelevant
    # (eval uses running stats); we only need shapes + the wobble.
    n = 6
    x = torch.randn(n, NODE_FEATURE_DIM)
    pos = torch.randn(n, 3) * 1.5
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
         [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]], dtype=torch.long
    )
    batch = torch.zeros(n, dtype=torch.long)

    with torch.no_grad():
        out = model(x, pos, edge_index, batch)
    assert out.shape == (1, CONFIG.model.out_dim), f"bad output shape: {out.shape}"
    print(f"  forward OK: {n} atoms / 1 graph -> output {tuple(out.shape)}")

    # ---- The KEY (anti-)property: rotation CHANGES the prediction ---------- #
    R = _random_rotation(seed=1)
    with torch.no_grad():
        out_rotated = model(x, pos @ R.T, edge_index, batch)

    max_diff = (out - out_rotated).abs().max().item()
    print(f"  rotation sensitivity: max|f(x) - f(Rx)| = {max_diff:.4e}")
    assert max_diff > 1e-3, (
        "prediction barely moved under rotation -- this model is supposed to be "
        "NON-equivariant; check that pos is actually being used"
    )

    print("\nnaive_geo_gnn self-check passed: model uses geometry and is "
          "(correctly) NOT rotation-invariant -- it wobbles, by design.")
