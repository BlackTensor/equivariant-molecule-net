"""EGNN — E(n) Equivariant Graph Neural Network (Satorras, Hoogeboom & Welling, 2021).

Task 2.2 — the CORE of the project: a graph network that DOES use 3D geometry,
and does so in a provably E(3)-equivariant way. This is the model that should
stay rock-steady under rotation (ESS ≈ 1.0) where the Vanilla GIN baseline,
having no notion of geometry at all, cannot benefit from coordinates.

Paper: "E(n) Equivariant Graph Neural Networks", ICML 2021.
        https://arxiv.org/abs/2102.09844   (~150 lines of real math; no e3nn.)

----------------------------------------------------------------------------
The four EGNN equations (one message-passing layer, for every edge i<-j)
----------------------------------------------------------------------------
Let h_i be the (invariant) feature vector of node i and x_i its 3D position.

  (1) edge message      m_ij = phi_e( h_i, h_j, ||x_i - x_j||^2, a_ij )
  (2) coordinate update x_i' = x_i + C * sum_j (x_i - x_j) * phi_x(m_ij)
  (3) aggregate         m_i  = sum_j  m_ij
  (4) node update       h_i' = phi_h( h_i, m_i )

phi_e, phi_x, phi_h are plain MLPs. a_ij are optional edge attributes (we feed
the QM9 bond-type one-hot). C is a normalizer (we use 1/(#neighbors)).

THE KEY IDEA, in one line:
  positions enter the network ONLY through the squared distance ||x_i - x_j||^2.

Distance is invariant to rotation, translation, AND reflection — i.e. to the
whole Euclidean group E(3). So every message m_ij (and therefore every node
feature h_i) is an E(3)-INVARIANT scalar. The coordinate update (2) moves atoms
along the relative vectors (x_i - x_j) scaled by those invariant weights, which
makes the positions transform EQUIVARIANTLY. See the long note on
``EGNNLayer._coord_update`` below for the full why-it-works argument.

What this means for us:
  * We read out the graph-level property from the final INVARIANT features h.
    => the predicted HOMO-LUMO gap / dipole magnitude does NOT change when the
       molecule is rotated, translated, or reflected. That is exactly the
       robustness the Equivariance Stability Score (Phase 4) is designed to
       reward.

Architecture:
    (h0 from x[N,11]) , pos[N,3]
      -> encoder Linear: x -> h [N, hidden]
      -> num_layers x EGNNLayer  (updates h invariantly, pos equivariantly)
      -> global_add_pool over h  -> [B, hidden]
      -> MLP head                -> [B, out_dim]

Run a standalone smoke + equivariance test (CPU, no dataset needed):

    python -m src.models.egnn
"""

from __future__ import annotations

import os
import sys

import torch
from torch import nn
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import degree

# Make `from src.train.config import CONFIG` work whether run as a module
# (`python -m src.models.egnn`) or imported elsewhere (mirrors the other modules).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.train.config import CONFIG  # noqa: E402

# QM9 / molecule_utils widths (see src/data/molecule_utils.py).
NODE_FEATURE_DIM: int = 11   # data.x
EDGE_FEATURE_DIM: int = 4    # data.edge_attr (bond-type one-hot)


class EGNNLayer(nn.Module):
    """One E(n)-equivariant message-passing layer (eqs. 1-4 above).

    Updates node features ``h`` invariantly and node positions ``pos``
    equivariantly. Stacking several of these is the whole EGNN.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int = EDGE_FEATURE_DIM,
        update_coords: bool = True,
    ) -> None:
        super().__init__()
        self.update_coords = update_coords

        # phi_e : edge MLP. Input = [h_i, h_j, dist^2, a_ij].
        #   The "+1" is the single scalar squared-distance feature — this is the
        #   ONLY way geometry enters, and being a distance it is E(3)-invariant.
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # phi_x : maps an edge message -> a single SCALAR weight for the
        # coordinate update. Final layer has no bias and is zero-initialized so
        # the model starts as (near) the identity on coordinates — a standard
        # EGNN stabilization trick.
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        nn.init.zeros_(self.coord_mlp[-1].weight)

        # phi_h : node MLP. Input = [h_i, aggregated_message_i].
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _coord_update(
        self,
        pos: torch.Tensor,
        rel: torch.Tensor,
        row: torch.Tensor,
        edge_weight: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Equivariant coordinate update — eq. (2). Read the note carefully.

        ``rel = pos[row] - pos[col]`` is the relative vector along each edge,
        ``edge_weight = phi_x(m_ij)`` is a learned INVARIANT scalar per edge.

            x_i' = x_i + C * sum_{j in N(i)} (x_i - x_j) * phi_x(m_ij)

        ------------------------------------------------------------------
        WHY THIS IS E(3)-EQUIVARIANT (rotation + translation + reflection)
        ------------------------------------------------------------------
        Apply any rigid transform to every atom:   x  ->  R x + t,
        where R is an orthogonal matrix (rotation OR reflection, R^T R = I)
        and t is a translation.

          * Relative vectors lose the translation and rotate with R:
                (x_i + t-stuff) ...  (x_i - x_j) -> R x_i - R x_j = R (x_i - x_j).
            The +t cancels in the difference, so translation has NO effect here.

          * The weights phi_x(m_ij) depend on positions only via ||x_i - x_j||^2,
            and ||R(x_i - x_j)|| = ||x_i - x_j|| because R is orthogonal. So the
            messages — and hence every weight — are UNCHANGED by the transform.
            They are invariant scalars.

          * Therefore each term transforms as
                (x_i - x_j) * w_ij  ->  R (x_i - x_j) * w_ij,
            and summing linearly:
                x_i' = x_i + C * sum_j (x_i - x_j) w_ij
                     ->  R x_i + t + C * sum_j R (x_i - x_j) w_ij
                     =  R ( x_i + C * sum_j (x_i - x_j) w_ij ) + t
                     =  R x_i' + t.

        The updated coordinates transform by exactly the SAME (R, t) as the
        input. That is the definition of equivariance. No spherical harmonics,
        no e3nn — equivariance falls out of (a) using only invariant distances
        as geometric input and (b) only ever moving atoms along relative
        vectors. The node features h, seen only through invariant messages,
        stay invariant throughout — so a readout from h is E(3)-INVARIANT,
        which is what we want for predicting scalar molecular properties.
        ------------------------------------------------------------------
        """
        # Per-edge contribution: relative vector scaled by its invariant weight.
        trans = rel * edge_weight                       # [E, 3]
        # Sum contributions into the source node i (index_add over `row`).
        agg = pos.new_zeros((num_nodes, 3))
        agg.index_add_(0, row, trans)                   # sum_j (x_i - x_j) w_ij
        # C = 1 / deg(i): mean over neighbors keeps the update scale sane
        # regardless of how many bonds an atom has. (deg is invariant too.)
        deg = degree(row, num_nodes=num_nodes, dtype=pos.dtype).clamp(min=1).unsqueeze(-1)
        return pos + agg / deg

    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one layer. Returns updated (h, pos).

        Args:
            h:          [N, hidden] invariant node features.
            pos:        [N, 3] coordinates.
            edge_index: [2, E] (row = i = source, col = j = neighbor).
            edge_attr:  [E, edge_dim] bond-type one-hot (a_ij).
        """
        row, col = edge_index  # row -> i, col -> j
        num_nodes = h.size(0)

        # --- eq. (1): edge messages ---------------------------------------- #
        rel = pos[row] - pos[col]                       # [E, 3] relative vectors
        # Squared distance is the ONLY geometric input -> E(3)-invariant scalar.
        dist2 = (rel ** 2).sum(dim=-1, keepdim=True)    # [E, 1]
        edge_in = torch.cat([h[row], h[col], dist2, edge_attr], dim=-1)
        m_ij = self.edge_mlp(edge_in)                   # [E, hidden], invariant

        # --- eq. (2): equivariant coordinate update ------------------------ #
        if self.update_coords:
            edge_weight = self.coord_mlp(m_ij)          # [E, 1] invariant weight
            pos = self._coord_update(pos, rel, row, edge_weight, num_nodes)

        # --- eq. (3): aggregate messages into each node -------------------- #
        m_i = h.new_zeros((num_nodes, m_ij.size(-1)))
        m_i.index_add_(0, row, m_ij)                    # sum_j m_ij

        # --- eq. (4): node feature update (residual) ----------------------- #
        h = h + self.node_mlp(torch.cat([h, m_i], dim=-1))
        return h, pos


class EGNN(nn.Module):
    """EGNN regressor: invariant readout from E(3)-equivariant message passing.

    Args:
        in_dim:        node-feature width (defaults to QM9's 11).
        edge_dim:      edge-feature width (defaults to QM9's 4 bond types).
        hidden_dim:    hidden width (default ``CONFIG.model.hidden_dim``).
        num_layers:    number of EGNN layers (default ``CONFIG.model.num_layers``).
        out_dim:       number of regression targets (default ``CONFIG.model.out_dim``).
        dropout:       dropout in the prediction head.
        update_coords: if True (default) atoms move equivariantly between layers;
                       set False to keep coordinates fixed (pure distance model).

    forward(x, pos, edge_index, edge_attr, batch) -> [B, out_dim]
        The output is read from the final INVARIANT features, so it does not
        change under rotation / translation / reflection of ``pos``.
    """

    def __init__(
        self,
        in_dim: int = NODE_FEATURE_DIM,
        edge_dim: int = EDGE_FEATURE_DIM,
        hidden_dim: int | None = None,
        num_layers: int | None = None,
        out_dim: int | None = None,
        dropout: float | None = None,
        update_coords: bool = True,
    ) -> None:
        super().__init__()
        mc = CONFIG.model
        hidden_dim = mc.hidden_dim if hidden_dim is None else hidden_dim
        num_layers = mc.num_layers if num_layers is None else num_layers
        out_dim = mc.out_dim if out_dim is None else out_dim
        dropout = mc.dropout if dropout is None else dropout

        # Embed raw node features -> hidden width. This sees NO coordinates, so
        # the initial features are trivially invariant.
        self.encoder = nn.Linear(in_dim, hidden_dim)

        self.layers = nn.ModuleList(
            EGNNLayer(hidden_dim, edge_dim=edge_dim, update_coords=update_coords)
            for _ in range(num_layers)
        )

        # Graph-level head applied to pooled INVARIANT node features.
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict graph-level targets equivariantly.

        Args:
            x:          [N, in_dim] node features.
            pos:        [N, 3] coordinates (used ONLY via invariant distances).
            edge_index: [2, E] bond graph (both directions).
            edge_attr:  [E, edge_dim] bond-type one-hot.
            batch:      [N] graph-id per node; ``None`` -> single graph.

        Returns:
            [B, out_dim] predictions (normalized space — denormalize for units).
        """
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        h = self.encoder(x)
        for layer in self.layers:
            h, pos = layer(h, pos, edge_index, edge_attr)

        # Read out from h only -> the prediction is E(3)-invariant by design.
        h = global_add_pool(h, batch)
        return self.head(h)


def build_egnn(**overrides) -> EGNN:
    """Convenience factory mirroring ``CONFIG`` defaults; pass kwargs to override."""
    return EGNN(**overrides)


def _random_rotation(seed: int = 0) -> torch.Tensor:
    """A random proper rotation matrix (det = +1) via QR, for the equiv test."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(3, 3, generator=g)
    q, r = torch.linalg.qr(a)
    # Fix signs so the diagonal of R is positive -> q is a uniform rotation.
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    if torch.det(q) < 0:        # ensure a rotation, not a reflection
        q[:, 0] = -q[:, 0]
    return q


if __name__ == "__main__":
    # CPU smoke + EQUIVARIANCE test. No dataset download needed: we build a tiny
    # fake batch and check (a) shapes line up and (b) the model is genuinely
    # E(3)-invariant in its readout — the property this whole module exists for.
    torch.manual_seed(CONFIG.seed)

    model = build_egnn()
    n_params = sum(p.numel() for p in model.parameters())
    print("EGNN (E(3)-equivariant, from scratch):")
    print(f"  hidden_dim={CONFIG.model.hidden_dim}  num_layers={CONFIG.model.num_layers}"
          f"  out_dim={CONFIG.model.out_dim}  params={n_params:,}")

    # Fake batch: molecule A (3 atoms), molecule B (2 atoms) -> 5 nodes.
    x = torch.randn(5, NODE_FEATURE_DIM)
    pos = torch.randn(5, 3)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 3, 4],
         [1, 0, 2, 1, 4, 3]], dtype=torch.long
    )
    edge_attr = torch.tensor(
        [[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0],
         [0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=torch.float
    )
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)

    model.eval()
    with torch.no_grad():
        out = model(x, pos, edge_index, edge_attr, batch)
    assert out.shape == (2, CONFIG.model.out_dim), f"bad output shape: {out.shape}"
    print(f"  forward OK: 5 nodes / 2 graphs -> output {tuple(out.shape)}")

    # ---- E(3)-INVARIANCE of the prediction ------------------------------- #
    # Transform every coordinate by a random rotation R and a translation t.
    # Because positions enter only through invariant distances and the readout
    # is from invariant features, the output must be (numerically) unchanged.
    R = _random_rotation(seed=1)
    t = torch.tensor([3.0, -1.5, 0.7])
    pos_transformed = pos @ R.T + t            # x -> R x + t  (per row)

    with torch.no_grad():
        out_transformed = model(x, pos_transformed, edge_index, edge_attr, batch)

    max_diff = (out - out_transformed).abs().max().item()
    print(f"  E(3) invariance: max|f(x) - f(Rx + t)| = {max_diff:.2e}")
    assert max_diff < 1e-4, "prediction is NOT invariant under rotation+translation!"

    # Reflection (improper transform) should also leave the readout unchanged,
    # since distances are reflection-invariant too -> full E(3), not just SE(3).
    reflect = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))
    with torch.no_grad():
        out_reflected = model(x, pos @ reflect.T, edge_index, edge_attr, batch)
    refl_diff = (out - out_reflected).abs().max().item()
    print(f"  reflection invariance: max diff = {refl_diff:.2e}")
    assert refl_diff < 1e-4, "prediction is NOT reflection-invariant!"

    print("\negnn self-check passed (forward shapes + E(3) invariance verified).")
