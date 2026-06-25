"""SE(3)-equivariant model built on the ``e3nn`` library (NOT from scratch).

Task 2.3 — the third model in the comparison. Where the EGNN earns its
equivariance from a single clever trick (only ever using invariant distances +
moving along relative vectors), this model earns it from the full machinery of
irreducible representations of SO(3): spherical harmonics of the edge vectors
combined through equivariant tensor products. That machinery is exactly what
``e3nn`` exists to provide, and CLAUDE.md is explicit: **use e3nn, do NOT
reinvent the math.** So this file is deliberately thin — it wires QM9 data into
e3nn's prebuilt gate-points message-passing network and reads out two invariant
scalars.

Why e3nn's gate-points network is SE(3)-equivariant (the short version)
----------------------------------------------------------------------
Every quantity in the network carries an ``Irreps`` type — a label saying how
it transforms under rotation (l=0 scalars are invariant, l=1 vectors rotate,
etc.). The convolution:

  1. takes each edge vector ``r_ij = pos_i - pos_j`` and expands its DIRECTION
     in spherical harmonics Y_l(r_ij / |r_ij|)  -> these transform as l-irreps,
  2. embeds the edge LENGTH |r_ij| (a rotation-invariant scalar) in a radial
     basis to produce the tensor-product weights,
  3. combines neighbor features with those harmonics via tensor products whose
     output irreps are chosen so the result transforms correctly,
  4. applies equivariant nonlinearities (gates) that never mix incompatible
     irreps.

Because length is invariant and direction is handled by spherical harmonics,
rotating the molecule rotates every internal feature *consistently with its
irrep*. We request an output of pure scalars (``2x0e``), so the prediction is
SE(3)-INVARIANT: rotating or translating the molecule cannot change it. (Edge
vectors are built as differences, so translation drops out — exactly like the
EGNN.) That invariance is what the rotation stress test / ESS rewards.

Note on the name: CLAUDE.md's stack table calls this the "SE(3)-Transformer".
The canonical e3nn realization of an SE(3)-equivariant point network is this
tensor-product gate-points convolution (Tensor-Field-Network lineage). We build
on it directly rather than hand-rolling attention, per the "do NOT implement
from scratch" instruction.

This model uses ``pos`` (3D geometry) directly; it builds its OWN radius graph
internally from coordinates, so — unlike the Vanilla GIN and EGNN — it does not
consume the precomputed bond ``edge_index`` / ``edge_attr``. We still accept
those arguments so all three models share one forward signature.

Run a standalone smoke + SE(3)-invariance test (CPU, no dataset needed):

    python -m src.models.se3_transformer
"""

from __future__ import annotations

import os
import sys

import torch
from torch import nn
from e3nn import o3
from e3nn.nn.models.v2106.gate_points_networks import SimpleNetwork

# Make `from src.train.config import CONFIG` work whether run as a module
# (`python -m src.models.se3_transformer`) or imported elsewhere.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.train.config import CONFIG  # noqa: E402

# QM9 / molecule_utils node-feature width (see src/data/molecule_utils.py).
# All 11 features are rotation-invariant scalars -> irreps "11x0e".
NODE_FEATURE_DIM: int = 11

# Geometry / normalization defaults tuned for small QM9 molecules. These are
# physical-ish hyperparameters, all in angstroms / counts, and are free to
# adjust later from a config without affecting equivariance.
DEFAULT_MAX_RADIUS: float = 5.0     # neighbor cutoff for the internal radius graph (Å)
DEFAULT_NUM_NEIGHBORS: float = 12.0  # avg neighbors within the cutoff (a normalizer)
DEFAULT_NUM_NODES: float = 18.0      # avg atoms per QM9 molecule (a normalizer)


class SE3Transformer(nn.Module):
    """SE(3)-equivariant regressor wrapping e3nn's gate-points network.

    Args:
        in_dim:        node-feature width (QM9 = 11 invariant scalars).
        out_dim:       number of regression targets (default ``CONFIG.model.out_dim``).
        max_radius:    cutoff (Å) for the internal radius graph over ``pos``.
        num_neighbors: average neighbor count (normalizer, not a hard limit).
        num_nodes:     average atoms per graph (pooling normalizer).
        mul:           multiplicity of each hidden irrep (≈ "hidden width").
        num_layers:    number of equivariant conv layers (default ``CONFIG.model.num_layers``).
        lmax:          max spherical-harmonic degree used (rotation order).

    forward(x, pos, edge_index, edge_attr, batch) -> [B, out_dim]
        Output irreps are pure scalars (``Nx0e``) so the prediction is
        SE(3)-invariant. ``edge_index`` / ``edge_attr`` are accepted for a
        common signature but unused — geometry comes from ``pos`` directly.
    """

    def __init__(
        self,
        in_dim: int = NODE_FEATURE_DIM,
        out_dim: int | None = None,
        max_radius: float = DEFAULT_MAX_RADIUS,
        num_neighbors: float = DEFAULT_NUM_NEIGHBORS,
        num_nodes: float = DEFAULT_NUM_NODES,
        mul: int = 32,
        num_layers: int | None = None,
        lmax: int = 2,
    ) -> None:
        super().__init__()
        mc = CONFIG.model
        out_dim = mc.out_dim if out_dim is None else out_dim
        num_layers = mc.num_layers if num_layers is None else num_layers

        # Irreps describe how each tensor transforms under rotation:
        #   input  = `in_dim` invariant scalars  -> "11x0e"
        #   output = `out_dim` invariant scalars -> "2x0e"  (SE(3)-invariant)
        self.irreps_in = o3.Irreps(f"{in_dim}x0e")
        self.irreps_out = o3.Irreps(f"{out_dim}x0e")

        # e3nn does ALL the equivariant heavy lifting: spherical harmonics of
        # edge directions, radial-basis edge-length embedding, equivariant
        # tensor-product convolutions, and gated nonlinearities.
        self.net = SimpleNetwork(
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_out,
            max_radius=max_radius,
            num_neighbors=num_neighbors,
            num_nodes=num_nodes,
            mul=mul,
            layers=num_layers,
            lmax=lmax,
            pool_nodes=True,  # graph-level readout (sum over atoms, normalized)
        )

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict graph-level targets SE(3)-equivariantly.

        Args:
            x:          [N, in_dim] invariant node features.
            pos:        [N, 3] coordinates (geometry source; radius graph built here).
            edge_index: accepted for a common signature; UNUSED.
            edge_attr:  accepted for a common signature; UNUSED.
            batch:      [N] graph-id per node; ``None`` -> single graph.

        Returns:
            [B, out_dim] predictions (normalized space — denormalize for units).
        """
        # e3nn's SimpleNetwork consumes a dict and builds the graph from `pos`.
        data = {"pos": pos, "x": x}
        if batch is not None:
            data["batch"] = batch
        out = self.net(data)
        # When pooling a single graph e3nn returns shape [out_dim]; make it
        # [1, out_dim] so the output is always [B, out_dim] like the other models.
        if out.dim() == 1:
            out = out.unsqueeze(0)
        return out


def build_se3_transformer(**overrides) -> SE3Transformer:
    """Convenience factory mirroring ``CONFIG`` defaults; pass kwargs to override."""
    return SE3Transformer(**overrides)


def _random_rotation(seed: int = 0) -> torch.Tensor:
    """A random proper rotation matrix (det = +1) via QR, for the equiv test."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(3, 3, generator=g)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    if torch.det(q) < 0:        # ensure a rotation, not a reflection
        q[:, 0] = -q[:, 0]
    return q


if __name__ == "__main__":
    # CPU smoke + SE(3)-INVARIANCE test. No dataset download needed: build a
    # small fake molecule and check (a) shapes line up and (b) the prediction
    # is genuinely invariant to rotation + translation — the property this
    # whole module exists to provide.
    torch.manual_seed(CONFIG.seed)

    # Keep it small for a fast CPU test (e3nn tensor products are heavier than
    # plain GNN ops). Geometry is what matters here, not size.
    model = build_se3_transformer(mul=16, num_layers=2, lmax=2)
    n_params = sum(p.numel() for p in model.parameters())
    print("SE3Transformer (e3nn gate-points, SE(3)-equivariant):")
    print(f"  irreps_in={model.irreps_in}  irreps_out={model.irreps_out}")
    print(f"  out_dim={CONFIG.model.out_dim}  params={n_params:,}")

    # One fake molecule: 6 atoms spread in 3D (well within the 5 Å cutoff).
    n = 6
    x = torch.randn(n, NODE_FEATURE_DIM)
    pos = torch.randn(n, 3) * 1.5
    batch = torch.zeros(n, dtype=torch.long)

    model.eval()
    with torch.no_grad():
        out = model(x, pos, batch=batch)
    assert out.shape == (1, CONFIG.model.out_dim), f"bad output shape: {out.shape}"
    print(f"  forward OK: {n} atoms / 1 graph -> output {tuple(out.shape)}")

    # ---- SE(3)-INVARIANCE of the prediction ------------------------------ #
    # Transform every coordinate by a random rotation R and a translation t.
    # Output irreps are scalars (0e) and edge vectors are translation-invariant
    # differences, so the prediction must be numerically unchanged.
    R = _random_rotation(seed=1)
    t = torch.tensor([2.0, -3.0, 0.5])
    pos_transformed = pos @ R.T + t            # x -> R x + t  (per row)

    with torch.no_grad():
        out_transformed = model(x, pos_transformed, batch=batch)

    max_diff = (out - out_transformed).abs().max().item()
    print(f"  SE(3) invariance: max|f(x) - f(Rx + t)| = {max_diff:.2e}")
    assert max_diff < 1e-4, "prediction is NOT invariant under rotation+translation!"

    # Two graphs in one batch should pool to [2, out_dim].
    x2 = torch.randn(n, NODE_FEATURE_DIM)
    pos2 = torch.randn(n, 3) * 1.5
    x_cat = torch.cat([x, x2], dim=0)
    pos_cat = torch.cat([pos, pos2], dim=0)
    batch_cat = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])
    with torch.no_grad():
        out_batched = model(x_cat, pos_cat, batch=batch_cat)
    assert out_batched.shape == (2, CONFIG.model.out_dim), f"bad batched shape: {out_batched.shape}"
    print(f"  batched OK: 2 graphs -> output {tuple(out_batched.shape)}")

    print("\nse3_transformer self-check passed (forward shapes + SE(3) invariance verified).")
