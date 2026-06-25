"""Phase 5 explainability — per-atom importance via Integrated Gradients (IG).

Task 5.1: given a molecule (SMILES) and a model, return a per-ATOM importance
score saying how much each atom contributed to the predicted property. The
Phase 7 frontend (task 7.5) overlays these as a heatmap on the 3D molecule.

Why Integrated Gradients (and not a one-shot gradient)?
-------------------------------------------------------
A plain input gradient ``d f / d x`` is noisy and saturates: once a feature is
"large enough" its local gradient can be ~0 even though the feature clearly
mattered. Integrated Gradients (Sundararajan et al. 2017,
https://arxiv.org/abs/1703.01365) fixes this by averaging the gradient along a
straight path from a featureless BASELINE x0 to the real input x, then scaling
by the input displacement:

    IG_k(x) = (x_k - x0_k) * (1/m) * sum_{s=1..m} d f / d x_k  at  x0 + (s-0.5)/m * (x - x0)

We attribute w.r.t. the node-feature matrix ``x`` ([N, 11]) because it has
exactly one row per atom, so summing each row's attributions gives a single
scalar per atom — uniform across ALL four models (vanilla / naive_geo / egnn /
se3), none of which would otherwise share an explanation surface. The baseline
is the all-zeros feature vector (the natural "no atom information" reference).

IG satisfies COMPLETENESS: the attributions sum to f(x) - f(x0). We use that as
a built-in correctness check (the self-test asserts the residual is under ~1% of
the prediction's span from the baseline, given enough path steps).

A nice consequence for the equivariant models: EGNN/SE(3) predictions are
rotation-invariant and ``x`` does not change under rotation, so their per-atom
importance is itself rotation-invariant (verified in the self-test).

Run a standalone self-check (loads trained weights in ``weights/``, CPU):

    python -m src.eval.atom_importance
"""

from __future__ import annotations

import os
import sys

import torch
from torch_geometric.data import Data

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.train.config import CONFIG  # noqa: E402
from src.data.molecule_utils import smiles_to_data  # noqa: E402
from src.data.qm9_loader import TargetNormalizer  # noqa: E402
from src.eval.rotation_test import load_model  # noqa: E402

# Atomic number -> element symbol, for QM9's element set (H, C, N, O, F).
_ATOMIC_SYMBOL: dict[int, str] = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}

# Default number of IG path steps. The Vanilla GIN's ReLU/BatchNorm kinks make
# the Riemann sum converge slowly, so we use a generous step count to keep the
# completeness error well under 1% for every model; still fast on CPU for the
# small molecules this project handles. (EGNN/SE(3) converge by ~64 steps.)
DEFAULT_IG_STEPS: int = 256


def _model_forward(
    model_name: str, model: torch.nn.Module, x: torch.Tensor, data: Data
) -> torch.Tensor:
    """Run ``model`` on node features ``x`` (the IG variable) + ``data``'s graph.

    ``x`` is passed separately from ``data`` so it can carry gradients while the
    rest of the graph (pos / edges) stays fixed. Returns the [1, out_dim]
    prediction in NORMALIZED space (denormalization happens in the caller). The
    per-model dispatch mirrors ``rotation_test.predict`` exactly so explanations
    use the same forward path as the predictions they explain.
    """
    if model_name == "vanilla":
        return model(x, data.edge_index)
    if model_name == "naive_geo":
        return model(x, data.pos, data.edge_index)
    # egnn and se3 share the (x, pos, edge_index, edge_attr) signature; se3
    # ignores edge_index/edge_attr internally.
    return model(x, data.pos, data.edge_index, data.edge_attr)


def integrated_gradients(
    model_name: str,
    model: torch.nn.Module,
    normalizer: TargetNormalizer,
    data: Data,
    target_idx: int,
    steps: int = DEFAULT_IG_STEPS,
    baseline: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float, float]:
    """Integrated Gradients of one target w.r.t. node features ``data.x``.

    Attribution is taken on the DENORMALIZED prediction (physical units: eV for
    the gap, Debye for the dipole), so the magnitudes are interpretable and
    completeness is stated in those units.

    Args:
        model_name: one of the keys in ``rotation_test.MODEL_BUILDERS``.
        model:      the loaded model (already ``.eval()``).
        normalizer: target normalizer matching the model (affine, so it passes
                    gradients straight through).
        data:       a single-molecule PyG ``Data`` (uses ``x``, ``pos``, edges).
        target_idx: which output column to explain (0-based).
        steps:      number of Riemann (midpoint) steps along the IG path.
        baseline:   reference input x0; defaults to all-zeros (no atom info).

    Returns:
        (attributions, pred, baseline_pred) where ``attributions`` is an
        [N, 11] tensor of per-feature IG values, and ``pred`` / ``baseline_pred``
        are the denormalized scalar predictions at x and x0. By completeness,
        ``attributions.sum() ~= pred - baseline_pred``.
    """
    x = data.x.detach()
    if baseline is None:
        baseline = torch.zeros_like(x)
    diff = x - baseline  # input displacement the gradients get scaled by

    total_grad = torch.zeros_like(x)
    # Midpoint rule: sample alpha at (s - 0.5)/steps for s = 1..steps. The
    # midpoint rule is unbiased for the path integral and needs no endpoint
    # special-casing, which keeps completeness tight for a given step budget.
    for s in range(steps):
        alpha = (s + 0.5) / steps
        x_interp = (baseline + alpha * diff).clone().requires_grad_(True)
        out = _model_forward(model_name, model, x_interp, data)  # [1, out_dim], normalized
        target = normalizer.denormalize(out)[0, target_idx]      # scalar, physical units
        (grad,) = torch.autograd.grad(target, x_interp)
        total_grad += grad

    avg_grad = total_grad / steps
    attributions = diff * avg_grad  # [N, 11]

    # Endpoint predictions for the completeness check (no grad needed).
    with torch.no_grad():
        pred = normalizer.denormalize(
            _model_forward(model_name, model, x, data)
        )[0, target_idx].item()
        baseline_pred = normalizer.denormalize(
            _model_forward(model_name, model, baseline, data)
        )[0, target_idx].item()

    return attributions, pred, baseline_pred


def _resolve_target_idx(target: str | int, targets: tuple[str, ...]) -> tuple[int, str]:
    """Turn a target name or index into a validated (index, name) pair."""
    if isinstance(target, int):
        if not 0 <= target < len(targets):
            raise ValueError(f"target index {target} out of range for {targets}")
        return target, targets[target]
    if target not in targets:
        raise ValueError(f"unknown target {target!r}; choose from {list(targets)}")
    return targets.index(target), target


def compute_atom_importance(
    smiles: str,
    model_name: str,
    target: str | int = CONFIG.data.targets[0],
    steps: int = DEFAULT_IG_STEPS,
    weights_dir: str | None = None,
    device: str = "cpu",
) -> dict:
    """Per-atom importance for ``smiles`` under ``model_name`` for one target.

    Returns a JSON-serializable dict (the shape the ``/explain`` endpoint and
    the frontend heatmap consume):

        {
          "smiles": <canonical SMILES>,
          "model": <model_name>,
          "target": <target name>,
          "elements": ["C", "C", "O", "H", ...],      # one symbol per atom
          "importance": [...],                          # signed per-atom IG
          "importance_abs": [...],                       # |signed|, magnitude
          "importance_normalized": [...],               # importance_abs in [0, 1]
          "prediction": float,                           # denormalized f(x)
          "baseline_prediction": float,                  # denormalized f(x0)
          "completeness_error": float,                   # |sum(attr) - (pred - base)|
          "ig_steps": steps,
        }

    Per-atom score = signed SUM of that atom's 11 feature attributions. The
    signed sum preserves completeness (atom scores sum to pred - baseline_pred);
    ``importance_abs`` / ``importance_normalized`` give the magnitude a heatmap
    needs, ranked the same regardless of sign.
    """
    data = smiles_to_data(smiles)
    model, normalizer = load_model(model_name, device=device, weights_dir=weights_dir)
    target_idx, target_name = _resolve_target_idx(target, normalizer.targets)

    attributions, pred, baseline_pred = integrated_gradients(
        model_name, model, normalizer, data, target_idx, steps=steps
    )

    per_atom = attributions.sum(dim=1)            # [N] signed contribution per atom
    per_atom_abs = per_atom.abs()
    max_abs = per_atom_abs.max().clamp(min=1e-12)  # avoid div-by-0 on a null molecule
    per_atom_norm = per_atom_abs / max_abs

    elements = [_ATOMIC_SYMBOL.get(int(num), "?") for num in data.z.tolist()]
    completeness_error = abs(per_atom.sum().item() - (pred - baseline_pred))

    return {
        "smiles": data.smiles,
        "model": model_name,
        "target": target_name,
        "elements": elements,
        "importance": per_atom.tolist(),
        "importance_abs": per_atom_abs.tolist(),
        "importance_normalized": per_atom_norm.tolist(),
        "prediction": pred,
        "baseline_prediction": baseline_pred,
        "completeness_error": completeness_error,
        "ig_steps": steps,
    }


if __name__ == "__main__":
    # Self-check on an O/N-bearing molecule. We confirm the mechanism works:
    # correct shape, IG completeness, and rotation-invariant importance for the
    # equivariant models. (The chemistry-sanity interpretation is task 5.2.)
    SMILES = "CCO"  # ethanol: has the polar O that should matter for the dipole
    TARGET = "dipole_moment"

    print(f"Atom importance on {SMILES!r}, target={TARGET!r} (Integrated Gradients):\n")

    for model_name in ("vanilla", "naive_geo", "egnn", "se3"):
        result = compute_atom_importance(SMILES, model_name, target=TARGET)

        n_atoms = len(result["elements"])
        assert len(result["importance"]) == n_atoms, "one score per atom expected"
        assert all(0.0 <= v <= 1.0 for v in result["importance_normalized"]), \
            "normalized scores must lie in [0, 1]"
        # IG completeness: per-atom scores must sum to pred - baseline_pred.
        # Judge the error RELATIVE to how much the prediction moved from the
        # baseline, since that span sets the natural scale of the attributions.
        span = abs(result["prediction"] - result["baseline_prediction"]) + 1e-9
        rel_err = result["completeness_error"] / span
        assert rel_err < 0.01, (
            f"{model_name} IG completeness error {result['completeness_error']:.2e} "
            f"({rel_err:.1%} of the prediction span) too large -- increase steps "
            "or check the forward path"
        )

        ranked = sorted(
            zip(result["elements"], result["importance_normalized"]),
            key=lambda t: t[1],
            reverse=True,
        )
        top = ", ".join(f"{el}:{score:.2f}" for el, score in ranked[:4])
        print(f"  {model_name:10s} atoms={n_atoms:2d}  pred={result['prediction']:+.4f}  "
              f"completeness_err={result['completeness_error']:.1e}  top: {top}")

    # Rotation-invariance: rotating the molecule must NOT change importance for
    # the equivariant models (their prediction is invariant and x is unchanged),
    # which is exactly the reproducibility we want from the explanation.
    import math

    data = smiles_to_data(SMILES)
    theta = math.radians(37.0)
    Rz = torch.tensor([
        [math.cos(theta), -math.sin(theta), 0.0],
        [math.sin(theta), math.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    rotated = data.clone()
    rotated.pos = data.pos @ Rz.T

    for model_name in ("egnn", "se3"):
        model, normalizer = load_model(model_name)
        target_idx, _ = _resolve_target_idx(TARGET, normalizer.targets)
        attr_a, _, _ = integrated_gradients(model_name, model, normalizer, data, target_idx)
        attr_b, _, _ = integrated_gradients(model_name, model, normalizer, rotated, target_idx)
        max_diff = (attr_a.sum(dim=1) - attr_b.sum(dim=1)).abs().max().item()
        assert max_diff < 1e-3, (
            f"{model_name} importance changed under rotation (max diff {max_diff:.2e}) "
            "-- equivariant explanations should be rotation-invariant"
        )
        print(f"  {model_name} importance rotation-invariant: max diff = {max_diff:.2e}")

    print("\natom_importance self-check passed: per-atom IG scores with verified "
          "completeness, and rotation-invariant explanations for EGNN/SE(3).")
