"""Phase 4 rotation stress test — the flagship result.

Task 4.2: for a given molecule, generate 24 rotated copies (0-345 deg, 15 deg
steps), predict with all three trained models (Vanilla GIN, EGNN, SE(3)-
Transformer), and report both the raw per-rotation predictions and the
Equivariance Stability Score (:func:`src.eval.metrics.equivariance_stability_score`)
per model per target.

The ESS normalizer is the cross-molecule variance of (unrotated) predictions
over a small fixed reference set of different molecules — see CLAUDE.md and
``metrics.py`` for why that keeps ESS comparable across the gap/dipole targets.

Run a standalone self-check (loads the trained weights in ``weights/``, CPU,
no QM9 download needed):

    python -m src.eval.rotation_test
"""

from __future__ import annotations

import math
import os
import sys

import torch
from torch_geometric.data import Data

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.train.config import CONFIG, set_seed  # noqa: E402
from src.data.qm9_loader import TargetNormalizer  # noqa: E402
from src.data.molecule_utils import smiles_to_data  # noqa: E402
from src.models.vanilla_gnn import build_vanilla_gnn  # noqa: E402
from src.models.naive_geo_gnn import build_naive_geo_gnn  # noqa: E402
from src.models.egnn import build_egnn  # noqa: E402
from src.models.se3_transformer import build_se3_transformer  # noqa: E402
from src.eval.metrics import equivariance_stability_score  # noqa: E402

MODEL_BUILDERS = {
    "vanilla": build_vanilla_gnn,
    "naive_geo": build_naive_geo_gnn,
    "egnn": build_egnn,
    "se3": build_se3_transformer,
}

# Models with NO trained weights on disk: they are evaluated with random
# (seeded) initialization. ``naive_geo`` exists purely to demonstrate that a
# geometry-aware-but-non-equivariant model wobbles under rotation, and an
# untrained model already shows that wobble (see naive_geo_gnn.py).
UNTRAINED_MODELS: frozenset[str] = frozenset({"naive_geo"})

# Small, diverse, QM9-element-only molecules used purely as the "different
# molecules" reference set for the ESS normalizer (see metrics.py). Kept
# distinct from whatever molecule the caller is testing.
DEFAULT_REFERENCE_SMILES: tuple[str, ...] = (
    "CCO", "CC#N", "C", "CO", "CCC", "c1ccccc1", "CC(=O)O", "CN",
)

# A single fixed, non-axis-aligned rotation axis. Using a tilted axis (rather
# than e.g. pure z) avoids accidentally hiding an equivariance bug that only
# shows up for off-axis rotations.
_DEFAULT_AXIS: tuple[float, float, float] = (1.0, 1.0, 1.0)


def _rotation_matrix(axis: tuple[float, float, float], angle_deg: float) -> torch.Tensor:
    """Rodrigues' rotation formula: 3x3 proper rotation matrix about ``axis``."""
    a = torch.tensor(axis, dtype=torch.float)
    a = a / a.norm()
    theta = math.radians(angle_deg)
    K = torch.tensor(
        [[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]]
    )
    return torch.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)


def generate_rotated_copies(
    data: Data,
    num_rotations: int = CONFIG.rotation.num_rotations,
    step_deg: float = 15.0,
    axis: tuple[float, float, float] = _DEFAULT_AXIS,
) -> tuple[list[float], list[Data]]:
    """Build ``num_rotations`` rotated copies of ``data`` (only ``pos`` changes).

    Returns (angles_deg, copies) where angles run 0, step_deg, 2*step_deg, ...
    """
    angles = [i * step_deg for i in range(num_rotations)]
    copies = []
    for angle in angles:
        R = _rotation_matrix(axis, angle)
        rotated = data.clone()
        rotated.pos = data.pos @ R.T
        copies.append(rotated)
    return angles, copies


def _identity_normalizer() -> TargetNormalizer:
    """A no-op normalizer (mean 0, std 1) for untrained models.

    An untrained model's outputs are arbitrary, so there are no fitted stats to
    apply. ESS only depends on the RATIO of rotation variance to cross-molecule
    variance, both measured in these same arbitrary units, so leaving the raw
    outputs untouched is fine for the wobble demonstration.
    """
    out_dim = CONFIG.model.out_dim
    return TargetNormalizer(
        mean=torch.zeros(out_dim),
        std=torch.ones(out_dim),
        targets=CONFIG.data.targets,
    )


def load_model(
    model_name: str, device: str = "cpu", weights_dir: str | None = None
) -> tuple[torch.nn.Module, TargetNormalizer]:
    """Load a model + its target normalizer for the rotation test.

    Trained models come from ``weights/<name>_best.pt``. Untrained models (see
    :data:`UNTRAINED_MODELS`, currently just ``naive_geo``) are built with
    random but SEEDED weights so their ESS is reproducible across runs.
    """
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"model must be one of {list(MODEL_BUILDERS)}, got {model_name!r}")

    if model_name in UNTRAINED_MODELS:
        # Seed first so the random initialization is identical every run.
        set_seed(CONFIG.seed)
        model = MODEL_BUILDERS[model_name]().to(device)
        model.eval()
        return model, _identity_normalizer()

    weights_dir = CONFIG.train.weights_dir if weights_dir is None else weights_dir
    path = os.path.join(weights_dir, f"{model_name}_best.pt")
    checkpoint = torch.load(path, map_location=device)

    model = MODEL_BUILDERS[model_name]().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    normalizer = TargetNormalizer(
        mean=checkpoint["normalizer_mean"],
        std=checkpoint["normalizer_std"],
        targets=checkpoint["targets"],
    )
    return model, normalizer


def predict(
    model_name: str, model: torch.nn.Module, normalizer: TargetNormalizer, data: Data
) -> torch.Tensor:
    """Predict one molecule, denormalized to physical units. Returns [out_dim]."""
    with torch.no_grad():
        if model_name == "vanilla":
            out = model(data.x, data.edge_index)
        elif model_name == "naive_geo":
            # Uses pos (concatenated as features) but no edge_attr.
            out = model(data.x, data.pos, data.edge_index)
        else:
            # egnn and se3 both use pos; se3 ignores edge_index/edge_attr internally.
            out = model(data.x, data.pos, data.edge_index, data.edge_attr)
    return normalizer.denormalize(out).squeeze(0)


def run_rotation_test(
    smiles: str,
    model_names: tuple[str, ...] = ("vanilla", "naive_geo", "egnn", "se3"),
    num_rotations: int = CONFIG.rotation.num_rotations,
    reference_smiles: tuple[str, ...] = DEFAULT_REFERENCE_SMILES,
    weights_dir: str | None = None,
    device: str = "cpu",
) -> dict:
    """Rotate ``smiles`` 24 ways, predict with every model, and score ESS.

    Returns a JSON-serializable dict:
        {
          "smiles": <canonical SMILES>,
          "angles_deg": [0, 15, ..., 345],
          "models": {
            "<model_name>": {
              "predictions": {"<target>": [v_0, ..., v_23], ...},
              "ess": {"<target>": float, ...},
            },
            ...
          },
        }
    """
    data = smiles_to_data(smiles)
    angles, rotated_copies = generate_rotated_copies(data, num_rotations=num_rotations)
    reference_data = [smiles_to_data(s) for s in reference_smiles]

    results: dict = {"smiles": data.smiles, "angles_deg": angles, "models": {}}

    for model_name in model_names:
        model, normalizer = load_model(model_name, device=device, weights_dir=weights_dir)

        rot_preds = torch.stack(
            [predict(model_name, model, normalizer, d) for d in rotated_copies]
        )  # [num_rotations, out_dim]
        ref_preds = torch.stack(
            [predict(model_name, model, normalizer, d) for d in reference_data]
        )  # [num_reference, out_dim]

        predictions_by_target: dict[str, list[float]] = {}
        ess_by_target: dict[str, float] = {}
        for i, target_name in enumerate(normalizer.targets):
            predictions_by_target[target_name] = rot_preds[:, i].tolist()
            ess_by_target[target_name] = equivariance_stability_score(
                rot_preds[:, i].numpy(), ref_preds[:, i].numpy()
            )

        results["models"][model_name] = {
            "predictions": predictions_by_target,
            "ess": ess_by_target,
        }

    return results


if __name__ == "__main__":
    results = run_rotation_test("CCO")  # ethanol

    print(f"Rotation test on {results['smiles']!r}: "
          f"{len(results['angles_deg'])} rotations "
          f"(0-{results['angles_deg'][-1]:.0f} deg, step={results['angles_deg'][1]:.0f} deg)\n")

    for model_name, model_result in results["models"].items():
        print(f"{model_name}:")
        for target_name, ess in model_result["ess"].items():
            preds = model_result["predictions"][target_name]
            print(f"  {target_name:14s} ESS={ess:7.4f}  "
                  f"pred range=[{min(preds):.4f}, {max(preds):.4f}]")

    # Vanilla never reads `pos`, so rotating the molecule cannot change its
    # input at all -> predictions must be EXACTLY identical across rotations.
    # (Flat, but for the WRONG reason: blindness, not robustness.)
    vanilla_preds = results["models"]["vanilla"]["predictions"]
    for target_name, preds in vanilla_preds.items():
        assert max(preds) - min(preds) < 1e-6, (
            f"vanilla {target_name} predictions vary under rotation, but it "
            "never consumes `pos` -- something is wrong"
        )

    # naive_geo DOES read `pos` (raw coords concatenated as features) but is not
    # equivariant, so its predictions must visibly WOBBLE under rotation and its
    # ESS must drop well below the equivariant models'. This is the "wobbly"
    # line the demo needs -- the whole reason this model was added.
    naive_result = results["models"]["naive_geo"]
    for target_name, preds in naive_result["predictions"].items():
        assert max(preds) - min(preds) > 1e-3, (
            f"naive_geo {target_name} barely moved under rotation, but it is "
            "supposed to be non-equivariant -- check pos is being used"
        )

    # EGNN / SE(3)-Transformer are provably E(3)/SE(3)-equivariant by
    # construction (see their module docstrings), so their readout should be
    # rotation-invariant up to floating-point error regardless of training
    # quality -> ESS should be very close to 1.0.
    for model_name in ("egnn", "se3"):
        for target_name, ess in results["models"][model_name]["ess"].items():
            assert ess > 0.99, (
                f"{model_name} {target_name} ESS={ess:.4f} is far from 1.0 -- "
                "equivariance may be broken"
            )
            # The flagship gap: naive_geo must score clearly lower than the
            # equivariant models on the very same target.
            assert naive_result["ess"][target_name] < ess, (
                f"naive_geo ESS ({naive_result['ess'][target_name]:.4f}) is not "
                f"below {model_name}'s ({ess:.4f}) for {target_name} -- the "
                "wobbly-vs-flat contrast is missing"
            )

    print("\nrotation_test self-check passed: vanilla is flat-but-blind "
          "(ignores pos), naive_geo wobbles (uses pos, not equivariant), and "
          "EGNN/SE(3) are flat-AND-robust (equivariant, ESS > 0.99).")
