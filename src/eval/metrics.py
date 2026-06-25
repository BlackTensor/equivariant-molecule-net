"""Phase 4 metrics — Equivariance Stability Score (ESS) + MAE/RMSE.

Task 4.1: implement the ESS exactly as defined in CLAUDE.md, plus plain
regression helpers (MAE/RMSE) reused by training and evaluation.

ESS formula (see CLAUDE.md "Custom Metric: Equivariance Stability Score"):

    For a molecule, generate N rotated copies (default N=24, every 15 deg).
    Predict the property for each rotated copy.
    ESS = 1 - (variance_of_predictions / variance_normalizer)

- A perfectly equivariant model -> predictions identical -> variance ~ 0
  -> ESS ~ 1.0
- A geometry-ignorant model -> predictions wobble -> higher variance
  -> lower ESS

The normalizer must be the variance of predictions ACROSS DIFFERENT
MOLECULES (not across rotations), so ESS is comparable across properties
with different natural scales. This keeps a near-zero-variance normalizer
from blowing ESS up/down when a property barely varies between molecules.
"""

from __future__ import annotations

import numpy as np

# Floor so a near-constant property across molecules (normalizer ~ 0) can't
# make ESS explode; this only matters for pathological/synthetic inputs.
_EPS = 1e-12

# Local default seed; keeps this module import-light (no dependency on
# src.train.config / torch) while still being reproducible.
_SELF_TEST_SEED = 42


def rotation_variance(predictions: np.ndarray) -> float:
    """Variance of a single molecule's predictions across its N rotations.

    ``predictions`` is a 1D array of length N (one scalar prediction per
    rotated copy of the SAME molecule).
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    return float(np.var(predictions))


def cross_molecule_variance(reference_predictions: np.ndarray) -> float:
    """Variance normalizer: spread of predictions ACROSS DIFFERENT molecules.

    ``reference_predictions`` is a 1D array with one (unrotated) prediction
    per molecule in some reference set, used purely to put ESS on a scale
    that is comparable across properties.
    """
    reference_predictions = np.asarray(reference_predictions, dtype=np.float64)
    return float(np.var(reference_predictions))


def equivariance_stability_score(
    rotated_predictions: np.ndarray,
    reference_predictions: np.ndarray,
) -> float:
    """ESS = 1 - (variance_of_predictions / variance_normalizer).

    Args:
        rotated_predictions: 1D array of length N, one prediction per
            rotated copy of a single molecule.
        reference_predictions: 1D array of predictions across different
            molecules (unrotated), used as the variance normalizer.

    Returns:
        ESS as a float. ~1.0 for a perfectly equivariant model, lower
        (can go negative) the more predictions wobble under rotation.
    """
    numerator = rotation_variance(rotated_predictions)
    denominator = cross_molecule_variance(reference_predictions)
    return 1.0 - (numerator / max(denominator, _EPS))


def mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Mean Absolute Error."""
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    return float(np.mean(np.abs(predictions - targets)))


def rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Root Mean Squared Error."""
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    return float(np.sqrt(np.mean((predictions - targets) ** 2)))


def _self_test() -> None:
    """Unit-test ESS on synthetic constant vs. noisy predictions.

    Run: python -m src.eval.metrics
    """
    rng = np.random.default_rng(_SELF_TEST_SEED)

    # A spread of "different molecule" reference predictions, used as the
    # normalizer for every case below so ESS values are comparable.
    reference_predictions = rng.normal(loc=5.0, scale=2.0, size=200)

    # Case 1: perfectly equivariant model -> identical prediction for every
    # rotation -> variance == 0 -> ESS == 1.0 exactly.
    constant_predictions = np.full(24, 3.7)
    ess_constant = equivariance_stability_score(constant_predictions, reference_predictions)
    assert abs(ess_constant - 1.0) < 1e-9, f"expected ESS == 1.0, got {ess_constant}"

    # Case 2: geometry-ignorant model -> noisy predictions across rotations
    # of the SAME molecule -> variance > 0 -> ESS < 1.0.
    noisy_predictions = rng.normal(loc=3.7, scale=1.5, size=24)
    ess_noisy = equivariance_stability_score(noisy_predictions, reference_predictions)
    assert ess_noisy < ess_constant, "noisy predictions must score lower ESS than constant ones"

    # Case 3: more noise -> lower ESS than less noise (monotonic in variance).
    very_noisy_predictions = rng.normal(loc=3.7, scale=4.0, size=24)
    ess_very_noisy = equivariance_stability_score(very_noisy_predictions, reference_predictions)
    assert ess_very_noisy < ess_noisy, "more rotation noise must lower ESS further"

    # MAE/RMSE sanity: zero error -> zero; RMSE >= MAE for any error vector.
    preds = np.array([1.0, 2.0, 3.0, 4.0])
    targs = np.array([1.0, 2.0, 3.0, 4.0])
    assert mae(preds, targs) == 0.0
    assert rmse(preds, targs) == 0.0

    preds = np.array([1.0, 2.0, 3.0, 10.0])
    targs = np.array([1.0, 2.0, 3.0, 4.0])
    m, r = mae(preds, targs), rmse(preds, targs)
    assert r >= m, "RMSE must be >= MAE (Cauchy-Schwarz)"

    print(f"ESS (constant, perfectly equivariant):   {ess_constant:.6f}")
    print(f"ESS (noisy, scale=1.5):                  {ess_noisy:.6f}")
    print(f"ESS (very noisy, scale=4.0):              {ess_very_noisy:.6f}")
    print(f"MAE/RMSE on noisy example: mae={m:.4f} rmse={r:.4f}")
    print("\nPhase 4 metrics self-test passed: ESS == 1.0 for constant "
          "predictions and decreases monotonically with rotation noise.")


if __name__ == "__main__":
    _self_test()
