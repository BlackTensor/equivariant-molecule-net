"""Task 4.3 — run the rotation stress test over a handful of molecules.

Drives :func:`src.eval.rotation_test.run_rotation_test` across several small
QM9-element molecules (H/C/N/O/F only), confirms the flagship pattern, and
saves everything to a single JSON file for the frontend / write-up:

    naive_geo  -> predictions WOBBLE under rotation (geometry-aware, NOT
                  equivariant)            -> lower ESS
    vanilla    -> predictions exactly FLAT (geometry-blind: ignores pos)
    egnn, se3  -> predictions FLAT *and* robust (equivariant by construction)
                  -> ESS ~ 1.0

Run:

    python -m src.eval.run_rotation_suite
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.train.config import CONFIG  # noqa: E402
from src.eval.rotation_test import run_rotation_test  # noqa: E402

# A handful of diverse, small molecules using only QM9's element set (H,C,N,O,F).
# Spanning single/double/triple bonds, an aromatic ring, N/O/F heteroatoms.
SUITE: dict[str, str] = {
    "ethanol": "CCO",
    "acetonitrile": "CC#N",
    "acetic_acid": "CC(=O)O",
    "methylamine": "CN",
    "fluoromethane": "CF",
    "benzene": "c1ccccc1",
    "glycine": "C(C(=O)O)N",
}

# Where the flagship result is written (kept in git — small, meaningful).
RESULTS_DIR = os.path.join(_REPO_ROOT, "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "rotation_test_results.json")

# Models whose readout is equivariant by construction -> must stay flat.
EQUIVARIANT_MODELS = ("egnn", "se3")


def _pred_range(preds: list[float]) -> float:
    """Peak-to-peak spread of a model's predictions across the 24 rotations."""
    return max(preds) - min(preds)


def run_suite() -> dict:
    """Run the rotation test on every molecule in ``SUITE`` and bundle results."""
    molecules: dict[str, dict] = {}
    for name, smiles in SUITE.items():
        print(f"  running {name:13s} ({smiles}) ...", flush=True)
        molecules[name] = run_rotation_test(smiles)

    return {
        "metadata": {
            "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "task": "4.3 rotation stress test",
            "num_rotations": CONFIG.rotation.num_rotations,
            "targets": list(CONFIG.data.targets),
            "models": ["vanilla", "naive_geo", "egnn", "se3"],
            "seed": CONFIG.seed,
            "note": (
                "vanilla is flat because it is geometry-blind (ignores pos); "
                "naive_geo wobbles because it uses raw coordinates non-"
                "equivariantly; egnn/se3 are flat AND robust (equivariant)."
            ),
        },
        "molecules": molecules,
    }


def confirm_pattern(results: dict) -> None:
    """Assert the expected wobbly-vs-flat pattern holds for every molecule."""
    for name, mol in results["molecules"].items():
        models = mol["models"]

        # vanilla: ignores pos -> EXACTLY flat across rotations.
        for target, preds in models["vanilla"]["predictions"].items():
            assert _pred_range(preds) < 1e-6, (
                f"{name}/{target}: vanilla varied under rotation but ignores pos"
            )

        # naive_geo: uses raw coords non-equivariantly -> must WOBBLE, and its
        # ESS must sit below every equivariant model on the same target.
        naive = models["naive_geo"]
        for target, preds in naive["predictions"].items():
            assert _pred_range(preds) > 1e-3, (
                f"{name}/{target}: naive_geo did not wobble (range "
                f"{_pred_range(preds):.2e}) -- check pos is used"
            )

        # egnn/se3: equivariant -> ESS ~ 1.0 and strictly above naive_geo.
        for model_name in EQUIVARIANT_MODELS:
            for target, ess in models[model_name]["ess"].items():
                assert ess > 0.99, (
                    f"{name}/{target}: {model_name} ESS={ess:.4f} far from 1.0"
                )
                assert naive["ess"][target] < ess, (
                    f"{name}/{target}: naive_geo ESS ({naive['ess'][target]:.4f}) "
                    f"not below {model_name} ({ess:.4f})"
                )


def _print_summary(results: dict) -> None:
    """Print a compact per-molecule ESS / wobble table for the HOMO-LUMO gap."""
    target = results["metadata"]["targets"][0]  # homo_lumo_gap
    print(f"\nSummary (target = {target}):")
    header = f"  {'molecule':13s} {'vanilla':>9s} {'naive_geo':>11s} {'egnn':>7s} {'se3':>7s}   {'naive wobble':>13s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, mol in results["molecules"].items():
        m = mol["models"]
        v = m["vanilla"]["ess"][target]
        ng = m["naive_geo"]["ess"][target]
        eg = m["egnn"]["ess"][target]
        se = m["se3"]["ess"][target]
        wob = _pred_range(m["naive_geo"]["predictions"][target])
        print(f"  {name:13s} {v:9.4f} {ng:11.4f} {eg:7.4f} {se:7.4f}   {wob:11.4f} eV")


def main() -> None:
    print(f"Rotation stress test over {len(SUITE)} molecules "
          f"({CONFIG.rotation.num_rotations} rotations each):")
    results = run_suite()

    confirm_pattern(results)
    _print_summary(results)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    rel = os.path.relpath(RESULTS_PATH, _REPO_ROOT)
    print(f"\nPattern confirmed for all {len(SUITE)} molecules. Saved -> {rel}")


if __name__ == "__main__":
    main()
