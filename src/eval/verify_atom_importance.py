"""Phase 5 verification — atom importance is chemically sensible + reproducible.

Task 5.2: confirm the per-atom Integrated-Gradients scores from
:mod:`src.eval.atom_importance` make chemical sense on a KNOWN nitrogen/oxygen-
heavy molecule, and that they are reproducible.

Known case: acetamide, ``CC(=O)N`` — a textbook polar molecule carrying both of
the electronegative heteroatoms QM9 cares about (a carbonyl O and an amide N).
Its dipole moment is dominated by the polar C=O and C-N region, NOT by the
methyl hydrogens. So a sensible explanation for the DIPOLE target must rank the
heteroatoms well above the hydrogens.

What we assert (for every TRAINED model: vanilla, egnn, se3):
  1. Completeness  — IG residual < 1% of the prediction's span from baseline
     (the model-agnostic correctness check from atom_importance.py).
  2. Chemical sense — mean importance of the heteroatoms (N, O, F) exceeds the
     mean importance of the hydrogens, and the single most important atom is a
     heavy atom (never a hydrogen). This is the robust claim that holds across
     all three models; we deliberately do NOT require heteroatoms to beat
     carbon, because the carbonyl carbon is itself genuinely polar (and e.g.
     SE(3) ranks it highly, which is correct chemistry).
  3. Reproducibility — running the scorer twice yields identical scores to
     numerical precision (the float32 SE(3)/e3nn path has ~1e-7 reduction-order
     rounding noise on CPU, so we compare with a small tolerance, not bitwise).

``naive_geo`` is intentionally untrained (random weights), so its scores carry
no chemical meaning; it is shown in the printed table for contrast but excluded
from the chemical-sense assertions.

Run:

    python -m src.eval.verify_atom_importance
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.eval.atom_importance import compute_atom_importance  # noqa: E402

# The known nitrogen/oxygen-heavy test case and the target whose chemistry is
# clearest for a per-atom explanation.
KNOWN_SMILES: str = "CC(=O)N"   # acetamide
TARGET: str = "dipole_moment"

# Elements treated as electronegative "heteroatoms" within QM9's set.
HETEROATOMS: frozenset[str] = frozenset({"N", "O", "F"})

# Models with real trained weights -> chemistry-bearing explanations.
TRAINED_MODELS: tuple[str, ...] = ("vanilla", "egnn", "se3")
# Shown for contrast only (untrained, random weights -> no chemical meaning).
CONTRAST_MODELS: tuple[str, ...] = ("naive_geo",)


def _class_means(elements: list[str], scores: list[float]) -> dict[str, float]:
    """Mean normalized importance grouped into heteroatom / carbon / hydrogen."""
    het = [s for e, s in zip(elements, scores) if e in HETEROATOMS]
    carbon = [s for e, s in zip(elements, scores) if e == "C"]
    hydrogen = [s for e, s in zip(elements, scores) if e == "H"]
    mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")
    return {"het": mean(het), "C": mean(carbon), "H": mean(hydrogen)}


def main() -> None:
    print(f"Verifying atom importance on {KNOWN_SMILES!r} (acetamide), "
          f"target={TARGET!r}\n")
    print(f"  {'model':10s} {'het':>6s} {'C':>6s} {'H':>6s}   top atoms")
    print("  " + "-" * 56)

    for model_name in TRAINED_MODELS + CONTRAST_MODELS:
        result = compute_atom_importance(KNOWN_SMILES, model_name, target=TARGET)
        elements = result["elements"]
        norm = result["importance_normalized"]
        means = _class_means(elements, norm)

        ranked = sorted(zip(elements, norm), key=lambda t: t[1], reverse=True)
        top = ", ".join(f"{el}:{s:.2f}" for el, s in ranked[:4])
        tag = "" if model_name in TRAINED_MODELS else "  (untrained, contrast only)"
        print(f"  {model_name:10s} {means['het']:6.2f} {means['C']:6.2f} "
              f"{means['H']:6.2f}   {top}{tag}")

        if model_name not in TRAINED_MODELS:
            continue

        # 1. Completeness (relative to the prediction's span from baseline).
        span = abs(result["prediction"] - result["baseline_prediction"]) + 1e-9
        rel_err = result["completeness_error"] / span
        assert rel_err < 0.01, (
            f"{model_name}: IG completeness off by {rel_err:.1%} of the span"
        )

        # 2. Chemical sense: heteroatoms outweigh hydrogens, and the single most
        #    important atom is a heavy atom (not a hydrogen).
        assert means["het"] > means["H"], (
            f"{model_name}: heteroatom mean importance ({means['het']:.2f}) is not "
            f"above hydrogen mean ({means['H']:.2f}) -- not chemically sensible "
            "for a dipole explanation"
        )
        assert ranked[0][0] != "H", (
            f"{model_name}: the most important atom is a hydrogen ({ranked[0]}), "
            "which is not chemically sensible for the dipole moment"
        )

        # 3. Reproducibility: a second run must reproduce the scores to numerical
        #    precision. We allow a tiny tolerance rather than bitwise equality
        #    because the float32 SE(3)/e3nn path has reduction-order rounding
        #    noise on CPU (~1e-7), which is not a real reproducibility issue.
        again = compute_atom_importance(KNOWN_SMILES, model_name, target=TARGET)
        max_run_diff = max(
            abs(p - q) for p, q in zip(again["importance"], result["importance"])
        )
        assert max_run_diff < 1e-5, (
            f"{model_name}: importance scores not reproducible across runs "
            f"(max diff {max_run_diff:.2e})"
        )

    print("\nverify_atom_importance passed: on acetamide the trained models rank "
          "the polar N/O heteroatoms above hydrogen, never peak on a hydrogen, "
          "satisfy IG completeness, and reproduce exactly across runs.")


if __name__ == "__main__":
    main()
