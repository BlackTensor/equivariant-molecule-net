"""FastAPI backend for EquiDrug (Phase 6).

Task 6.1: the ``/predict`` endpoint. A user POSTs a SMILES string and gets back
each of the three models' predictions for both QM9 targets (HOMO-LUMO gap in eV,
dipole moment in Debye), already denormalized into physical units.

The three models compared throughout the project:
  * ``vanilla`` — GIN baseline, ignores 3D coordinates (the weak model)
  * ``egnn``    — E(3)-equivariant GNN (Satorras et al. 2021)
  * ``se3``     — SE(3)-Transformer (e3nn)

We deliberately reuse :func:`src.eval.rotation_test.load_model` and
:func:`src.eval.rotation_test.predict` so the API runs models exactly the way
the rotation stress test does — no second, drifting copy of the inference logic.

Run locally:

    uvicorn src.api.main:app --reload

then POST ``{"smiles": "CCO"}`` to ``http://127.0.0.1:8000/predict``.
(Interactive docs are auto-generated at ``/docs``.)
"""

from __future__ import annotations

import os
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.data.molecule_utils import (  # noqa: E402
    MoleculeEmbedError,
    smiles_to_data,
    smiles_to_molblock,
)
from src.eval.rotation_test import load_model, predict, run_rotation_test  # noqa: E402
from src.eval.atom_importance import compute_atom_importance  # noqa: E402
from src.train.config import CONFIG  # noqa: E402

# The three models the project pitch compares for a single prediction.
# (``naive_geo`` is the rotation-test's wobble demonstrator — see ROTATION_MODELS.)
API_MODELS: tuple[str, ...] = ("vanilla", "egnn", "se3")

# Models shown in the rotation stress test. ``naive_geo`` is included here on
# purpose: it is the geometry-aware-but-NOT-equivariant model whose predictions
# genuinely wobble under rotation (low ESS). It is the flagship contrast against
# the flat-and-equivariant EGNN/SE(3) lines, so the rotation chart needs it.
ROTATION_MODELS: tuple[str, ...] = ("vanilla", "naive_geo", "egnn", "se3")

# Human-friendly labels for the response, keyed by model name.
MODEL_LABELS: dict[str, str] = {
    "vanilla": "Vanilla GNN (GIN)",
    "naive_geo": "Naive Geometric GNN (non-equivariant)",
    "egnn": "EGNN",
    "se3": "SE(3)-Transformer",
}

app = FastAPI(
    title="EquiDrug API",
    description="Geometric molecular property prediction under symmetry.",
    version="0.1.0",
)

# CORS for the Phase 7 React + Vite frontend. The browser blocks cross-origin
# requests by default, so we explicitly allow the local Vite dev server
# (5173) and a couple of common alternates. All free / local — no hosted CORS
# service. When deployed to HF Spaces (Phase 8) the frontend is same-origin, so
# this only matters for local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models are loaded once at startup and reused for every request (loading weights
# per-request would be needlessly slow). Populated by the startup handler below.
_MODELS: dict[str, tuple] = {}


@app.on_event("startup")
def _load_models() -> None:
    """Load every model + its target normalizer once, on the CPU."""
    for name in API_MODELS:
        _MODELS[name] = load_model(name, device="cpu")


class PredictRequest(BaseModel):
    """Input payload for ``/predict``."""

    smiles: str = Field(..., examples=["CCO"], description="Molecule as a SMILES string.")


class PredictResponse(BaseModel):
    """Output payload for ``/predict``.

    ``predictions`` maps model name -> {label, predictions:{target: {value, unit}}}.
    """

    smiles: str = Field(..., description="Canonical SMILES of the parsed molecule.")
    num_atoms: int = Field(..., description="Atom count including explicit hydrogens.")
    predictions: dict


@app.get("/")
def root() -> dict:
    """Tiny health/info endpoint."""
    return {"name": "EquiDrug API", "models": list(API_MODELS), "status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest) -> PredictResponse:
    """SMILES in -> all three models' predictions out (physical units)."""
    # Build the 3D graph once and run every model on the same geometry.
    try:
        data = smiles_to_data(request.smiles)
    except MoleculeEmbedError as exc:
        # Invalid SMILES / non-QM9 element / 3D embedding failure -> client error.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    predictions: dict = {}
    for name in API_MODELS:
        model, normalizer = _MODELS[name]
        out = predict(name, model, normalizer, data)  # [out_dim], physical units
        units = normalizer.units()
        predictions[name] = {
            "label": MODEL_LABELS[name],
            "predictions": {
                target: {"value": float(out[i]), "unit": units[i]}
                for i, target in enumerate(normalizer.targets)
            },
        }

    return PredictResponse(
        smiles=data.smiles,
        num_atoms=data.num_nodes,
        predictions=predictions,
    )


class StructureRequest(BaseModel):
    """Input payload for ``/structure``."""

    smiles: str = Field(..., examples=["CCO"], description="Molecule as a SMILES string.")


class StructureResponse(BaseModel):
    """Output payload for ``/structure``.

    ``sdf`` is a V2000 MOL block of the RDKit-embedded 3D geometry — the same
    coordinates the models see — ready to hand to 3Dmol.js in the frontend.
    """

    smiles: str = Field(..., description="Canonical SMILES of the parsed molecule.")
    num_atoms: int = Field(..., description="Atom count including explicit hydrogens.")
    sdf: str = Field(..., description="3D structure as a V2000 MOL/SDF block.")


@app.post("/structure", response_model=StructureResponse)
def structure_endpoint(request: StructureRequest) -> StructureResponse:
    """SMILES in -> 3D MOL block out, for the 3Dmol.js viewer (task 7.2).

    The molecule is embedded in 3D with RDKit (explicit Hs, seeded ETKDG, MMFF
    relaxed) — identical to the geometry the predictors use — so the rendered
    structure matches what the models actually run on.
    """
    try:
        sdf = smiles_to_molblock(request.smiles)
        # Parse once more for a canonical SMILES + atom count in the response.
        data = smiles_to_data(request.smiles)
    except MoleculeEmbedError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return StructureResponse(smiles=data.smiles, num_atoms=data.num_nodes, sdf=sdf)


class RotateTestRequest(BaseModel):
    """Input payload for ``/rotate_test``."""

    smiles: str = Field(..., examples=["CCO"], description="Molecule as a SMILES string.")


class RotateTestResponse(BaseModel):
    """Output payload for ``/rotate_test``.

    ``angles_deg`` is the shared x-axis (0, 15, ..., 345). ``models`` maps each
    model name -> {label, predictions:{target: [v_0..v_23]}, ess:{target: float}}.
    """

    smiles: str = Field(..., description="Canonical SMILES of the parsed molecule.")
    angles_deg: list = Field(..., description="Rotation angles in degrees.")
    models: dict


@app.post("/rotate_test", response_model=RotateTestResponse)
def rotate_test_endpoint(request: RotateTestRequest) -> RotateTestResponse:
    """SMILES in -> 24-rotation predictions + ESS per model out.

    Rotates the molecule through 24 orientations (15 deg steps about a fixed
    tilted axis), predicts every target with each model at every orientation,
    and scores the Equivariance Stability Score (ESS) per model per target.
    EGNN/SE(3) stay flat (ESS ~ 1.0); the naive geometric GNN wobbles (low ESS).
    """
    try:
        # Reuse the exact Phase 4 flagship test so the API and the offline
        # results stay in lockstep. It raises MoleculeEmbedError on bad input.
        results = run_rotation_test(request.smiles, model_names=ROTATION_MODELS)
    except MoleculeEmbedError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Attach human-friendly labels alongside the raw per-model results.
    models: dict = {}
    for name, model_result in results["models"].items():
        models[name] = {"label": MODEL_LABELS[name], **model_result}

    return RotateTestResponse(
        smiles=results["smiles"],
        angles_deg=results["angles_deg"],
        models=models,
    )


# Every model has an explanation surface (per-atom Integrated Gradients on the
# node features), so /explain accepts all four — including naive_geo.
EXPLAIN_MODELS: tuple[str, ...] = ROTATION_MODELS


class ExplainRequest(BaseModel):
    """Input payload for ``/explain``."""

    smiles: str = Field(..., examples=["CCO"], description="Molecule as a SMILES string.")
    model: str = Field(
        ..., examples=["egnn"], description=f"Model to explain; one of {list(EXPLAIN_MODELS)}."
    )
    target: str = Field(
        CONFIG.data.targets[0],
        examples=["dipole_moment"],
        description=f"Property to explain; one of {list(CONFIG.data.targets)}.",
    )


@app.post("/explain")
def explain_endpoint(request: ExplainRequest) -> dict:
    """SMILES + model in -> per-atom importance out.

    Returns per-atom Integrated Gradients attributions (signed, absolute, and
    [0,1]-normalized for a heatmap), the element symbols, the prediction it
    explains, and the IG completeness error. See ``atom_importance.py`` for the
    exact contract — this endpoint returns that dict verbatim plus a model label.
    """
    # Validate the model up front so the client gets a clear 400, not a 500.
    if request.model not in EXPLAIN_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"model must be one of {list(EXPLAIN_MODELS)}, got {request.model!r}",
        )

    try:
        result = compute_atom_importance(request.smiles, request.model, target=request.target)
    except MoleculeEmbedError as exc:
        # Invalid SMILES / non-QM9 element / 3D embedding failure.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        # Unknown target name (from _resolve_target_idx).
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result["label"] = MODEL_LABELS[request.model]
    return result
