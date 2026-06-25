# EquiDrug — Geometric AI for Drug Discovery

> An interactive framework that quantitatively measures and visualizes the geometric robustness of molecular neural networks under symmetry transformations.

Enter a SMILES string. Three models predict quantum-chemical properties. The molecule rotates. Watch which models hold steady and which wobble — and exactly *why*.

---

## What It Does

1. Converts a SMILES string to a 3D molecular graph via RDKit.
2. Runs three models side-by-side: Vanilla GNN, EGNN, SE(3)-Transformer.
3. Predicts **HOMO-LUMO gap** (eV) and **dipole moment** (Debye).
4. Stress-tests each model across 24 rotations (0°–345°, 15° steps).
5. Scores geometric robustness with the **Equivariance Stability Score (ESS)**.
6. Renders everything in an interactive React dashboard with a live 3D viewer.

---

## Architecture

```
SMILES input
     │
     ▼
RDKit 3D embedder  ──►  atoms, coords, bonds  ──►  PyG Data object
     │
     ├──► Vanilla GNN (GIN)           ignores coords  ──► prediction
     ├──► EGNN (Satorras et al. 2021) invariant dists ──► prediction
     └──► SE(3)-Transformer (e3nn)    spherical harm. ──► prediction
                                              │
                                              ▼
                                  24-rotation stress test
                                              │
                                              ▼
                                Equivariance Stability Score (ESS)
                                              │
                              ┌───────────────┴───────────────┐
                              ▼                               ▼
                        FastAPI backend               React + Vite frontend
                        /predict                      3Dmol.js  │  Recharts
                        /rotate_test                  live predictions
                        /explain                      rotation chart
                                                      atom-importance overlay
```

---

## Key Results

Trained on QM9 (~50 k molecules), evaluated on the held-out test split.

### Prediction accuracy (MAE, normalized units)

| Model | HOMO-LUMO gap | Dipole moment |
|---|---|---|
| Vanilla GNN (GIN) | 0.339 | 0.339 |
| EGNN | **0.170** | **0.170** |
| SE(3)-Transformer | 0.215 | 0.215 |

### Equivariance Stability Score (ESS, averaged over 7 molecules)

| Model | ESS (gap) | ESS (dipole) | Behaviour under rotation |
|---|---|---|---|
| Vanilla GNN | 1.000 | 1.000 | Flat — geometry-blind (coords ignored entirely) |
| Naive Geo GNN | ~0.997 | ~0.998 | Wobbles — uses raw coords non-equivariantly |
| EGNN | **≈1.000** | **≈1.000** | Rock-steady — provably E(3)-invariant readout |
| SE(3)-Transformer | **≈1.000** | **≈1.000** | Rock-steady — SE(3)-equivariant by construction |

ESS = 1 − (variance of predictions across rotations / inter-molecule variance normalizer). A perfectly equivariant model achieves ESS = 1.0.

---

## Stack

| Layer | Tool |
|---|---|
| Language | Python 3.10+ |
| Data | QM9 via PyTorch Geometric (auto-download) |
| Molecule parsing | RDKit |
| ML | PyTorch + PyTorch Geometric |
| Vanilla GNN | GIN (PyG built-in) |
| EGNN | Manual implementation — Satorras et al. 2021 |
| SE(3)-Transformer | e3nn |
| Backend | FastAPI + Uvicorn |
| Frontend | React + Vite + 3Dmol.js + Recharts |
| Training compute | Google Colab (free GPU) |
| Model weights | Hugging Face Hub (free) |

---

## Running Locally

### Prerequisites

```bash
pip install -r requirements.txt
```

PyTorch Geometric extras (adjust CUDA version as needed):

```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.2.2+cpu.html
```

### 1. Download model weights

Weights are stored on Hugging Face Hub. The backend pulls them automatically on first run, or place them manually in `weights/`:

```
weights/
  vanilla_best.pt
  egnn_best.pt
  se3_best.pt
  naive_geo_best.pt
  normalization.json
```

### 2. Start the FastAPI backend

```bash
uvicorn src.api.main:app --reload
```

API available at `http://127.0.0.1:8000`. Interactive docs at `/docs`.

### 3. Start the React frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend available at `http://localhost:5173`.

---

## API Endpoints

| Method | Path | Input | Output |
|---|---|---|---|
| POST | `/predict` | `{"smiles": "CCO"}` | All-3-model predictions (HOMO-LUMO gap + dipole) |
| POST | `/rotate_test` | `{"smiles": "CCO"}` | 24-rotation predictions + ESS per model |
| POST | `/explain` | `{"smiles": "CCO", "model": "egnn"}` | Per-atom importance scores |

---

## EGNN Equivariance: the Core Idea

Positions enter the EGNN **only** through squared pairwise distances `‖xᵢ − xⱼ‖²`. Distances are invariant to rotation, translation, and reflection. So every message and every node feature stays invariant, and the graph-level readout is **E(3)-invariant** — the predicted property is identical regardless of how the molecule is oriented. The coordinate update moves atoms along relative vectors scaled by those invariant weights, which makes the positions themselves transform **equivariantly**. No spherical harmonics, no Clebsch-Gordan coefficients — equivariance falls out of two simple choices.

See [`src/models/egnn.py`](src/models/egnn.py) for the full derivation in code comments.

---

## Training (Google Colab)

Open [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb) in Google Colab (free GPU tier). The notebook:

1. Clones the repo and installs dependencies.
2. Trains each model in sequence, logging MAE/RMSE per epoch to CSV.
3. Saves the best checkpoint and uploads it to Hugging Face Hub.

Do **not** train locally — QM9 on CPU takes hours per epoch.

---

## Project Layout

```
equivariant-molecule-net/
├── requirements.txt
├── verify_env.py              # confirm the full stack imports cleanly
├── src/
│   ├── data/
│   │   ├── qm9_loader.py      # QM9 load + 50k subset + train/val/test split
│   │   └── molecule_utils.py  # SMILES -> 3D PyG graph via RDKit
│   ├── models/
│   │   ├── vanilla_gnn.py     # GIN baseline (ignores coordinates)
│   │   ├── egnn.py            # EGNN from scratch (~200 lines)
│   │   └── se3_transformer.py # SE(3)-Transformer via e3nn
│   ├── train/
│   │   ├── train.py           # trains one model, saves best weights, logs CSV
│   │   └── config.py          # global seed, device, hyperparameters
│   ├── eval/
│   │   ├── metrics.py         # ESS + MAE/RMSE
│   │   ├── rotation_test.py   # 24-rotation stress test
│   │   └── atom_importance.py # integrated-gradients atom scoring
│   └── api/
│       └── main.py            # FastAPI app
├── notebooks/
│   └── train_colab.ipynb      # Colab training notebook
├── frontend/                  # React + Vite app
└── results/
    └── rotation_test_results.json
```

---

## License

MIT
