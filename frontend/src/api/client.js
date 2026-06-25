// API client for the EquiDrug FastAPI backend (src/api/main.py).
//
// One thin module that mirrors the three backend endpoints exactly:
//   POST /predict       { smiles }                  -> predictions for all 3 models
//   POST /rotate_test   { smiles }                  -> 24-rotation predictions + ESS
//   POST /explain       { smiles, model, target }   -> per-atom importance
//
// The base URL is read from VITE_API_BASE_URL (see .env.example) and defaults to
// the local Uvicorn dev server. Port 5173 (the Vite dev server) is allow-listed
// by the backend's CORS middleware, so these calls work in local development.

export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

/**
 * POST JSON to `path` and return the parsed JSON response.
 * Throws an Error carrying the backend's `detail` message on a non-2xx status,
 * so callers can surface a useful message (e.g. "invalid SMILES").
 */
async function postJson(path, body) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    // FastAPI errors come back as { detail: "..." }. Fall back to status text.
    let detail = response.statusText;
    try {
      const data = await response.json();
      if (data && data.detail) detail = data.detail;
    } catch {
      /* response had no JSON body — keep the status text */
    }
    throw new Error(`API ${response.status}: ${detail}`);
  }

  return response.json();
}

/** GET the root health/info endpoint: { name, models, status }. */
export async function getHealth() {
  const response = await fetch(`${API_BASE_URL}/`);
  if (!response.ok) throw new Error(`API ${response.status}: ${response.statusText}`);
  return response.json();
}

/**
 * 3D structure of a molecule as a V2000 MOL/SDF block (RDKit-embedded).
 * Returns { smiles, num_atoms, sdf } — feed `sdf` straight into 3Dmol.js.
 */
export function getStructure(smiles) {
  return postJson("/structure", { smiles });
}

/** All three models' predictions (physical units) for one molecule. */
export function predict(smiles) {
  return postJson("/predict", { smiles });
}

/** 24-rotation predictions + ESS per model for one molecule. */
export function rotateTest(smiles) {
  return postJson("/rotate_test", { smiles });
}

/** Per-atom importance for a given molecule, model, and target property. */
export function explain(smiles, model, target) {
  const body = target ? { smiles, model, target } : { smiles, model };
  return postJson("/explain", body);
}
