import { useEffect, useState } from "react";
import { explain, getHealth } from "./api/client.js";
import MoleculeViewer from "./components/MoleculeViewer.jsx";
import PredictionPanel from "./components/PredictionPanel.jsx";
import RotationChart from "./components/RotationChart.jsx";
import CompareView from "./components/CompareView.jsx";

// Models / targets the /explain endpoint accepts (mirrors EXPLAIN_MODELS and
// CONFIG.data.targets in the backend). Stable project constants.
const EXPLAIN_MODELS = [
  { value: "vanilla", label: "Vanilla GNN" },
  { value: "naive_geo", label: "Naive Geometric" },
  { value: "egnn", label: "EGNN" },
  { value: "se3", label: "SE(3)-Transformer" },
];
const EXPLAIN_TARGETS = [
  { value: "homo_lumo_gap", label: "HOMO-LUMO gap" },
  { value: "dipole_moment", label: "Dipole moment" },
];

// App shell. Tasks 7.1–7.4 added the connectivity check, 3D viewer, prediction
// panel, and rotation chart. Task 7.5 adds the atom-importance heatmap overlay
// (controls here; coloring happens inside MoleculeViewer).
export default function App() {
  const [health, setHealth] = useState(null);
  const [error, setError] = useState(null);

  // Which view is active: a single molecule (with the importance overlay) or
  // the side-by-side two-molecule comparison (task 7.6).
  const [view, setView] = useState("single"); // "single" | "compare"

  // The SMILES currently rendered, plus the draft text in the input box.
  const [smiles, setSmiles] = useState("CCO"); // ethanol by default
  const [draft, setDraft] = useState("CCO");

  // Atom-importance overlay state (task 7.5).
  const [overlay, setOverlay] = useState(false);
  const [explainModel, setExplainModel] = useState("egnn");
  const [explainTarget, setExplainTarget] = useState("dipole_moment");
  const [explainData, setExplainData] = useState(null); // full /explain response
  const [explainStatus, setExplainStatus] = useState("idle"); // idle|loading|ready|error
  const [explainError, setExplainError] = useState(null);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch((err) => setError(err.message));
  }, []);

  // Fetch per-atom importance whenever the overlay is on and its inputs change.
  // Turning the overlay off (or changing SMILES) clears the stale heatmap.
  useEffect(() => {
    if (!overlay || !smiles) {
      setExplainData(null);
      setExplainStatus("idle");
      return;
    }
    let cancelled = false;
    setExplainStatus("loading");
    setExplainError(null);
    explain(smiles, explainModel, explainTarget)
      .then((res) => {
        if (cancelled) return;
        setExplainData(res);
        setExplainStatus("ready");
      })
      .catch((err) => {
        if (cancelled) return;
        setExplainData(null);
        setExplainError(err.message);
        setExplainStatus("error");
      });
    return () => {
      cancelled = true;
    };
  }, [overlay, smiles, explainModel, explainTarget]);

  function onSubmit(event) {
    event.preventDefault();
    const next = draft.trim();
    if (next) setSmiles(next);
  }

  const importance =
    overlay && explainData ? explainData.importance_normalized : null;

  // Backend reachability for the status pill: connected once /health resolves.
  const connected = Boolean(health) && !error;

  return (
    <main className="app">
      <header className="site-header">
        <span
          className={`status-pill ${connected ? "is-connected" : "is-disconnected"}`}
          title={connected ? "Backend connected" : "Backend unreachable"}
        >
          <span className="status-dot" aria-hidden="true">●</span>
          {connected ? "Connected" : "Disconnected"}
        </span>
        <h1 className="brand-title">EquiDrug</h1>
        <p className="brand-tagline">Interactive Geometric AI for Drug Discovery</p>
        <p className="brand-subtitle">
          Quantifying symmetry robustness of molecular neural networks under 3D rotation
        </p>
        <p className="brand-author">Shayan Ansari · 2026</p>
      </header>

      <div className="view-toggle">
        <button
          type="button"
          className={view === "single" ? "active" : ""}
          onClick={() => setView("single")}
        >
          Single molecule
        </button>
        <button
          type="button"
          className={view === "compare" ? "active" : ""}
          onClick={() => setView("compare")}
        >
          Compare two
        </button>
      </div>

      {view === "compare" ? (
        <CompareView />
      ) : (
        <>
      <form className="smiles-form" onSubmit={onSubmit}>
        <label htmlFor="smiles-input">Molecule (SMILES)</label>
        <div className="smiles-row">
          <input
            id="smiles-input"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder="e.g. CCO, c1ccccc1, CC#N"
          />
          <button type="submit">Render</button>
        </div>
      </form>

      <section className="importance-controls">
        <label className="overlay-toggle">
          <input
            type="checkbox"
            checked={overlay}
            onChange={(e) => setOverlay(e.target.checked)}
          />
          Atom-importance heatmap
        </label>

        {overlay && (
          <div className="importance-selects">
            <label>
              Model
              <select
                value={explainModel}
                onChange={(e) => setExplainModel(e.target.value)}
              >
                {EXPLAIN_MODELS.map((m) => (
                  <option key={m.value} value={m.value}>
                    {m.label}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Property
              <select
                value={explainTarget}
                onChange={(e) => setExplainTarget(e.target.value)}
              >
                {EXPLAIN_TARGETS.map((t) => (
                  <option key={t.value} value={t.value}>
                    {t.label}
                  </option>
                ))}
              </select>
            </label>
          </div>
        )}

        {overlay && explainStatus === "loading" && <span>Computing importance…</span>}
        {overlay && explainStatus === "error" && (
          <span className="err">Could not explain ({explainError}).</span>
        )}
        {overlay && explainStatus === "ready" && explainData && (
          <span className="importance-meta">
            {explainData.label}: prediction{" "}
            <strong>{explainData.prediction.toFixed(3)}</strong>
          </span>
        )}
      </section>

      <MoleculeViewer smiles={smiles} importance={importance} />
      <PredictionPanel smiles={smiles} />
      <RotationChart smiles={smiles} />
        </>
      )}

      <footer className="site-footer">
        <span>Designed &amp; Built by Shayan Ansari · 2026</span>
      </footer>
    </main>
  );
}
