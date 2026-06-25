import { useState } from "react";
import MoleculeViewer from "./MoleculeViewer.jsx";
import PredictionPanel from "./PredictionPanel.jsx";
import RotationChart from "./RotationChart.jsx";

// Task 7.6: side-by-side "compare two molecules" view.
//
// Each column owns its own SMILES input and reuses the same self-contained
// panels as the single-molecule view (3D viewer, predictions, rotation chart),
// so you can eyeball two molecules at once — e.g. aspirin vs ibuprofen, the
// shareable LinkedIn comparison. Defaults are preloaded for that exact demo.
//
// Note: the models were trained on QM9 (≤9 heavy atoms), so on drug-sized
// molecules the predicted *values* are extrapolations — but the rotation/ESS
// story (equivariant = flat, non-equivariant = wobbly) still holds, which is
// the whole point of the comparison.

// One column: its own SMILES box + the three reusable panels.
function MoleculeColumn({ title, initialSmiles }) {
  const [smiles, setSmiles] = useState(initialSmiles);
  const [draft, setDraft] = useState(initialSmiles);

  function onSubmit(event) {
    event.preventDefault();
    const next = draft.trim();
    if (next) setSmiles(next);
  }

  return (
    <div className="compare-col">
      <h3 className="compare-col-title">{title}</h3>
      <form className="smiles-form" onSubmit={onSubmit}>
        <div className="smiles-row">
          <input
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder="SMILES"
            aria-label={`${title} SMILES`}
          />
          <button type="submit">Render</button>
        </div>
      </form>

      <MoleculeViewer smiles={smiles} />
      <PredictionPanel smiles={smiles} />
      <RotationChart smiles={smiles} />
    </div>
  );
}

export default function CompareView() {
  return (
    <section className="compare-view">
      <p className="compare-blurb">
        Two molecules, the same three models, side by side. Compare predictions
        and rotation stability (ESS) at a glance.
      </p>
      <div className="compare-grid">
        <MoleculeColumn title="Molecule A" initialSmiles="CC(=O)Oc1ccccc1C(=O)O" />
        <MoleculeColumn title="Molecule B" initialSmiles="CC(C)Cc1ccc(cc1)C(C)C(=O)O" />
      </div>
    </section>
  );
}
