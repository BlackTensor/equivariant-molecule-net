import { useEffect, useState } from "react";
import { predict } from "../api/client.js";

// Task 7.3: live prediction panel.
//
// Calls the backend /predict endpoint whenever the SMILES changes and shows all
// three models' predictions side by side. The response shape (from src/api/main.py):
//   { smiles, num_atoms, predictions: {
//       <model>: { label, predictions: { <target>: { value, unit } } }, ...
//   } }
// Models are rendered in a fixed order so the columns stay stable across inputs.
const MODEL_ORDER = ["vanilla", "egnn", "se3"];

// Pretty-print a target key like "homo_lumo_gap" -> "Homo Lumo Gap".
function prettyTarget(key) {
  return key
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

export default function PredictionPanel({ smiles }) {
  const [data, setData] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | loading | ready | error
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!smiles) return;

    let cancelled = false;
    setStatus("loading");
    setError(null);

    predict(smiles)
      .then((res) => {
        if (cancelled) return;
        setData(res);
        setStatus("ready");
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err.message);
        setStatus("error");
      });

    return () => {
      cancelled = true;
    };
  }, [smiles]);

  // Order the models, falling back to whatever the backend actually returned.
  const models =
    data &&
    MODEL_ORDER.filter((m) => data.predictions[m]).map((m) => ({
      name: m,
      ...data.predictions[m],
    }));

  // Target keys come from any model (all three predict the same targets).
  const targets = models && models.length > 0
    ? Object.keys(models[0].predictions)
    : [];

  return (
    <section className="prediction-panel">
      <h2>Predictions</h2>

      {status === "loading" && <p>Predicting…</p>}
      {status === "error" && (
        <p className="err">Could not predict ({error}).</p>
      )}

      {status === "ready" && models && (
        <div className="prediction-cards">
          {models.map((model) => (
            <div key={model.name} className="prediction-card">
              <h3>{model.label}</h3>
              <dl>
                {targets.map((t) => {
                  const cell = model.predictions[t];
                  return (
                    <div key={t} className="prediction-row">
                      <dt>{prettyTarget(t)}</dt>
                      <dd>
                        {cell.value.toFixed(3)} <span className="unit">{cell.unit}</span>
                      </dd>
                    </div>
                  );
                })}
              </dl>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
