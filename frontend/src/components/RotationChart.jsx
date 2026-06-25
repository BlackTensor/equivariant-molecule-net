import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { rotateTest } from "../api/client.js";

// Task 7.4: the flagship "wobbly-vs-flat" rotation chart.
//
// Calls /rotate_test (24 rotations, 15° steps) and plots prediction vs rotation
// angle for all FOUR models on one axis. The story the chart tells:
//   * vanilla    — ignores 3D coords, so it's flat (a trivial kind of invariance)
//   * naive_geo  — geometry-aware but NOT equivariant → predictions WOBBLE (low ESS)
//   * egnn / se3 — equivariant → flat lines that don't move under rotation (ESS ≈ 1)
// The Equivariance Stability Score (ESS) for each model is shown per line.
//
// Response shape (src/api/main.py):
//   { smiles, angles_deg: [0,15,...,345],
//     models: { <name>: { label, predictions: { <target>: [24 vals] }, ess: { <target>: num } } } }
const MODEL_ORDER = ["vanilla", "naive_geo", "egnn", "se3"];

// Distinct colors so the four lines are easy to tell apart. naive_geo is the
// red "wobbler" we want to draw the eye to; the equivariant pair are greens.
// Brightened for the dark theme so all four lines pop against #161616.
const MODEL_COLORS = {
  vanilla: "#8a8a8a",
  naive_geo: "#ff5a5c",
  egnn: "#3fb950",
  se3: "#4d9fff",
};

// Shared dark styling for the Recharts axes/grid/tooltip.
const AXIS_TICK = { fill: "#999999", fontSize: 12 };
const AXIS_LABEL_FILL = "#888888";
const GRID_STROKE = "#2a2a2a";
const AXIS_STROKE = "#3a3a3a";

export default function RotationChart({ smiles }) {
  const [data, setData] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | loading | ready | error
  const [error, setError] = useState(null);
  const [target, setTarget] = useState(null); // selected property key

  useEffect(() => {
    if (!smiles) return;

    let cancelled = false;
    setStatus("loading");
    setError(null);

    rotateTest(smiles)
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

  // Models present, in our preferred display order.
  const models = useMemo(
    () =>
      data
        ? MODEL_ORDER.filter((m) => data.models[m]).map((m) => ({
            name: m,
            ...data.models[m],
          }))
        : [],
    [data]
  );

  // Available targets come from any model; default to the first one.
  const targets = models.length > 0 ? Object.keys(models[0].predictions) : [];
  const activeTarget = target && targets.includes(target) ? target : targets[0];

  // Reshape the per-model arrays into Recharts rows keyed by angle:
  //   [{ angle: 0, vanilla: .., naive_geo: .., egnn: .., se3: .. }, ...]
  //
  // Each model's series is CENTERED at 0 by subtracting its own mean prediction.
  // The untrained naive_geo model predicts on a wildly different absolute scale,
  // which would otherwise dominate the y-axis and flatten the trained models'
  // lines into a single squashed band. Centering removes the absolute offset so
  // the chart shows only the *shape* of each line — i.e. how much it wobbles
  // under rotation — which is exactly the equivariant-vs-not comparison.
  const chartData = useMemo(() => {
    if (!data || !activeTarget) return [];

    // Per-model mean over the 24 rotation angles for the active target.
    const means = {};
    for (const model of models) {
      const vals = model.predictions[activeTarget];
      means[model.name] = vals.reduce((a, b) => a + b, 0) / vals.length;
    }

    return data.angles_deg.map((angle, i) => {
      const row = { angle };
      for (const model of models) {
        row[model.name] = model.predictions[activeTarget][i] - means[model.name];
      }
      return row;
    });
  }, [data, models, activeTarget]);

  return (
    <section className="rotation-panel">
      <h2>Rotation stress test</h2>
      <p className="rotation-blurb">
        Prediction vs. rotation angle (0–345°). Equivariant models stay flat;
        the non-equivariant one wobbles. Higher ESS = more stable under rotation.
      </p>

      {status === "loading" && <p>Running 24-rotation test…</p>}
      {status === "error" && (
        <p className="err">Could not run rotation test ({error}).</p>
      )}

      {status === "ready" && data && (
        <>
          {targets.length > 1 && (
            <div className="target-toggle">
              {targets.map((t) => (
                <button
                  key={t}
                  type="button"
                  className={t === activeTarget ? "active" : ""}
                  onClick={() => setTarget(t)}
                >
                  {prettyTarget(t)}
                </button>
              ))}
            </div>
          )}

          <div className="rotation-chart">
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={chartData} margin={{ top: 8, right: 16, bottom: 56, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={GRID_STROKE} />
                <XAxis
                  dataKey="angle"
                  type="number"
                  domain={[0, 345]}
                  ticks={[0, 45, 90, 135, 180, 225, 270, 315, 345]}
                  tick={AXIS_TICK}
                  stroke={AXIS_STROKE}
                  label={{ value: "rotation angle (°)", position: "insideBottom", offset: -12, fill: AXIS_LABEL_FILL }}
                />
                <YAxis
                  tick={AXIS_TICK}
                  stroke={AXIS_STROKE}
                  label={{ value: `Δ from mean (${activeTarget === "homo_lumo_gap" ? "eV" : "Debye"})`, angle: -90, position: "insideLeft", fill: AXIS_LABEL_FILL }}
                  width={70}
                  tickFormatter={(v) => v.toFixed(2)}
                />
                <Tooltip
                  formatter={(v) => Number(v).toFixed(4)}
                  labelFormatter={(a) => `${a}°`}
                  contentStyle={{ background: "#262626", border: "1px solid #3a3a3a", borderRadius: 6, color: "#e8e8e8" }}
                  labelStyle={{ color: "#999999" }}
                  itemStyle={{ color: "#e8e8e8" }}
                  cursor={{ stroke: "#3a3a3a" }}
                />
                <Legend
                  verticalAlign="bottom"
                  wrapperStyle={{ paddingTop: 20, color: "#c8c8c8" }}
                />
                {models.map((model) => (
                  <Line
                    key={model.name}
                    type="monotone"
                    dataKey={model.name}
                    name={model.label}
                    stroke={MODEL_COLORS[model.name]}
                    // Emphasize the non-equivariant "bad" model (red) with a
                    // thicker stroke so its wobble stands out from the rest.
                    strokeWidth={model.name === "naive_geo" ? 3 : 1.5}
                    dot={false}
                    isAnimationActive={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Explicit ESS readout per model, so the number is legible even if
              the legend is busy. */}
          <table className="ess-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>ESS ({prettyTarget(activeTarget)})</th>
              </tr>
            </thead>
            <tbody>
              {models.map((model) => (
                <tr key={model.name}>
                  <td>
                    <span
                      className="swatch"
                      style={{ background: MODEL_COLORS[model.name] }}
                    />
                    {model.label}
                  </td>
                  <td className="ess-value">{model.ess[activeTarget].toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </section>
  );
}

// "homo_lumo_gap" -> "Homo Lumo Gap"
function prettyTarget(key) {
  return key
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

// Best-effort y-axis unit label, read from the first model's units if present.
// (/rotate_test doesn't carry units, so this just labels the target name.)
function unitLabel(models, target) {
  if (!target) return "";
  return prettyTarget(target);
}
