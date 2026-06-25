import { useEffect, useRef, useState } from "react";
import * as $3Dmol from "3dmol";
import { getStructure } from "../api/client.js";

// Task 7.2 + 7.5: 3D molecule viewer with an optional atom-importance heatmap.
//
// Structure comes from /structure (the same RDKit geometry the models run on)
// and is rendered with 3Dmol.js (drag-rotate / scroll-zoom out of the box).
//
// Task 7.5 overlay: when an `importance` array (one [0,1] value per atom, in the
// SDF's atom order) is passed in, each atom sphere is colored by a blue→white→red
// heat gradient (red = most important). Hovering an atom shows its element and
// score. Pass `importance={null}` to fall back to normal CPK ball-and-stick.

// Blue (low) -> white (mid) -> red (high) heatmap color for a value in [0,1].
function heatColor(t) {
  const v = Math.max(0, Math.min(1, t));
  const lerp = (a, b, k) => Math.round(a + (b - a) * k);
  let r, g, b;
  if (v < 0.5) {
    const k = v / 0.5; // blue -> white
    r = lerp(43, 255, k);
    g = lerp(123, 255, k);
    b = lerp(182, 255, k);
  } else {
    const k = (v - 0.5) / 0.5; // white -> red
    r = lerp(255, 215, k);
    g = lerp(255, 25, k);
    b = lerp(255, 28, k);
  }
  const hex = (n) => n.toString(16).padStart(2, "0");
  return `#${hex(r)}${hex(g)}${hex(b)}`;
}

export default function MoleculeViewer({ smiles, importance = null }) {
  const containerRef = useRef(null);
  const viewerRef = useRef(null);
  const modelLoadedRef = useRef(false);
  // Keep the latest importance in a ref so the structure-fetch effect (keyed on
  // smiles only) always styles with the current overlay, without re-fetching.
  const importanceRef = useRef(importance);
  importanceRef.current = importance;

  const [status, setStatus] = useState("idle"); // idle | loading | ready | error
  const [error, setError] = useState(null);
  const [info, setInfo] = useState(null); // { smiles, num_atoms }

  // Apply ball-and-stick, optionally tinting atoms by the importance heatmap.
  function applyStyle(viewer) {
    const model = viewer.getModel();
    if (!model) return;
    const atoms = model.selectedAtoms({});
    const imp = importanceRef.current;

    if (imp && imp.length === atoms.length) {
      // Heatmap overlay: color each atom's sphere by its importance score.
      // We select by the atom's own serial so the color lands on exactly the
      // atom we read importance[i] for (atoms come back in index order).
      atoms.forEach((atom, i) => {
        atom.importanceVal = imp[i]; // stashed for the hover label
        viewer.setStyle(
          { serial: atom.serial },
          { stick: { radius: 0.15 }, sphere: { scale: 0.38, color: heatColor(imp[i]) } }
        );
      });
      // Hover shows element + score so you can read the heatmap precisely.
      viewer.setHoverable(
        {},
        true,
        (atom, vwr) => {
          if (!atom.label) {
            const score = atom.importanceVal != null ? atom.importanceVal.toFixed(2) : "?";
            atom.label = vwr.addLabel(`${atom.elem} • ${score}`, {
              position: atom,
              backgroundColor: "#1a1a2e",
              fontColor: "white",
              fontSize: 12,
            });
            vwr.render();
          }
        },
        (atom, vwr) => {
          if (atom.label) {
            vwr.removeLabel(atom.label);
            delete atom.label;
            vwr.render();
          }
        }
      );
    } else {
      // Plain CPK ball-and-stick (no overlay).
      viewer.setStyle({}, { stick: { radius: 0.15 }, sphere: { scale: 0.25 } });
      viewer.setHoverable({}, false, () => {}, () => {});
    }
    viewer.render();
  }

  // Create the 3Dmol viewer once, when the container mounts.
  useEffect(() => {
    if (!containerRef.current) return;
    const viewer = $3Dmol.createViewer(containerRef.current, {
      backgroundColor: "#111111",
    });
    viewerRef.current = viewer;
    return () => {
      viewer.clear();
      viewerRef.current = null;
      modelLoadedRef.current = false;
    };
  }, []);

  // Fetch + (re)draw the structure whenever the SMILES changes.
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !smiles) return;

    let cancelled = false;
    setStatus("loading");
    setError(null);
    modelLoadedRef.current = false;

    getStructure(smiles)
      .then((res) => {
        if (cancelled) return;
        viewer.clear();
        viewer.addModel(res.sdf, "sdf");
        modelLoadedRef.current = true;
        applyStyle(viewer); // honors the current importance overlay
        viewer.zoomTo();
        viewer.render();
        setInfo({ smiles: res.smiles, num_atoms: res.num_atoms });
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

  // Re-style (without refetching) when the importance overlay changes.
  useEffect(() => {
    const viewer = viewerRef.current;
    if (viewer && modelLoadedRef.current) applyStyle(viewer);
  }, [importance]);

  const overlayActive = !!(importance && info && importance.length === info.num_atoms);

  return (
    <section className="viewer-panel">
      <h2>3D structure</h2>
      <div
        ref={containerRef}
        className="viewer-canvas"
        style={{ position: "relative", width: "100%", height: "360px" }}
      />
      {status === "loading" && <p>Building 3D structure…</p>}
      {status === "ready" && info && (
        <p className="viewer-meta">
          <code>{info.smiles}</code> — {info.num_atoms} atoms (incl. H). Drag to
          rotate, scroll to zoom.
        </p>
      )}
      {status === "error" && (
        <p className="err">Could not render molecule ({error}).</p>
      )}

      {overlayActive && (
        <div className="heatmap-legend">
          <span>Atom importance:</span>
          <span className="legend-label">low</span>
          <span
            className="legend-bar"
            style={{
              background: `linear-gradient(to right, ${heatColor(0)}, ${heatColor(0.5)}, ${heatColor(1)})`,
            }}
          />
          <span className="legend-label">high</span>
        </div>
      )}
    </section>
  );
}
