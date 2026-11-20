import React, { useEffect, useRef, useState } from "react";

export default function App() {
  const [modelsReady, setModelsReady] = useState(false);
  const [loadError, setLoadError] = useState("");
  const [samples, setSamples] = useState([]);
  const [selectedSample, setSelectedSample] = useState("");
  const [uploadedFile, setUploadedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [query, setQuery] = useState("");
  const [selectedRecipe, setSelectedRecipe] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const fileInputRef = useRef(null);
  const hasPhoto = Boolean(uploadedFile || selectedSample);

  useEffect(() => {
    let active = true;
    let timerId = null;

    async function checkHealth() {
      try {
        const res = await fetch("/api/health");
        const data = await res.json();
        if (!active) return;
        if (data.load_error) {
          setLoadError(data.load_error);
          return;
        }
        if (data.models_loaded) {
          setModelsReady(true);
          fetch("/api/samples")
            .then((r) => r.json())
            .then((s) => {
              if (active && s.samples?.length) setSamples(s.samples);
            })
            .catch(() => {});
          return;
        }
      } catch {
        // Server still binding; retry shortly
      }
      if (active) timerId = setTimeout(checkHealth, 500);
    }

    checkHealth();
    return () => {
      active = false;
      if (timerId) clearTimeout(timerId);
    };
  }, []);

  useEffect(() => {
    if (!selectedRecipe) return;
    const onKey = (e) => {
      if (e.key === "Escape") setSelectedRecipe(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selectedRecipe]);

  async function runPrediction(override = {}) {
    if (loading) return;
    const fileToUse = override.file !== undefined ? override.file : uploadedFile;
    const sampleToUse = override.sampleName !== undefined ? override.sampleName : selectedSample;
    if (!fileToUse && !sampleToUse) return;

    setLoading(true);
    setError("");
    try {
      const formData = new FormData();
      const qToUse = override.q !== undefined ? override.q : query;

      formData.append("query", qToUse);
      formData.append("compare", "true");
      formData.append("use_dish_hint", "true");

      if (fileToUse) {
        formData.append("image", fileToUse);
      } else {
        formData.append("sample_name", sampleToUse);
      }

      const res = await fetch("/api/predict", { method: "POST", body: formData });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server responded with ${res.status}`);
      }
      setResult(await res.json());
    } catch (err) {
      setError(err.message || "Prediction failed");
    } finally {
      setLoading(false);
    }
  }

  function handleSelectSample(item) {
    if (loading) return;
    setUploadedFile(null);
    setPreviewUrl("");
    setSelectedSample(item.name);
    setQuery(item.suggested_query);
    runPrediction({ file: null, sampleName: item.name, q: item.suggested_query });
  }

  function handleFileChange(e) {
    if (loading) return;
    const file = e.target.files?.[0];
    if (!file) return;
    setUploadedFile(file);
    setSelectedSample("");
    setPreviewUrl(URL.createObjectURL(file));
    setQuery("");
    runPrediction({ file, sampleName: "", q: "" });
  }

  const vision = result?.vision;
  const parsed = result?.retrieval?.parsed_query;
  const recipes = result?.retrieval?.results || [];

  return (
    <div className="container">
      <header className="topbar">
        <div>
          <h1 className="brand-title">Food Photo to Recipe Finder</h1>
          <p className="brand-sub">SWE015 Use Case 3 · Quantized INT8 ResNet18 + Grad-CAM + Quantized INT8 MiniLM</p>
        </div>
      </header>

      {!modelsReady ? (
        <section className="card loading-screen">
          <h2 className="section-title">
            {loadError ? "Failed to load models" : "Models are loading into memory..."}
          </h2>
          <p style={{ fontSize: "0.92rem", color: loadError ? "#B91C1C" : "#6B7280", maxWidth: "460px" }}>
            {loadError || "Initializing quantized ONNX ResNet18, Grad-CAM, and quantized MiniLM. The app will open automatically as soon as they are ready."}
          </p>
        </section>
      ) : (
        <>
          <div className="workspace-grid">
            {/* Left Card: Step 1 big upload box -> Step 2 active input view */}
            <section className="card">
              <span className="label-kicker">1. Dish Photo &amp; Request</span>

              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                disabled={loading}
                style={{ display: "none" }}
                onChange={handleFileChange}
              />

              {!hasPhoto ? (
                <>
                  <div
                    className={`upload-hero ${loading ? "is-disabled" : ""}`}
                    onClick={() => !loading && fileInputRef.current?.click()}
                  >
                    <div style={{ fontWeight: 700, fontSize: "1.08rem" }}>
                      Upload a dish photo to get started
                    </div>
                    <div style={{ fontSize: "0.86rem", color: "#6B7280", maxWidth: "320px" }}>
                      Choose any JPG or PNG food image from your device, or pick one of the sample dishes below.
                    </div>
                    <span className="pill-btn">Choose Photo</span>
                  </div>

                  <div style={{ marginTop: "16px" }}>
                    <div style={{ fontSize: "0.78rem", fontWeight: 600, color: "#6B7280" }}>
                      Or try a sample dish:
                    </div>
                    <div className="sample-list" style={{ marginBottom: 0 }}>
                      {samples.map((s) => (
                        <button
                          key={s.name}
                          type="button"
                          disabled={loading}
                          onClick={() => handleSelectSample(s)}
                          className="pill-outline"
                        >
                          <img src={s.image_url} alt={s.display_name} />
                          <span>{s.display_name}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                </>
              ) : (
                <>
                  <div className="sample-list">
                    {samples.map((s) => (
                      <button
                        key={s.name}
                        type="button"
                        disabled={loading}
                        onClick={() => handleSelectSample(s)}
                        className={`pill-outline ${!uploadedFile && selectedSample === s.name ? "active" : ""}`}
                      >
                        <img src={s.image_url} alt={s.display_name} />
                        <span>{s.display_name}</span>
                      </button>
                    ))}
                  </div>

                  <div
                    className={`upload-box ${loading ? "is-disabled" : ""}`}
                    onClick={() => !loading && fileInputRef.current?.click()}
                  >
                    <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
                      {(previewUrl || vision?.original_image) && (
                        <img
                          src={previewUrl || vision?.original_image}
                          alt="Dish preview"
                          className="upload-thumb"
                        />
                      )}
                      <div>
                        <div style={{ fontWeight: 600, fontSize: "0.88rem" }}>
                          {uploadedFile ? uploadedFile.name : selectedSample}
                        </div>
                        <div style={{ fontSize: "0.78rem", color: "#6B7280" }}>
                          Click to upload another JPG/PNG food photo
                        </div>
                      </div>
                    </div>
                    <span className="pill-outline">Change</span>
                  </div>

                  <textarea
                    className="query-input"
                    value={query}
                    disabled={loading}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Describe what you want, e.g. 'vegetarian, under 30 minutes'..."
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        runPrediction();
                      }
                    }}
                  />

                  <div className="controls-row">
                    <button
                      type="button"
                      className="pill-btn"
                      style={{ width: "100%" }}
                      onClick={() => runPrediction()}
                      disabled={loading}
                    >
                      {loading ? "Running..." : "Find Recipes"}
                    </button>
                  </div>

                  {parsed && (
                    <div className="tag-row">
                      {parsed.dish_hint && <span className="tag tag-violet">dish: {parsed.dish_hint.replace("_", " ")}</span>}
                      {parsed.max_time_min && <span className="tag">max_time &le; {parsed.max_time_min} min</span>}
                      {parsed.diet_constraints?.map((d) => (
                        <span key={d} className="tag">{d.replace("_", " ")}</span>
                      ))}
                    </div>
                  )}

                  {error && (
                    <p style={{ marginTop: "16px", color: "#B91C1C", fontSize: "0.86rem", fontWeight: 600 }}>
                      {error}
                    </p>
                  )}
                </>
              )}
            </section>

            {/* Right Card: Vision + Grad-CAM */}
            <section className="card">
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: "8px" }}>
                <div>
                  <span className="label-kicker">
                    {vision?.model_name || "ResNet18 Quantized (ONNX)"} · {vision?.precision || "INT8"}
                  </span>
                  <h2 className="dish-heading">
                    {loading && !vision ? "Analyzing photo..." : vision ? vision.display_name : "No photo selected"}
                  </h2>
                </div>
                {vision && (
                  <div style={{ display: "flex", gap: "8px" }}>
                    <span className="tag tag-violet">{(vision.confidence * 100).toFixed(1)}%</span>
                    <span className="tag">{vision.latency_ms} ms</span>
                  </div>
                )}
              </div>

              {!vision ? (
                <div className="empty-vision">
                  {loading
                    ? "Running INT8 ResNet18, Grad-CAM, and recipe retrieval..."
                    : "Upload or select a dish photo on the left to view the predicted class, Grad-CAM heatmap, and matching recipes."}
                </div>
              ) : (
                <>
                  <div className="cam-pair">
                    <div className="cam-frame">
                      <img src={vision.original_image} alt="Input dish" />
                      <div className="cam-caption">Input RGB (224&times;224)</div>
                    </div>
                    <div className="cam-frame">
                      <img src={vision.gradcam_image} alt="Grad-CAM heatmap" />
                      <div className="cam-caption">Grad-CAM (layer4[-1])</div>
                    </div>
                  </div>

                  {vision.top3 && (
                    <div className="prob-list">
                      {vision.top3.map((item) => (
                        <div key={item.dish} className="prob-row">
                          <span>{item.display_name}</span>
                          <div className="prob-track">
                            <div className="prob-fill" style={{ width: `${Math.max(2, item.confidence * 100)}%` }} />
                          </div>
                          <span style={{ textAlign: "right", fontWeight: 600 }}>
                            {(item.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </>
              )}
            </section>
          </div>

          {/* Top-5 Retrieved Recipes (Vertically Stacked Rows; click opens modal) */}
          {recipes.length > 0 && (
            <section className="section">
              <div className="section-head">
                <h2 className="section-title">Top-5 Matching Recipes</h2>
                {result?.retrieval && (
                  <span className="tag">
                    {result.retrieval.model_name} ({result.retrieval.precision}) · {result.retrieval.latency_ms} ms
                  </span>
                )}
              </div>

              <div className="recipes-list">
                {recipes.map((rec, idx) => (
                  <article
                    key={rec.id}
                    role="button"
                    tabIndex={0}
                    className="card recipe-row-btn"
                    onClick={() => setSelectedRecipe(rec)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        setSelectedRecipe(rec);
                      }
                    }}
                  >
                    <div style={{ flex: 1 }}>
                      <span className="label-kicker" style={{ marginBottom: 0 }}>
                        #{idx + 1} · {rec.dish_category.replace("_", " ")}
                      </span>
                      <h3 className="recipe-title">{rec.title}</h3>
                      <p style={{ fontSize: "0.88rem", color: "#4B5563" }}>{rec.description}</p>
                      <p className="recipe-meta-line">
                        {[
                          `${rec.prep_time_min} min`,
                          ...(rec.diet_tags || []).map((t) => t.replace("_", " ")),
                        ].join(" · ")}
                      </p>
                    </div>

                    <span className="tag tag-violet" style={{ flexShrink: 0 }}>
                      Score {rec.score.toFixed(3)}
                    </span>
                  </article>
                ))}
              </div>
            </section>
          )}

          {/* Side-by-Side Model Comparison (Compact 2-Table Grid) */}
          {result?.comparison && (
            <section className="section">
              <div className="section-head">
                <h2 className="section-title">Side-by-Side Model Comparison</h2>
              </div>

              <div className="compare-tables-grid">
                <div className="card">
                  <span className="label-kicker">Vision Ladder</span>
                  <table className="compare-table">
                    <thead>
                      <tr>
                        <th>Model</th>
                        <th>Prec</th>
                        <th>Prediction</th>
                        <th>Latency</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.comparison.vision.map((v) => (
                        <tr key={v.model_name} className={v.precision === "INT8" ? "row-int8" : ""}>
                          <td>{v.model_name}</td>
                          <td>{v.precision}</td>
                          <td>{v.display_name} ({(v.confidence * 100).toFixed(1)}%)</td>
                          <td>{v.latency_ms} ms</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="card">
                  <span className="label-kicker">Retrieval Ladder</span>
                  <table className="compare-table">
                    <thead>
                      <tr>
                        <th>Model</th>
                        <th>Prec</th>
                        <th>#1 Match (Score)</th>
                        <th>Latency</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.comparison.retrieval.map((r) => {
                        const top1 = r.top_recipes?.[0];
                        return (
                          <tr key={r.model_name} className={r.precision === "INT8" ? "row-int8" : ""}>
                            <td>{r.model_name}</td>
                            <td>{r.precision}</td>
                            <td>{top1 ? `${top1.title} (${top1.score.toFixed(2)})` : "—"}</td>
                            <td>{r.latency_ms} ms</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            </section>
          )}
        </>
      )}

      {/* Recipe Details Modal */}
      {selectedRecipe && (
        <div className="modal-backdrop" onClick={() => setSelectedRecipe(null)}>
          <div
            className="modal-panel"
            role="dialog"
            aria-modal="true"
            aria-labelledby="modal-recipe-title"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="modal-header">
              <div>
                <span className="label-kicker">{selectedRecipe.dish_category.replace("_", " ")}</span>
                <h2 id="modal-recipe-title" className="dish-heading">{selectedRecipe.title}</h2>
              </div>
              <button type="button" className="pill-outline" onClick={() => setSelectedRecipe(null)}>
                Close
              </button>
            </div>

            <p style={{ fontSize: "0.92rem", color: "#4B5563" }}>{selectedRecipe.description}</p>

            <div className="tag-row">
              <span className="tag tag-violet">Score {selectedRecipe.score.toFixed(3)}</span>
              <span className="tag">{selectedRecipe.prep_time_min} min</span>
              {selectedRecipe.diet_tags?.map((t) => (
                <span key={t} className="tag">{t.replace("_", " ")}</span>
              ))}
            </div>

            <div className="modal-section">
              <strong>Ingredients ({selectedRecipe.ingredients?.length || 0})</strong>
              <ul>
                {selectedRecipe.ingredients?.map((ing, i) => <li key={i}>{ing}</li>)}
              </ul>
            </div>

            <div className="modal-section">
              <strong>Preparation Steps</strong>
              <ol>
                {selectedRecipe.steps?.map((st, i) => <li key={i}>{st}</li>)}
              </ol>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
