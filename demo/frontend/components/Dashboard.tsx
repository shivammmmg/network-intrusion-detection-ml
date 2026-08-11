"use client";

import { useEffect, useMemo, useState } from "react";

import { api, ApiError } from "../lib/api";
import { cloneRecord, hasDisagreement, isModified, percentage, scenarioLabel } from "../lib/presentation";
import type { DemoExample, FeatureRecord, HealthResponse, ModelMetadata, PredictionResult, SchemaResponse } from "../lib/types";
import { PredictionCard } from "./PredictionCard";

type View = "overview" | "live" | "compare";

const fallbackMessage = "Model service is starting or unavailable. Start the FastAPI backend and refresh this page.";

function formValue(value: FeatureRecord[string] | undefined): string {
  return typeof value === "number" && Number.isNaN(value) ? "" : String(value ?? "");
}

export function Dashboard() {
  const [view, setView] = useState<View>("overview");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [models, setModels] = useState<ModelMetadata[]>([]);
  const [schema, setSchema] = useState<SchemaResponse | null>(null);
  const [examples, setExamples] = useState<DemoExample[]>([]);
  const [selectedExample, setSelectedExample] = useState<DemoExample | undefined>();
  const [record, setRecord] = useState<FeatureRecord>({});
  const [modelId, setModelId] = useState("random_forest");
  const [singleResult, setSingleResult] = useState<PredictionResult | null>(null);
  const [compareResults, setCompareResults] = useState<PredictionResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void Promise.all([api.health(), api.models(), api.schema(), api.examples()])
      .then(([nextHealth, nextModels, nextSchema, nextExamples]) => {
        setHealth(nextHealth);
        setModels(nextModels.models);
        setSchema(nextSchema);
        setExamples(nextExamples.examples);
        const initial = nextExamples.examples[0];
        if (initial) {
          setSelectedExample(initial);
          setRecord(cloneRecord(initial.record));
        }
      })
      .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : fallbackMessage))
      .finally(() => setLoading(false));
  }, []);

  const modified = useMemo(() => isModified(record, selectedExample), [record, selectedExample]);
  const serviceReady = Boolean(health?.ready);
  const selectedModel = models.find((model) => model.id === modelId);

  function chooseExample(example: DemoExample) {
    setSelectedExample(example);
    setRecord(cloneRecord(example.record));
    setSingleResult(null);
    setCompareResults([]);
    setError(null);
  }

  function updateValue(name: string, value: string, type: string) {
    const parsed: string | number = type === "string" ? value : value === "" ? Number.NaN : Number(value);
    setRecord((current) => ({ ...current, [name]: parsed }));
    setSingleResult(null);
    setCompareResults([]);
  }

  async function runSingle() {
    if (!serviceReady) return;
    setPredicting(true);
    setError(null);
    try {
      setSingleResult(await api.predict(modelId, record));
      setCompareResults([]);
    } catch (reason) {
      setError(reason instanceof ApiError ? reason.message : "Prediction could not be completed.");
    } finally {
      setPredicting(false);
    }
  }

  async function runCompare() {
    if (!serviceReady) return;
    setPredicting(true);
    setError(null);
    try {
      setCompareResults((await api.predictAll(record)).results);
      setSingleResult(null);
    } catch (reason) {
      setError(reason instanceof ApiError ? reason.message : "Comparison could not be completed.");
    } finally {
      setPredicting(false);
    }
  }

  return (
    <main className="shell">
      <header className="topbar">
        <div className="brand"><span className="brand-mark">◈</span><div><p>EECS 3404</p><h1>Intrusion Detection Lab</h1></div></div>
        <div className={`service-status ${serviceReady ? "ready" : "offline"}`}><span />{serviceReady ? "Frozen model service ready" : "Model service unavailable"}</div>
      </header>

      <nav className="nav-tabs" aria-label="Demo sections">
        {(["overview", "live", "compare"] as View[]).map((item) => (
          <button key={item} className={view === item ? "active" : ""} onClick={() => setView(item)}>{item === "live" ? "Live Detection" : item === "compare" ? "Compare Models" : "Overview"}</button>
        ))}
      </nav>

      {loading && <section className="loading-panel"><div className="spinner" />Loading frozen model metadata…</section>}
      {!loading && error && <section className="alert error" role="alert"><b>Unable to complete request.</b> {error}</section>}
      {!loading && !serviceReady && <section className="alert warning">{fallbackMessage}</section>}

      {!loading && view === "overview" && (
        <section className="overview-grid">
          <div className="hero-card">
            <p className="eyebrow">Frozen, read-only course project interface</p>
            <h2>Explainable and Drift-Aware Machine Learning for Network Intrusion Detection</h2>
            <p>Classify UNSW-NB15 network traffic as <b>Normal</b> or <b>Attack</b> with four finalized models.</p>
            <div className="hero-actions"><button className="primary-button" onClick={() => setView("compare")}>Compare all models</button><button className="secondary-button" onClick={() => setView("live")}>Run one model</button></div>
          </div>
          <aside className="method-card">
            <p className="eyebrow">Method safeguards</p>
            <ul><li>39 active primary-model features</li><li>TTL leakage-prone fields excluded</li><li>Validation-selected locked thresholds</li><li>Frozen verified artifacts; no retraining</li></ul>
            <p className="source-sha">Frozen source: {health?.frozen_source_sha.slice(0, 8) ?? "loading"}</p>
          </aside>
          <div className="model-strip">{models.map((model) => <div key={model.id}><span>{model.family}</span><b>{model.display_name}</b><small>{model.transformed_feature_count} transformed features</small></div>)}</div>
        </section>
      )}

      {!loading && (view === "live" || view === "compare") && schema && (
        <section className="workspace">
          <aside className="input-panel">
            <div className="panel-heading"><div><p className="eyebrow">Curated held-out examples</p><h2>Choose a record</h2></div>{modified && <span className="modified-badge">Modified example</span>}</div>
            <div className="example-list">{examples.map((example) => <button key={example.sample_index} className={selectedExample?.sample_index === example.sample_index && !modified ? "example-card selected" : "example-card"} onClick={() => chooseExample(example)}><b>{scenarioLabel[example.scenario] ?? example.scenario}</b><span>Known held-out label: {example.evaluation_metadata.known_held_out_label ? "Attack" : "Normal"}</span></button>)}</div>
            {selectedExample && !modified && <p className="metadata-note">Selected: {scenarioLabel[selectedExample.scenario] ?? selectedExample.scenario}. Known held-out label is evaluation metadata only; it is never sent to a model.</p>}
            {selectedExample && modified && <p className="metadata-note warning-text">Ground-truth comparison disabled because this record has been edited.</p>}

            <div className="quick-fields"><p className="eyebrow">Primary visible fields</p>{["proto", "service", "state"].map((name) => <label key={name}>{name}<select value={formValue(record[name])} onChange={(event) => updateValue(name, event.target.value, "string")}>{schema.categorical_categories[name]?.map((option) => <option key={option} value={option}>{option}</option>)}</select></label>)}</div>

            <details className="advanced-editor"><summary>Advanced: View / edit all 39 features</summary><div className="field-grid">{schema.fields.map((field) => <label key={field.name}>{field.name}{field.kind === "categorical" ? <select value={formValue(record[field.name])} onChange={(event) => updateValue(field.name, event.target.value, field.type)}>{schema.categorical_categories[field.name]?.map((option) => <option key={option} value={option}>{option}</option>)}</select> : <input aria-label={field.name} type="number" min={field.non_negative ? 0 : undefined} step={field.type === "integer" ? 1 : "any"} value={formValue(record[field.name])} onChange={(event) => updateValue(field.name, event.target.value, field.type)} />}</label>)}</div></details>
            <button className="reset-button" disabled={!selectedExample || !modified} onClick={() => selectedExample && chooseExample(selectedExample)}>Reset to selected example</button>
          </aside>

          <div className="result-panel">
            {view === "live" ? <><div className="panel-heading"><div><p className="eyebrow">Single model mode</p><h2>Live detection</h2></div></div><div className="action-row"><label>Model<select value={modelId} onChange={(event) => { setModelId(event.target.value); setSingleResult(null); }}>{models.map((model) => <option key={model.id} value={model.id}>{model.display_name}</option>)}</select></label><button className="primary-button" disabled={!serviceReady || predicting || Object.keys(record).length === 0} onClick={runSingle}>{predicting ? "Running…" : "Run prediction"}</button></div>{singleResult && <div className="single-result"><PredictionCard result={singleResult} model={selectedModel} example={selectedExample} modified={modified} /></div>}</> : <><div className="panel-heading"><div><p className="eyebrow">Same normalized input for every model</p><h2>Compare all models</h2></div><button className="primary-button" disabled={!serviceReady || predicting || Object.keys(record).length === 0} onClick={runCompare}>{predicting ? "Comparing…" : "Compare all"}</button></div>{compareResults.length > 0 && <>{hasDisagreement(compareResults) && <p className="disagreement-note">Models disagree on this record because each learned a different decision boundary and uses its own validation-selected operating threshold.</p>}<div className="prediction-grid">{compareResults.map((result) => <PredictionCard key={result.model} result={result} model={models.find((model) => model.id === result.model)} example={selectedExample} modified={modified} />)}</div></>}</>}
            {!singleResult && compareResults.length === 0 && <div className="empty-state"><span>↗</span><h3>Ready for frozen inference</h3><p>Select a curated record, inspect its active features if needed, then run {view === "compare" ? "all four models" : "the selected model"}.</p>{selectedExample && <p>Selected scenario: <b>{scenarioLabel[selectedExample.scenario] ?? selectedExample.scenario}</b></p>}</div>}
            <p className="probability-caveat">P(attack) is the frozen model score for this dataset. It is not a calibrated real-world attack likelihood.</p>
          </div>
        </section>
      )}
    </main>
  );
}
