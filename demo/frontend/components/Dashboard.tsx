"use client";

import { useEffect, useMemo, useState } from "react";

import { api, ApiError } from "../lib/api";
import { cloneRecord, hasDisagreement, isModified, scenarioLabel } from "../lib/presentation";
import type { DemoExample, FeatureRecord, HealthResponse, ModelMetadata, PredictionResult, SchemaResponse } from "../lib/types";
import { PredictionCard } from "./PredictionCard";

type View = "overview" | "live" | "compare";

const navigation: Array<{ id: View; label: string }> = [
  { id: "overview", label: "Overview" },
  { id: "live", label: "Live Detection" },
  { id: "compare", label: "Compare Models" }
];

const snapshotMetrics = [
  { value: "4", label: "Models" },
  { value: "39", label: "Primary Features" },
  { value: "Binary", label: "Classification" },
  { value: "UNSW-NB15", label: "Dataset" }
];

const methodology = [
  "TTL leakage fields excluded",
  "Validation-selected thresholds",
  "Frozen finalized artifacts",
  "No retraining"
];

const pipelineSteps = ["Data", "Preprocessing", "Four Models", "Threshold Analysis", "Final Evaluation"];

const offlineModels = [
  { abbreviation: "LR", name: "Logistic Regression", tone: "cyan" },
  { abbreviation: "NN", name: "Neural Network", tone: "violet" },
  { abbreviation: "RF", name: "Random Forest", tone: "indigo" },
  { abbreviation: "XG", name: "XGBoost", tone: "teal" }
];

function formValue(value: FeatureRecord[string] | undefined): string {
  return typeof value === "number" && Number.isNaN(value) ? "" : String(value ?? "");
}

function ServiceNotice() {
  return (
    <section className="service-notice" role="status">
      <span className="warning-icon" aria-hidden="true"><span /></span>
      <div>
        <strong>Live inference unavailable</strong>
        <p>The dashboard results and methodology remain available, but live model predictions are temporarily offline.</p>
        <small>Artifact service unavailable</small>
      </div>
    </section>
  );
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
    void Promise.allSettled([api.health(), api.models(), api.schema(), api.examples()] as const)
      .then(([healthResult, modelsResult, schemaResult, examplesResult]) => {
        if (healthResult.status === "fulfilled") setHealth(healthResult.value);
        if (modelsResult.status === "fulfilled") setModels(modelsResult.value.models);
        if (schemaResult.status === "fulfilled") setSchema(schemaResult.value);
        if (examplesResult.status === "fulfilled") {
          setExamples(examplesResult.value.examples);
          const initial = examplesResult.value.examples[0];
          if (initial) {
            setSelectedExample(initial);
            setRecord(cloneRecord(initial.record));
          }
        }

        const failedRequest = [healthResult, modelsResult, schemaResult, examplesResult].find(
          (result) => result.status === "rejected"
        );
        if (failedRequest?.status === "rejected") {
          const reason: unknown = failedRequest.reason;
          setError(reason instanceof Error ? reason.message : "The model service did not return dashboard data.");
        } else {
          setError(null);
        }
      })
      .finally(() => setLoading(false));
  }, []);

  const modified = useMemo(() => isModified(record, selectedExample), [record, selectedExample]);
  const serviceReady = Boolean(health?.ready);
  const workspaceReady = serviceReady && Boolean(schema);
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

  function renderReadyWorkspace() {
    if (!schema) return null;

    return (
      <section className="workspace">
        <aside className="input-panel">
          <div className="panel-heading">
            <div><p className="eyebrow">Curated held-out examples</p><h2>Choose a record</h2></div>
            {modified && <span className="modified-badge">Modified example</span>}
          </div>
          <div className="example-list">
            {examples.map((example) => (
              <button
                key={example.sample_index}
                className={selectedExample?.sample_index === example.sample_index && !modified ? "example-card selected" : "example-card"}
                onClick={() => chooseExample(example)}
              >
                <b>{scenarioLabel[example.scenario] ?? example.scenario}</b>
                <span>Known held-out label: {example.evaluation_metadata.known_held_out_label ? "Attack" : "Normal"}</span>
              </button>
            ))}
          </div>
          {selectedExample && !modified && <p className="metadata-note">Selected: {scenarioLabel[selectedExample.scenario] ?? selectedExample.scenario}. Known held-out label is evaluation metadata only; it is never sent to a model.</p>}
          {selectedExample && modified && <p className="metadata-note warning-text">Ground-truth comparison disabled because this record has been edited.</p>}

          <div className="quick-fields">
            <p className="eyebrow">Primary visible fields</p>
            {["proto", "service", "state"].map((name) => (
              <label key={name}>{name}
                <select value={formValue(record[name])} onChange={(event) => updateValue(name, event.target.value, "string")}>
                  {schema.categorical_categories[name]?.map((option) => <option key={option} value={option}>{option}</option>)}
                </select>
              </label>
            ))}
          </div>

          <details className="advanced-editor">
            <summary>Advanced: View / edit all 39 features</summary>
            <div className="field-grid">
              {schema.fields.map((field) => (
                <label key={field.name}>{field.name}
                  {field.kind === "categorical" ? (
                    <select value={formValue(record[field.name])} onChange={(event) => updateValue(field.name, event.target.value, field.type)}>
                      {schema.categorical_categories[field.name]?.map((option) => <option key={option} value={option}>{option}</option>)}
                    </select>
                  ) : (
                    <input aria-label={field.name} type="number" min={field.non_negative ? 0 : undefined} step={field.type === "integer" ? 1 : "any"} value={formValue(record[field.name])} onChange={(event) => updateValue(field.name, event.target.value, field.type)} />
                  )}
                </label>
              ))}
            </div>
          </details>
          <button className="reset-button" disabled={!selectedExample || !modified} onClick={() => selectedExample && chooseExample(selectedExample)}>Reset to selected example</button>
        </aside>

        <div className="result-panel">
          {view === "live" ? (
            <>
              <div className="panel-heading"><div><p className="eyebrow">Single model mode</p><h2>Prediction result</h2></div></div>
              <div className="action-row">
                <label>Model
                  <select value={modelId} onChange={(event) => { setModelId(event.target.value); setSingleResult(null); }}>
                    {models.map((model) => <option key={model.id} value={model.id}>{model.display_name}</option>)}
                  </select>
                </label>
                <button className="primary-button" disabled={!serviceReady || predicting || Object.keys(record).length === 0} onClick={runSingle}>{predicting ? "Running…" : "Run prediction"}</button>
              </div>
              {singleResult && <div className="single-result"><PredictionCard result={singleResult} model={selectedModel} example={selectedExample} modified={modified} /></div>}
            </>
          ) : (
            <>
              <div className="panel-heading">
                <div><p className="eyebrow">Shared evaluation protocol</p><h2>Model comparison</h2></div>
                <button className="primary-button" disabled={!serviceReady || predicting || Object.keys(record).length === 0} onClick={runCompare}>{predicting ? "Comparing…" : "Compare all"}</button>
              </div>
              {compareResults.length > 0 && (
                <>
                  {hasDisagreement(compareResults) && <p className="disagreement-note">Models disagree on this record because each learned a different decision boundary and uses its own validation-selected operating threshold.</p>}
                  <div className="prediction-grid">{compareResults.map((result) => <PredictionCard key={result.model} result={result} model={models.find((model) => model.id === result.model)} example={selectedExample} modified={modified} />)}</div>
                </>
              )}
            </>
          )}
          {!singleResult && compareResults.length === 0 && (
            <div className="empty-state">
              <span className="empty-state-mark" aria-hidden="true" />
              <h3>Ready for frozen inference</h3>
              <p>Select a curated record, inspect its active features if needed, then run {view === "compare" ? "all four models" : "the selected model"}.</p>
              {selectedExample && <p>Selected scenario: <b>{scenarioLabel[selectedExample.scenario] ?? selectedExample.scenario}</b></p>}
            </div>
          )}
          <p className="probability-caveat">P(attack) is the frozen model score for this dataset. It is not a calibrated real-world attack likelihood.</p>
        </div>
      </section>
    );
  }

  return (
    <main className="shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">◈</span>
          <div><p>EECS 3404</p><h1>Intrusion Detection Lab</h1></div>
        </div>

        <nav className="nav-tabs" aria-label="Dashboard sections">
          {navigation.map((item) => (
            <button key={item.id} className={view === item.id ? "active" : ""} aria-pressed={view === item.id} onClick={() => setView(item.id)}>{item.label}</button>
          ))}
        </nav>

        <div className={`service-status ${serviceReady ? "ready" : "offline"}`} role="status" aria-live="polite">
          <span aria-hidden="true" />{serviceReady ? "Model service ready" : "Model service offline"}
        </div>
      </header>

      {loading && <section className="loading-panel"><div className="spinner" />Loading frozen model metadata…</section>}

      {!loading && view === "overview" && (
        <section className="overview-page">
          <div className="overview-hero">
            <article className="hero-panel">
              <p className="eyebrow">EECS 3404 <span>•</span> Network Security Analytics</p>
              <span className="dataset-chip"><span aria-hidden="true" />UNSW-NB15 <b>•</b> Binary Classification</span>
              <h2>Explainable and Drift-Aware <br />Machine Learning for <br />Network Intrusion Detection</h2>
              <p className="hero-copy">Classify UNSW-NB15 network traffic as <b>Normal</b> or <b>Attack</b> with four finalized models.</p>
              <div className="hero-actions">
                <button className="primary-button" onClick={() => setView("compare")}>Compare Models</button>
                <button className="secondary-button" onClick={() => setView("live")}>Live Detection</button>
              </div>
            </article>

            <aside className="snapshot-panel">
              <div className="section-heading">
                <div><p className="eyebrow">At a glance</p><h2>Project Snapshot</h2></div>
                <span className="snapshot-orbit" aria-hidden="true"><span /></span>
              </div>
              <div className="snapshot-grid">
                {snapshotMetrics.map((metric) => <div className="snapshot-metric" key={metric.label}><strong>{metric.value}</strong><span>{metric.label}</span></div>)}
              </div>
              <div className="methodology-strip">
                <p className="micro-label">Methodology safeguards</p>
                <div>{methodology.map((item) => <span key={item}><i aria-hidden="true" />{item}</span>)}</div>
              </div>
              <p className="source-sha">Frozen source <span>{health?.frozen_source_sha.slice(0, 8) ?? "loading"}</span></p>
            </aside>
          </div>

          <section className="pipeline-panel" aria-labelledby="pipeline-title">
            <div className="pipeline-heading"><div><p className="eyebrow">Evaluation workflow</p><h2 id="pipeline-title">Project Pipeline</h2></div><span>Frozen, read-only methodology</span></div>
            <div className="pipeline-steps">
              {pipelineSteps.map((step, index) => (
                <div className="pipeline-step" key={step}>
                  <span>{String(index + 1).padStart(2, "0")}</span><b>{step}</b>{index < pipelineSteps.length - 1 && <i aria-hidden="true" />}
                </div>
              ))}
            </div>
          </section>

          {!serviceReady && <ServiceNotice />}
        </section>
      )}

      {!loading && view === "live" && (
        <section className="feature-page">
          <header className="page-heading"><p className="eyebrow">Single-flow analysis</p><h2>Live Detection</h2><p>Evaluate a network flow against the finalized intrusion-detection models.</p></header>
          {!serviceReady && <ServiceNotice />}
          {serviceReady && error && <p className="inline-error" role="alert">{error}</p>}
          {workspaceReady ? renderReadyWorkspace() : (
            <section className="offline-detection-grid">
              <article className="offline-input-card">
                <div className="panel-heading"><div><p className="eyebrow">Input contract</p><h2>Network Flow Input</h2></div><span className="offline-chip">Offline preview</span></div>
                <p className="panel-copy">Feature controls become available when the frozen artifact service is restored.</p>
                <div className="offline-fields">
                  {["Protocol", "Service", "Connection state"].map((field) => <label key={field}>{field}<select disabled><option>Service offline</option></select></label>)}
                </div>
                <div className="offline-feature-row"><span>Advanced feature editor</span><b>39 fields</b></div>
                <button className="primary-button" disabled>Run Prediction</button>
              </article>

              <article className="offline-result-card">
                <div className="result-visual" aria-hidden="true"><span><i /></span></div>
                <p className="eyebrow">Prediction Result</p>
                <h2>Live model service unavailable</h2>
                <p>Predictions will appear here when the model service is restored.</p>
                <div className="result-placeholder"><span /><span /><span /></div>
              </article>
            </section>
          )}
        </section>
      )}

      {!loading && view === "compare" && (
        <section className="feature-page">
          <header className="page-heading"><p className="eyebrow">Shared evaluation protocol</p><h2>Model Comparison</h2><p>Compare the four finalized classifiers under the shared evaluation protocol.</p></header>
          {!serviceReady && <ServiceNotice />}
          {serviceReady && error && <p className="inline-error" role="alert">{error}</p>}
          {workspaceReady ? renderReadyWorkspace() : (
            <section className="offline-model-grid" aria-label="Finalized project models">
              {offlineModels.map((model, index) => (
                <article className={`offline-model-card ${model.tone}`} key={model.name}>
                  <div className="model-card-top"><span className="model-monogram" aria-hidden="true">{model.abbreviation}</span><span className="model-index">0{index + 1}</span></div>
                  <p className="micro-label">Finalized classifier</p>
                  <h3>{model.name}</h3>
                  <p>Results unavailable while model service is offline.</p>
                  <div className="model-card-status"><span aria-hidden="true" />Awaiting artifact service</div>
                </article>
              ))}
            </section>
          )}
        </section>
      )}
    </main>
  );
}
