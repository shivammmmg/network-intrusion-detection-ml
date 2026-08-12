"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";

import { api, ApiError } from "../lib/api";
import { cloneRecord, isModified } from "../lib/presentation";
import { MODEL_DISPLAY } from "../lib/modelDisplay";
import type { DemoExample, FeatureRecord, HealthResponse, ModelMetadata, PredictionResult, SchemaResponse } from "../lib/types";

import { AppHeader } from "./AppHeader";
import { ServiceBanner } from "./ServiceBanner";
import { OverviewPanel } from "./OverviewPanel";
import { LiveDetectionWorkspace } from "./live/LiveDetectionWorkspace";
import { CompareWorkspace } from "./compare/CompareWorkspace";
import { OfflineWorkspaceNotice } from "./OfflineWorkspaceNotice";
import { ModelDetailsDialog } from "./ModelDetailsDialog";
import { AnalysisPanel } from "./AnalysisPanel";

type View = "overview" | "live" | "compare" | "analysis";

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
  const [detailsModelId, setDetailsModelId] = useState<string | null>(null);
  const detailsTriggerRef = useRef<HTMLElement | null>(null);
  const reduceMotion = useReducedMotion();

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
  const detailsModel = MODEL_DISPLAY.find((model) => model.id === detailsModelId);
  const detailsMetadata = models.find((model) => model.id === detailsModelId);
  const detailsResult = compareResults.find((result) => result.model === detailsModelId);

  function openModelDetails(modelIdToOpen: string, trigger: HTMLButtonElement) {
    detailsTriggerRef.current = trigger;
    setDetailsModelId(modelIdToOpen);
  }

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
    <main className="app-shell">
      <AppHeader view={view} onViewChange={setView} serviceReady={serviceReady} />

      {loading && (
        <section className="loading-panel" role="status">
          <span className="spinner" aria-hidden="true" />
          Loading finalized model metadata…
        </section>
      )}

      {!loading && (
        <AnimatePresence mode="wait">
          <motion.div
            key={view}
            initial={reduceMotion ? false : { opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.12 }}
          >
            {view === "overview" && (
              <section className="view-panel">
                <OverviewPanel health={health} serviceReady={serviceReady} onNavigate={setView} />
                {!serviceReady && <ServiceBanner />}
              </section>
            )}

            {view === "live" && (
              <section className="view-panel">
                <header className="view-heading">
                  <p className="section-label">Single-flow analysis</p>
                  <h2>Live Detection</h2>
                  <p>
                    Run one of the finalized classifiers on a real held-out UNSW-NB15 traffic record and inspect how its
                    decision score compares with the validation-selected threshold.
                  </p>
                </header>
                {!serviceReady && <ServiceBanner />}
                {serviceReady && error && <p className="inline-error" role="alert">{error}</p>}
                {workspaceReady && schema ? (
                  <LiveDetectionWorkspace
                    schema={schema}
                    examples={examples}
                    selectedExample={selectedExample}
                    modified={modified}
                    record={record}
                    models={models}
                    modelId={modelId}
                    onModelIdChange={(id) => { setModelId(id); setSingleResult(null); }}
                    onSelectExample={chooseExample}
                    onChangeField={updateValue}
                    onReset={() => selectedExample && chooseExample(selectedExample)}
                    onRunSingle={runSingle}
                    serviceReady={serviceReady}
                    predicting={predicting}
                    singleResult={singleResult}
                  />
                ) : (
                  <OfflineWorkspaceNotice />
                )}
              </section>
            )}

            {view === "compare" && (
              <section className="view-panel">
                <header className="view-heading">
                  <p className="section-label">Shared evaluation protocol</p>
                  <h2>Model Comparison</h2>
                  <p>
                    Run all four finalized classifiers on the same held-out UNSW-NB15 traffic record and compare their
                    decisions, scores, and validation-selected thresholds.
                  </p>
                </header>
                {!serviceReady && <ServiceBanner />}
                {serviceReady && error && <p className="inline-error" role="alert">{error}</p>}
                {workspaceReady && schema && (
                  <CompareWorkspace
                    schema={schema}
                    examples={examples}
                    selectedExample={selectedExample}
                    modified={modified}
                    record={record}
                    models={models}
                    onSelectExample={chooseExample}
                    onChangeField={updateValue}
                    onReset={() => selectedExample && chooseExample(selectedExample)}
                    onRunCompare={runCompare}
                    serviceReady={serviceReady}
                    predicting={predicting}
                    compareResults={compareResults}
                    onOpenModelDetails={openModelDetails}
                  />
                )}
              </section>
            )}

            {view === "analysis" && (
              <section className="view-panel">
                <header className="view-heading">
                  <p className="section-label">Diagnostics and robustness</p>
                  <h2>Analysis</h2>
                  <p>Charts and tables generated from the committed diagnostics outputs — nothing on this page is computed live.</p>
                </header>
                <AnalysisPanel />
              </section>
            )}
          </motion.div>
        </AnimatePresence>
      )}

      {detailsModel && (
        <ModelDetailsDialog
          model={detailsModel}
          metadata={detailsMetadata}
          comparisonResult={detailsResult}
          serviceReady={serviceReady}
          restoreFocusRef={detailsTriggerRef}
          onClose={() => setDetailsModelId(null)}
        />
      )}
    </main>
  );
}
