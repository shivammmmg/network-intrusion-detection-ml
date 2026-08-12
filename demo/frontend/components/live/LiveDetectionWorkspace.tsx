"use client";

import { AnimatePresence, motion, useReducedMotion } from "framer-motion";

import type { DemoExample, FeatureRecord, ModelMetadata, PredictionResult, SchemaResponse } from "../../lib/types";
import { ExampleGallery } from "./ExampleGallery";
import { TrafficSummary } from "./TrafficSummary";
import { ModelSegmentedControl } from "./ModelSegmentedControl";
import { ResultHero } from "./ResultHero";
import { FeatureEditor } from "../FeatureEditor";
import { Disclosure } from "../Disclosure";

interface LiveDetectionWorkspaceProps {
  schema: SchemaResponse;
  examples: DemoExample[];
  selectedExample?: DemoExample;
  modified: boolean;
  record: FeatureRecord;
  models: ModelMetadata[];
  modelId: string;
  onModelIdChange: (id: string) => void;
  onSelectExample: (example: DemoExample) => void;
  onChangeField: (name: string, value: string, type: string) => void;
  onReset: () => void;
  onRunSingle: () => void;
  inputError: string | null;
  serviceReady: boolean;
  predicting: boolean;
  singleResult: PredictionResult | null;
}

export function LiveDetectionWorkspace(props: LiveDetectionWorkspaceProps) {
  const {
    schema,
    examples,
    selectedExample,
    modified,
    record,
    models,
    modelId,
    onModelIdChange,
    onSelectExample,
    onChangeField,
    onReset,
    onRunSingle,
    inputError,
    serviceReady,
    predicting,
    singleResult
  } = props;

  const reduceMotion = useReducedMotion();
  const selectedModel = models.find((model) => model.id === modelId);
  const canRun = serviceReady && !predicting && Object.keys(record).length > 0;

  return (
    <div className="live-flow">
      <section className="live-block">
        <ExampleGallery examples={examples} selectedExample={selectedExample} modified={modified} onSelect={onSelectExample} />
      </section>

      <section className="live-block">
        <TrafficSummary record={record} modified={modified} />
        <Disclosure summary="Advanced feature editor">
          <FeatureEditor schema={schema} record={record} onChange={onChangeField} />
        </Disclosure>
        {inputError && <p className="inline-error" role="alert">{inputError}</p>}
        <button type="button" className="button button-ghost" disabled={!selectedExample || !modified} onClick={onReset}>
          Reset to selected example
        </button>
      </section>

      <section className="live-block">
        <div className="panel-row">
          <span className="section-label">3 · Choose model</span>
        </div>
        <div className="live-run-row">
          <ModelSegmentedControl models={models} modelId={modelId} onChange={onModelIdChange} />
          <button type="button" className="button button-primary" disabled={!canRun || Boolean(inputError)} onClick={onRunSingle}>
            {predicting ? "Running…" : "Run prediction"}
          </button>
        </div>
      </section>

      <section className="live-block live-result-block">
        <div className="panel-row">
          <span className="section-label">4 · Prediction</span>
        </div>
        <AnimatePresence mode="wait">
          {singleResult ? (
            <motion.div
              key={`live-${singleResult.model}`}
              initial={reduceMotion ? false : { opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.15 }}
            >
              <ResultHero result={singleResult} model={selectedModel} example={selectedExample} modified={modified} />
            </motion.div>
          ) : (
            <div className="live-empty-state">
              <p className="section-label">No prediction yet</p>
              <p>Select a curated example above, choose a model, then run the prediction.</p>
            </div>
          )}
        </AnimatePresence>
      </section>
    </div>
  );
}
