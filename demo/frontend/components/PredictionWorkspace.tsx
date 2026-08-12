"use client";

import { AnimatePresence, motion, useReducedMotion } from "framer-motion";

import type { DemoExample, FeatureRecord, ModelMetadata, PredictionResult, SchemaResponse } from "../lib/types";
import { hasDisagreement, scenarioLabel } from "../lib/presentation";
import { ExampleSelector } from "./ExampleSelector";
import { FeatureEditor } from "./FeatureEditor";
import { PredictionCard } from "./PredictionCard";
import { EmptyState } from "./EmptyState";

interface PredictionWorkspaceProps {
  mode: "single" | "compare";
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
  onRunCompare: () => void;
  serviceReady: boolean;
  predicting: boolean;
  singleResult: PredictionResult | null;
  compareResults: PredictionResult[];
}

export function PredictionWorkspace(props: PredictionWorkspaceProps) {
  const {
    mode,
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
    onRunCompare,
    serviceReady,
    predicting,
    singleResult,
    compareResults
  } = props;

  const reduceMotion = useReducedMotion();
  const selectedModel = models.find((model) => model.id === modelId);
  const hasResult = mode === "single" ? Boolean(singleResult) : compareResults.length > 0;
  const canRun = serviceReady && !predicting && Object.keys(record).length > 0;

  return (
    <section className="workspace">
      <aside className="input-panel" aria-label="Traffic record input">
        <ExampleSelector examples={examples} selectedExample={selectedExample} modified={modified} onSelect={onSelectExample} />
        <FeatureEditor schema={schema} record={record} onChange={onChangeField} />
        <button type="button" className="button button-ghost" disabled={!selectedExample || !modified} onClick={onReset}>
          Reset to selected example
        </button>
      </aside>

      <div className="result-panel">
        {mode === "single" ? (
          <div className="result-panel-head">
            <span className="section-label">Single-model prediction</span>
            <div className="action-row">
              <label className="field field-inline">
                <span className="field-label">Model</span>
                <select value={modelId} onChange={(event) => onModelIdChange(event.target.value)}>
                  {models.map((model) => (
                    <option key={model.id} value={model.id}>{model.display_name}</option>
                  ))}
                </select>
              </label>
              <button type="button" className="button button-primary" disabled={!canRun} onClick={onRunSingle}>
                {predicting ? "Running…" : "Run prediction"}
              </button>
            </div>
          </div>
        ) : (
          <div className="result-panel-head">
            <span className="section-label">Four-model comparison</span>
            <button type="button" className="button button-primary" disabled={!canRun} onClick={onRunCompare}>
              {predicting ? "Comparing…" : "Compare all models"}
            </button>
          </div>
        )}

        <AnimatePresence mode="wait">
          {hasResult ? (
            <motion.div
              key={mode === "single" ? `single-${singleResult?.model}` : `compare-${compareResults.length}`}
              initial={reduceMotion ? false : { opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.15 }}
            >
              {mode === "single" && singleResult && (
                <div className="single-result">
                  <PredictionCard result={singleResult} model={selectedModel} example={selectedExample} modified={modified} />
                </div>
              )}
              {mode === "compare" && compareResults.length > 0 && (
                <>
                  {hasDisagreement(compareResults) && (
                    <p className="disagreement-note" role="note">
                      Models disagree on this record: each classifier learned a different decision boundary and uses its own
                      validation-selected threshold.
                    </p>
                  )}
                  <div className="prediction-grid">
                    {compareResults.map((result) => (
                      <PredictionCard
                        key={result.model}
                        result={result}
                        model={models.find((model) => model.id === result.model)}
                        example={selectedExample}
                        modified={modified}
                      />
                    ))}
                  </div>
                </>
              )}
            </motion.div>
          ) : (
            <EmptyState
              mode={mode}
              scenario={selectedExample ? scenarioLabel[selectedExample.scenario] ?? selectedExample.scenario : undefined}
            />
          )}
        </AnimatePresence>

        <p className="footnote">P(attack) is the model&apos;s finalized decision score for this dataset — not a calibrated real-world attack probability.</p>
      </div>
    </section>
  );
}
