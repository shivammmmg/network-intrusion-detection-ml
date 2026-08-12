"use client";

import { AnimatePresence, motion, useReducedMotion } from "framer-motion";

import type { DemoExample, FeatureRecord, ModelMetadata, PredictionResult, SchemaResponse } from "../../lib/types";
import { ExampleGallery } from "../live/ExampleGallery";
import { TrafficSummary } from "../live/TrafficSummary";
import { FeatureEditor } from "../FeatureEditor";
import { Disclosure } from "../Disclosure";
import { ModelDirectory } from "../ModelDirectory";
import { ModelResultCard } from "./ModelResultCard";
import { ComparisonSummary } from "./ComparisonSummary";

interface CompareWorkspaceProps {
  schema: SchemaResponse;
  examples: DemoExample[];
  selectedExample?: DemoExample;
  modified: boolean;
  record: FeatureRecord;
  models: ModelMetadata[];
  onSelectExample: (example: DemoExample) => void;
  onChangeField: (name: string, value: string, type: string) => void;
  onReset: () => void;
  onRunCompare: () => void;
  serviceReady: boolean;
  predicting: boolean;
  compareResults: PredictionResult[];
  onOpenModelDetails: (modelId: string, trigger: HTMLButtonElement) => void;
}

export function CompareWorkspace(props: CompareWorkspaceProps) {
  const {
    schema,
    examples,
    selectedExample,
    modified,
    record,
    models,
    onSelectExample,
    onChangeField,
    onReset,
    onRunCompare,
    serviceReady,
    predicting,
    compareResults,
    onOpenModelDetails
  } = props;

  const reduceMotion = useReducedMotion();
  const canRun = serviceReady && !predicting && Object.keys(record).length > 0;
  const hasResult = compareResults.length > 0;
  const resultKey = compareResults.map((result) => `${result.model}:${result.attack_probability}`).join("|");

  return (
    <div className="compare-flow">
      <section className="compare-block">
        <ExampleGallery examples={examples} selectedExample={selectedExample} modified={modified} onSelect={onSelectExample} />
      </section>

      <section className="compare-block">
        <TrafficSummary record={record} modified={modified} />
        <Disclosure summary="Advanced feature editor">
          <FeatureEditor schema={schema} record={record} onChange={onChangeField} />
        </Disclosure>
        {modified && (
          <div className="compare-modified-note">
            <span className="status-tag is-warning">Modified input</span>
            <p className="hint-text is-warning">
              Ground-truth comparison is disabled because this record no longer matches the original held-out example.
            </p>
          </div>
        )}
        <button type="button" className="button button-ghost" disabled={!selectedExample || !modified} onClick={onReset}>
          Reset to selected example
        </button>
      </section>

      <section className="compare-block compare-cta-block">
        <div>
          <span className="section-label">3 · Compare</span>
          <p>Run the same record through Logistic Regression, Neural Network, Random Forest, and XGBoost.</p>
        </div>
        <button type="button" className="button button-primary" disabled={!canRun} onClick={onRunCompare}>
          {predicting ? "Comparing…" : "Compare all models"}
        </button>
      </section>

      <section className="compare-block compare-results-block">
        <div className="panel-row">
          <span className="section-label">4 · Results</span>
        </div>
        <AnimatePresence mode="wait">
          {hasResult ? (
            <motion.div
              key={`compare-${resultKey}`}
              initial={reduceMotion ? false : { opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.15 }}
            >
              <ComparisonSummary results={compareResults} models={models} example={selectedExample} modified={modified} />
              <div className="model-result-grid">
                {compareResults.map((result) => (
                  <ModelResultCard
                    key={result.model}
                    result={result}
                    model={models.find((model) => model.id === result.model)}
                    example={selectedExample}
                    modified={modified}
                  />
                ))}
              </div>
              <p className="footnote">
                This is each model&apos;s finalized decision score for this dataset, not a calibrated real-world attack probability.
              </p>
            </motion.div>
          ) : (
            <div className="compare-empty-state">
              <p className="section-label">No comparison yet</p>
              <p>Select a curated example above, then run the comparison.</p>
            </div>
          )}
        </AnimatePresence>
      </section>

      <Disclosure summary="About the models">
        <ModelDirectory serviceReady={serviceReady} onOpenDetails={onOpenModelDetails} />
      </Disclosure>
    </div>
  );
}
