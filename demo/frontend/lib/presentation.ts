import type { DemoExample, FeatureRecord, PredictionResult } from "./types";

export const scenarioLabel: Record<string, string> = {
  high_confidence_normal: "High-confidence normal",
  high_confidence_attack: "High-confidence attack",
  model_disagreement: "Model disagreement",
  random_forest_false_positive: "False positive example",
  xgboost_false_negative: "False negative example",
  near_threshold_random_forest: "Near-threshold example"
};

// Short, plain-language descriptions for the Live Detection curated-example
// gallery — purely presentational, no effect on which record/scenario is used.
export const scenarioDescription: Record<string, string> = {
  high_confidence_normal: "A known normal flow scored confidently as normal.",
  high_confidence_attack: "A known attack scored confidently as malicious.",
  model_disagreement: "A flow where the finalized models do not all reach the same decision.",
  random_forest_false_positive: "A normal flow incorrectly flagged by the selected model.",
  xgboost_false_negative: "An attack missed by the selected model.",
  near_threshold_random_forest: "A borderline flow close to the selected model's locked decision threshold."
};

export function cloneRecord(record: FeatureRecord): FeatureRecord {
  return { ...record };
}

export function isModified(record: FeatureRecord, example?: DemoExample): boolean {
  if (!example) return true;
  const expected = example.record;
  return Object.keys(expected).some((key) => record[key] !== expected[key]) || Object.keys(record).length !== Object.keys(expected).length;
}

export function percentage(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

export function correctness(result: PredictionResult, example: DemoExample): "Correct" | "False Positive" | "False Negative" {
  const truth = example.evaluation_metadata.known_held_out_label;
  if (result.prediction === truth) return "Correct";
  return result.prediction === 1 ? "False Positive" : "False Negative";
}

export function hasDisagreement(results: PredictionResult[]): boolean {
  return new Set(results.map((result) => result.prediction)).size > 1;
}

/** Signed percentage-point distance of a score above (+) or below (-) its locked threshold. */
export function thresholdDistancePts(result: PredictionResult): number {
  return (result.attack_probability - result.threshold) * 100;
}
