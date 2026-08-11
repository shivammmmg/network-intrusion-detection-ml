import type { DemoExample, FeatureRecord, PredictionResult } from "./types";

export const scenarioLabel: Record<string, string> = {
  high_confidence_normal: "High-confidence normal",
  high_confidence_attack: "High-confidence attack",
  model_disagreement: "Model disagreement",
  random_forest_false_positive: "False positive example",
  xgboost_false_negative: "False negative example",
  near_threshold_random_forest: "Near-threshold example"
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
