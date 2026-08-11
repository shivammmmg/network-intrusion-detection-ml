export type FeatureValue = number | string;
export type FeatureRecord = Record<string, FeatureValue>;

export interface HealthResponse {
  ready: boolean;
  frozen_source_sha: string;
  models: Record<string, boolean>;
}

export interface ModelMetadata {
  id: string;
  display_name: string;
  family: "linear" | "tree";
  threshold: number;
  transformed_feature_count: number;
  probability_semantics: string;
  caveat: string;
}

export interface ModelsResponse {
  input_contract: string;
  models: ModelMetadata[];
}

export interface SchemaField {
  name: string;
  type: "string" | "integer" | "number";
  kind: "categorical" | "numeric";
  required: boolean;
  binary: boolean;
  non_negative: boolean;
}

export interface SchemaResponse {
  input_contract: string;
  fields: SchemaField[];
  categorical_categories: Record<string, string[]>;
  excluded_fields: string[];
}

export interface PredictionResult {
  model: string;
  attack_probability: number;
  threshold: number;
  prediction: 0 | 1;
  label: "normal" | "attack";
  threshold_source: "validation_selected_locked";
  input_contract: string;
}

export interface PredictAllResponse {
  input_contract: string;
  results: PredictionResult[];
}

export interface DemoExample {
  sample_index: number;
  scenario: string;
  record: FeatureRecord;
  evaluation_metadata: {
    known_held_out_label: 0 | 1;
    expected_probabilities: Record<string, number>;
    expected_locked_predictions: Record<string, 0 | 1>;
  };
}

export interface ExamplesResponse {
  input_contract: string;
  examples: DemoExample[];
}
