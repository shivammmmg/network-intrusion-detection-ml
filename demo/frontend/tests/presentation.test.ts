import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";

import { api, isPredictionResult } from "../lib/api";
import { correctness, hasDisagreement, isModified, percentage, scenarioLabel } from "../lib/presentation";
import { createManualRecord, recordValidationError } from "../lib/recordValidation";
import type { DemoExample, ModelMetadata, PredictionResult, SchemaResponse } from "../lib/types";
import { PredictionCard } from "../components/PredictionCard";
import { ScoreScale } from "../components/live/ScoreScale";
import { ModelSegmentedControl } from "../components/live/ModelSegmentedControl";
import { ExampleGallery } from "../components/live/ExampleGallery";

const example: DemoExample = {
  sample_index: 12,
  scenario: "model_disagreement",
  record: { dur: 0.5, proto: "tcp" },
  evaluation_metadata: {
    known_held_out_label: 0,
    expected_probabilities: { random_forest: 0.7 },
    expected_locked_predictions: { random_forest: 1 }
  }
};

const attack: PredictionResult = {
  model: "random_forest",
  attack_probability: 0.7,
  threshold: 0.49,
  prediction: 1,
  label: "attack",
  threshold_source: "validation_selected_locked",
  input_contract: "unsw_nb15_primary_v1"
};

const model: ModelMetadata = {
  id: "random_forest",
  display_name: "Random Forest",
  family: "tree",
  threshold: 0.49,
  transformed_feature_count: 50,
  probability_semantics: "score",
  caveat: "demo"
};

const schema: SchemaResponse = {
  input_contract: "unsw_nb15_primary_v1",
  fields: [
    { name: "dur", type: "number", kind: "numeric", required: true, binary: false, non_negative: true },
    { name: "spkts", type: "integer", kind: "numeric", required: true, binary: false, non_negative: true },
    { name: "proto", type: "string", kind: "categorical", required: true, binary: false, non_negative: false }
  ],
  categorical_categories: { proto: ["tcp"] },
  excluded_fields: []
};

afterEach(() => vi.unstubAllGlobals());

describe("frontend response and example presentation", () => {
  it("accepts a complete backend prediction response and rejects malformed payloads", () => {
    expect(isPredictionResult(attack)).toBe(true);
    expect(isPredictionResult({ model: "random_forest", attack_probability: 0.7 })).toBe(false);
    expect(isPredictionResult({ ...attack, attack_probability: Number.NaN })).toBe(false);
    expect(isPredictionResult({ ...attack, prediction: 0 })).toBe(false);
  });

  it("blocks cleared, non-finite, negative, and fractional-integer values before JSON serialization", () => {
    expect(recordValidationError({ dur: Number.NaN, spkts: 1, proto: "tcp" }, schema)).toBe("dur must be a finite number.");
    expect(recordValidationError({ dur: 0, spkts: 1.5, proto: "tcp" }, schema)).toBe("spkts must be an integer.");
    expect(recordValidationError({ dur: -1, spkts: 1, proto: "tcp" }, schema)).toBe("dur must be zero or greater.");
    expect(recordValidationError({ dur: 0, spkts: 1, proto: "" }, schema)).toBe("proto is required.");
    expect(recordValidationError({ dur: 0, spkts: 1, proto: "tcp" }, schema)).toBeNull();
  });

  it("creates a fully blank, schema-shaped record for manual entry", () => {
    const manualRecord = createManualRecord(schema);
    expect(Object.keys(manualRecord)).toEqual(["dur", "spkts", "proto"]);
    expect(manualRecord.dur).toBeNaN();
    expect(manualRecord.spkts).toBeNaN();
    expect(manualRecord.proto).toBe("");
    expect(recordValidationError(manualRecord, schema)).toBe("dur must be a finite number.");
  });

  it("labels curated examples and detects modified input", () => {
    expect(scenarioLabel[example.scenario]).toBe("Model disagreement");
    expect(isModified(example.record, example)).toBe(false);
    expect(isModified({ ...example.record, dur: 0.6 }, example)).toBe(true);
  });

  it("renders locked-threshold outcomes without averaging models", () => {
    expect(percentage(attack.attack_probability)).toBe("70.00%");
    expect(correctness(attack, example)).toBe("False Positive");
    expect(hasDisagreement([attack, { ...attack, model: "neural_network", prediction: 0, label: "normal" }])).toBe(true);
  });

  it("uses the selected model route and turns validation payloads into a clean error", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: [{ msg: "Field required" }] }), { status: 422 }));
    vi.stubGlobal("fetch", fetchMock);

    await expect(api.predict("xgboost", example.record)).rejects.toEqual(expect.objectContaining({ name: "ApiError", message: "Invalid request: Field required", status: 422 }));
    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8000/predict/xgboost",
      expect.objectContaining({ method: "POST", body: JSON.stringify(example.record) })
    );
  });

  it("turns a timed-out inference request into a recoverable user-facing error", async () => {
    const timeout = new Error("aborted");
    timeout.name = "AbortError";
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(timeout));

    await expect(api.predict("xgboost", example.record)).rejects.toEqual(
      expect.objectContaining({ name: "ApiError", message: "Model service timed out. Please try again." })
    );
  });

  it("renders prediction cards with exact probability, threshold, and evaluation metadata", () => {
    const markup = renderToStaticMarkup(createElement(PredictionCard, { result: attack, model, example, modified: false }));
    expect(markup).toContain("70.00%");
    expect(markup).toContain("49.00%");
    expect(markup).toContain("False Positive");
    expect(markup).toContain("Random Forest");
  });

  it("renders a zero attack probability as zero width rather than fabricating a visible score", () => {
    const markup = renderToStaticMarkup(createElement(PredictionCard, { result: { ...attack, attack_probability: 0, prediction: 0, label: "normal" }, model, modified: true }));
    expect(markup).toContain("width:0%");
    expect(markup).toContain(">Normal<");
  });

  it("fills the live score pill to the exact returned attack score while retaining its threshold marker", () => {
    const markup = renderToStaticMarkup(createElement(ScoreScale, { probability: 0.7, threshold: 0.49, attack: true }));
    expect(markup).toContain("score-scale-fill is-attack");
    expect(markup).toContain("width:70%");
    expect(markup).toContain("left:49%");
  });

  it("renders every model as a separately selectable option", () => {
    const models = [
      model,
      { ...model, id: "xgboost", display_name: "XGBoost" },
      { ...model, id: "neural_network", display_name: "Neural Network" },
      { ...model, id: "logistic_regression", display_name: "Logistic Regression" }
    ];
    const markup = renderToStaticMarkup(createElement(ModelSegmentedControl, { models, modelId: "neural_network", onChange: () => undefined }));
    expect((markup.match(/aria-pressed=/g) ?? []).length).toBe(4);
    expect(markup).toContain(">Logistic Regression</button>");
  });

  it("inserts the custom traffic option after the high-confidence attack example", () => {
    const galleryExamples = [
      { ...example, sample_index: 1, scenario: "high_confidence_normal" },
      { ...example, sample_index: 2, scenario: "high_confidence_attack" },
      { ...example, sample_index: 3, scenario: "model_disagreement" }
    ];
    const markup = renderToStaticMarkup(
      createElement(ExampleGallery, {
        examples: galleryExamples,
        selectedExample: galleryExamples[0],
        modified: false,
        onSelect: () => undefined,
        onSelectCustom: () => undefined
      })
    );
    expect(markup.indexOf("High-confidence attack")).toBeLessThan(markup.indexOf("Custom traffic record"));
    expect(markup.indexOf("Custom traffic record")).toBeLessThan(markup.indexOf("Model disagreement"));
    expect(markup).toContain("Manual input · no held-out label");
  });

  it("renders all four comparison cards while suppressing stale ground-truth metadata after an edit", () => {
    const results = ["logistic_regression", "neural_network", "random_forest", "xgboost"].map((id, index) => ({
      ...attack,
      model: id,
      prediction: index === 1 ? 0 as const : 1 as const,
      label: index === 1 ? "normal" as const : "attack" as const
    }));
    const markup = renderToStaticMarkup(createElement("div", null, results.map((result) => createElement(PredictionCard, { key: result.model, result, model, example, modified: true }))));
    expect((markup.match(/prediction-card/g) ?? []).length).toBe(4);
    expect(markup).not.toContain("False Positive");
  });
});
