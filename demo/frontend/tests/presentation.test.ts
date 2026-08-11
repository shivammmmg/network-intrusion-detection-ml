import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";

import { api, isPredictionResult } from "../lib/api";
import { correctness, hasDisagreement, isModified, percentage, scenarioLabel } from "../lib/presentation";
import type { DemoExample, ModelMetadata, PredictionResult } from "../lib/types";
import { PredictionCard } from "../components/PredictionCard";

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

afterEach(() => vi.unstubAllGlobals());

describe("frontend response and example presentation", () => {
  it("accepts a complete backend prediction response and rejects malformed payloads", () => {
    expect(isPredictionResult(attack)).toBe(true);
    expect(isPredictionResult({ model: "random_forest", attack_probability: 0.7 })).toBe(false);
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
