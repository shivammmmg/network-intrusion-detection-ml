import type { DemoExample, ModelMetadata } from "../../lib/types";
import { percentage } from "../../lib/presentation";
import { LIVE_MODEL_ORDER } from "../../lib/liveModelOrder";

interface AgreementPreviewProps {
  example: DemoExample;
  models: ModelMetadata[];
  selectedModelId: string;
}

/**
 * Reuses each curated example's precomputed expected_probabilities /
 * expected_locked_predictions (already shipped with the example fixtures)
 * instead of triggering a live compare-all request — no extra inference,
 * no backend change.
 */
export function AgreementPreview({ example, models, selectedModelId }: AgreementPreviewProps) {
  const { expected_probabilities: probabilities, expected_locked_predictions: predictions } = example.evaluation_metadata;

  const rows = LIVE_MODEL_ORDER.map((id) => models.find((model) => model.id === id))
    .filter((model): model is ModelMetadata => Boolean(model))
    .filter((model) => probabilities[model.id] !== undefined && predictions[model.id] !== undefined)
    .map((model) => ({
      model,
      attack: predictions[model.id] === 1,
      probability: probabilities[model.id]
    }));

  if (rows.length === 0) return null;

  return (
    <div className="agreement-preview">
      <span className="section-label">How the finalized models viewed this flow</span>
      <div className="agreement-preview-rows">
        {rows.map((row) => (
          <div key={row.model.id} className={`agreement-preview-row ${row.model.id === selectedModelId ? "is-selected" : ""}`}>
            <span className="agreement-preview-name">{row.model.display_name}</span>
            <span className={`agreement-preview-label ${row.attack ? "is-attack" : "is-normal"}`}>
              {row.attack ? "Attack" : "Normal"}
            </span>
            <span className="agreement-preview-score">{percentage(row.probability)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
