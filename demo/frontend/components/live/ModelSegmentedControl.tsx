import type { ModelMetadata } from "../../lib/types";
import { LIVE_MODEL_ORDER } from "../../lib/liveModelOrder";

interface ModelSegmentedControlProps {
  models: ModelMetadata[];
  modelId: string;
  onChange: (id: string) => void;
}

export function ModelSegmentedControl({ models, modelId, onChange }: ModelSegmentedControlProps) {
  const ordered = LIVE_MODEL_ORDER.map((id) => models.find((model) => model.id === id)).filter(
    (model): model is ModelMetadata => Boolean(model)
  );

  return (
    <div className="model-segmented" role="group" aria-label="Choose model">
      {ordered.map((model) => (
        <button
          key={model.id}
          type="button"
          className={model.id === modelId ? "is-active" : ""}
          aria-pressed={model.id === modelId}
          onClick={() => onChange(model.id)}
        >
          {model.display_name}
        </button>
      ))}
    </div>
  );
}
