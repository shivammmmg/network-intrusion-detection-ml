import { MODEL_DISPLAY } from "../lib/modelDisplay";

interface ModelDirectoryProps {
  serviceReady: boolean;
  onOpenDetails: (modelId: string, trigger: HTMLButtonElement) => void;
}

export function ModelDirectory({ serviceReady, onOpenDetails }: ModelDirectoryProps) {
  return (
    <section className="model-directory" aria-label="Finalized project models">
      {MODEL_DISPLAY.map((model, index) => (
        <button
          key={model.id}
          type="button"
          className="model-directory-row"
          aria-haspopup="dialog"
          onClick={(event) => onOpenDetails(model.id, event.currentTarget)}
        >
          <span className="model-directory-index">{String(index + 1).padStart(2, "0")}</span>
          <span className="model-directory-id">
            <span className="model-directory-name">{model.name}</span>
            <span className="model-directory-category">{model.category}</span>
          </span>
          <span className="model-directory-action">{serviceReady ? "View details" : "Details unavailable"}</span>
        </button>
      ))}
    </section>
  );
}
