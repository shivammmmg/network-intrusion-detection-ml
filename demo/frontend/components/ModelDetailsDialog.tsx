import type { RefObject } from "react";
import { percentage } from "../lib/presentation";
import type { ModelDisplayInfo } from "../lib/modelDisplay";
import type { ModelMetadata, PredictionResult } from "../lib/types";
import { useFocusTrap } from "../hooks/useFocusTrap";

interface ModelDetailsDialogProps {
  model: ModelDisplayInfo;
  metadata?: ModelMetadata;
  comparisonResult?: PredictionResult;
  serviceReady: boolean;
  restoreFocusRef: RefObject<HTMLElement | null>;
  onClose: () => void;
}

export function ModelDetailsDialog({ model, metadata, comparisonResult, serviceReady, restoreFocusRef, onClose }: ModelDetailsDialogProps) {
  const { dialogRef, closeRef } = useFocusTrap(true, onClose, restoreFocusRef);

  return (
    <div className="dialog-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
      <section
        className="dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="model-dialog-title"
        aria-describedby="model-dialog-description"
        ref={dialogRef}
      >
        <header className="dialog-header">
          <div>
            <p className="section-label">{model.category}</p>
            <h2 id="model-dialog-title">{model.name}</h2>
          </div>
          <button type="button" className="dialog-close" onClick={onClose} ref={closeRef} aria-label="Close model details">
            <span aria-hidden="true">×</span>
          </button>
        </header>

        <p className={`status-line ${serviceReady ? "is-ready" : "is-offline"}`} role="status">
          <span className="status-dot" aria-hidden="true" />
          {serviceReady ? "Model service ready" : "Live inference unavailable"}
        </p>

        <div className="dialog-body" id="model-dialog-description">
          <dl className="metadata-list">
            <div><dt>Model category</dt><dd>{model.category}</dd></div>
            <div><dt>Input contract</dt><dd>39 primary features</dd></div>
            <div><dt>Evaluation</dt><dd>Validation-selected locked threshold; no retraining</dd></div>
            {metadata && <div><dt>Transformed features</dt><dd>{metadata.transformed_feature_count}</dd></div>}
            {metadata && <div><dt>Locked threshold</dt><dd>{percentage(metadata.threshold)}</dd></div>}
          </dl>

          {metadata ? (
            <div className="dialog-notes">
              <p>{metadata.caveat}</p>
              <p>{metadata.probability_semantics}</p>
              {comparisonResult && (
                <p>
                  Current comparison: <strong>{comparisonResult.label === "attack" ? "Attack" : "Normal"}</strong> at{" "}
                  {percentage(comparisonResult.attack_probability)} P(attack).
                </p>
              )}
            </div>
          ) : (
            <p className="dialog-notes">Evaluation details are unavailable while the model artifact service is offline.</p>
          )}
        </div>
      </section>
    </div>
  );
}
