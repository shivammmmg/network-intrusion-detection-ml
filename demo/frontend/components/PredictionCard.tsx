import type { DemoExample, ModelMetadata, PredictionResult } from "../lib/types";
import { correctness, percentage } from "../lib/presentation";

interface PredictionCardProps {
  result: PredictionResult;
  model?: ModelMetadata;
  example?: DemoExample;
  modified: boolean;
}

export function PredictionCard({ result, model, example, modified }: PredictionCardProps) {
  const attack = result.prediction === 1;
  const verdict = !modified && example ? correctness(result, example) : null;
  const probabilityStyle = { width: `${Math.min(100, Math.max(0, result.attack_probability * 100))}%` };
  const thresholdStyle = { left: `${result.threshold * 100}%` };

  return (
    <article className={`prediction-card ${attack ? "attack" : "normal"}`}>
      <div className="card-topline">
        <div>
          <p className="eyebrow">{model?.family ?? "finalized"} model</p>
          <h3>{model?.display_name ?? result.model}</h3>
        </div>
        <span className={`status-pill ${attack ? "attack" : "normal"}`}>{attack ? "Attack" : "Normal"}</span>
      </div>
      <div className="probability-block">
        <span>Attack probability</span>
        <strong>{percentage(result.attack_probability)}</strong>
      </div>
      <div className="threshold-track" aria-label={`Probability ${percentage(result.attack_probability)}, threshold ${percentage(result.threshold)}`}>
        <div className="track-fill" style={probabilityStyle} />
        <div className="threshold-marker" style={thresholdStyle}><span>threshold</span></div>
      </div>
      <div className="metric-row"><span>Locked threshold</span><b>{percentage(result.threshold)}</b></div>
      <p className="card-explanation">
        {attack
          ? "Attack is predicted because P(attack) is at or above this model’s validation-selected locked threshold."
          : "Normal is predicted because P(attack) is below this model’s validation-selected locked threshold."}
      </p>
      {verdict && <p className={`correctness ${verdict === "Correct" ? "correct" : "incorrect"}`}>{verdict}</p>}
    </article>
  );
}
