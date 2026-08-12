import type { DemoExample, ModelMetadata, PredictionResult } from "../lib/types";
import { correctness, percentage } from "../lib/presentation";
import { ProbabilityMeter } from "./ProbabilityMeter";

interface PredictionCardProps {
  result: PredictionResult;
  model?: ModelMetadata;
  example?: DemoExample;
  modified: boolean;
}

export function PredictionCard({ result, model, example, modified }: PredictionCardProps) {
  const attack = result.prediction === 1;
  const verdict = !modified && example ? correctness(result, example) : null;
  const marginPts = (result.attack_probability - result.threshold) * 100;

  return (
    <article className={`prediction-card ${attack ? "is-attack" : "is-normal"}`}>
      <header className="result-head">
        <div>
          <p className="result-family">{model?.family ?? "finalized"} model</p>
          <h3>{model?.display_name ?? result.model}</h3>
        </div>
        <span className={`verdict-tag ${attack ? "is-attack" : "is-normal"}`}>{attack ? "Attack" : "Normal"}</span>
      </header>

      <ProbabilityMeter probability={result.attack_probability} threshold={result.threshold} />

      <dl className="result-metrics">
        <div>
          <dt>P(attack)</dt>
          <dd>{percentage(result.attack_probability)}</dd>
        </div>
        <div>
          <dt>Locked threshold</dt>
          <dd>{percentage(result.threshold)}</dd>
        </div>
        <div>
          <dt>Margin</dt>
          <dd>{marginPts >= 0 ? "+" : ""}{marginPts.toFixed(1)} pts</dd>
        </div>
      </dl>

      <p className="result-note">
        {attack
          ? "Attack predicted: P(attack) is at or above this model's validation-selected locked threshold."
          : "Normal predicted: P(attack) is below this model's validation-selected locked threshold."}
      </p>

      {verdict && <p className={`result-verdict ${verdict === "Correct" ? "is-correct" : "is-incorrect"}`}>{verdict}</p>}
    </article>
  );
}
