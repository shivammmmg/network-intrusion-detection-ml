import type { DemoExample, ModelMetadata, PredictionResult } from "../../lib/types";
import { correctness, percentage, thresholdDistancePts } from "../../lib/presentation";
import { ScoreScale } from "../live/ScoreScale";

interface ModelResultCardProps {
  result: PredictionResult;
  model?: ModelMetadata;
  example?: DemoExample;
  modified: boolean;
}

export function ModelResultCard({ result, model, example, modified }: ModelResultCardProps) {
  const attack = result.prediction === 1;
  const distance = Math.abs(thresholdDistancePts(result));
  const verdict = !modified && example ? correctness(result, example) : null;

  return (
    <article className={`model-result-card ${attack ? "is-attack" : "is-normal"}`}>
      <p className="section-label">{model?.display_name ?? result.model}</p>
      <p className="model-result-verdict">{attack ? "ATTACK" : "NORMAL"}</p>

      <p className="model-result-score">
        Attack score <strong>{percentage(result.attack_probability)}</strong>
      </p>

      <ScoreScale probability={result.attack_probability} threshold={result.threshold} attack={attack} />

      <dl className="model-result-meta">
        <div>
          <dt>Locked threshold</dt>
          <dd>{percentage(result.threshold)}</dd>
        </div>
        <div>
          <dt>{attack ? "Above threshold" : "Below threshold"}</dt>
          <dd>{distance.toFixed(1)} pts</dd>
        </div>
      </dl>

      {verdict && (
        <p className={`model-result-verdict-tag ${verdict === "Correct" ? "is-correct" : "is-incorrect"}`}>
          {verdict === "Correct" ? "✓ Correct" : "✕ Incorrect"}
        </p>
      )}
    </article>
  );
}
