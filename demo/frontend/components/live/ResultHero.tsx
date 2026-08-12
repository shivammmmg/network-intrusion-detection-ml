import type { DemoExample, ModelMetadata, PredictionResult } from "../../lib/types";
import { correctness, percentage } from "../../lib/presentation";
import { ScoreScale } from "./ScoreScale";

interface ResultHeroProps {
  result: PredictionResult;
  model?: ModelMetadata;
  example?: DemoExample;
  modified: boolean;
}

export function ResultHero({ result, model, example, modified }: ResultHeroProps) {
  const attack = result.prediction === 1;
  const modelName = model?.display_name ?? result.model;
  const distancePts = Math.abs((result.attack_probability - result.threshold) * 100);
  const verdict = !modified && example ? correctness(result, example) : null;

  return (
    <article className={`result-hero ${attack ? "is-attack" : "is-normal"}`}>
      <p className="section-label">{modelName}</p>
      <h2 className="result-hero-verdict">{attack ? "ATTACK" : "NORMAL"}</h2>
      <p className="result-hero-score">
        Attack score <strong>{percentage(result.attack_probability)}</strong>
      </p>

      <ScoreScale probability={result.attack_probability} threshold={result.threshold} attack={attack} />

      <dl className="result-hero-meta">
        <div>
          <dt>Locked threshold</dt>
          <dd>{percentage(result.threshold)}</dd>
        </div>
        <div>
          <dt>{attack ? "Above threshold" : "Below threshold"}</dt>
          <dd>{distancePts.toFixed(1)} percentage points</dd>
        </div>
      </dl>

      {verdict && example && (
        <div className="result-hero-truth">
          <span>Ground truth <strong>{example.evaluation_metadata.known_held_out_label ? "Attack" : "Normal"}</strong></span>
          <span>Prediction <strong>{attack ? "Attack" : "Normal"}</strong></span>
          <span className={`result-hero-verdict-tag ${verdict === "Correct" ? "is-correct" : "is-incorrect"}`}>
            {verdict === "Correct" ? "✓ Correct" : "✕ Incorrect"}
          </span>
        </div>
      )}

      <p className="result-hero-explain">
        {attack
          ? `The attack score exceeds ${modelName}'s validation-selected decision threshold, so this flow is classified as an attack.`
          : `The attack score remains below ${modelName}'s validation-selected decision threshold, so this flow is classified as normal.`}
      </p>

      <p className="footnote">
        This is the model&apos;s finalized decision score for this dataset, not a calibrated real-world attack probability.
      </p>
    </article>
  );
}
