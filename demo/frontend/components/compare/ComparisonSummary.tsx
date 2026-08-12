import type { DemoExample, ModelMetadata, PredictionResult } from "../../lib/types";
import { hasDisagreement, percentage, thresholdDistancePts } from "../../lib/presentation";

interface ComparisonSummaryProps {
  results: PredictionResult[];
  models: ModelMetadata[];
  example?: DemoExample;
  modified: boolean;
}

function nameFor(models: ModelMetadata[], id: string): string {
  return models.find((model) => model.id === id)?.display_name ?? id;
}

export function ComparisonSummary({ results, models, example, modified }: ComparisonSummaryProps) {
  const attackResults = results.filter((result) => result.prediction === 1);
  const normalResults = results.filter((result) => result.prediction === 0);
  const total = results.length;

  const highest = [...results].sort((a, b) => b.attack_probability - a.attack_probability)[0];
  const closest = [...results].sort(
    (a, b) => Math.abs(thresholdDistancePts(a)) - Math.abs(thresholdDistancePts(b))
  )[0];
  const closestDistance = closest ? thresholdDistancePts(closest) : 0;

  const disagreement = hasDisagreement(results);
  const majority = attackResults.length >= normalResults.length ? attackResults : normalResults;
  const minority = attackResults.length >= normalResults.length ? normalResults : attackResults;
  const majorityLabel = attackResults.length >= normalResults.length ? "Attack" : "Normal";
  const minorityLabel = majorityLabel === "Attack" ? "Normal" : "Attack";

  return (
    <div className="comparison-summary">
      <dl className="comparison-stats">
        <div>
          <dt>Consensus</dt>
          <dd>
            {attackResults.length === total || normalResults.length === total
              ? `${total} / ${total} models agree: ${attackResults.length === total ? "Attack" : "Normal"}`
              : `${majority.length} / ${total} models predict ${majorityLabel}`}
          </dd>
        </div>
        {highest && (
          <div>
            <dt>Highest attack score</dt>
            <dd>{nameFor(models, highest.model)} — {percentage(highest.attack_probability)}</dd>
          </div>
        )}
        {closest && (
          <div>
            <dt>Closest to threshold</dt>
            <dd>
              {nameFor(models, closest.model)} — {Math.abs(closestDistance).toFixed(1)} pts{" "}
              {closestDistance >= 0 ? "above" : "below"}
            </dd>
          </div>
        )}
        {!modified && example && (
          <div>
            <dt>Ground truth</dt>
            <dd>{example.evaluation_metadata.known_held_out_label ? "Attack" : "Normal"}</dd>
          </div>
        )}
      </dl>

      {disagreement ? (
        <div className="disagreement-banner" role="note">
          <p className="disagreement-banner-title">Model disagreement detected</p>
          <p>
            {majority.length} of {total} models predict <strong>{majorityLabel}</strong> — {minority.map((result) => nameFor(models, result.model)).join(", ")}{" "}
            {minority.length === 1 ? "predicts" : "predict"} <strong>{minorityLabel}</strong>.
          </p>
          <p className="disagreement-banner-explain">
            The models rank this flow differently relative to their own locked decision thresholds.
          </p>
        </div>
      ) : (
        <p className="agreement-note">
          {total} of {total} models agree: <strong>{attackResults.length === total ? "Attack" : "Normal"}</strong>
        </p>
      )}
    </div>
  );
}
