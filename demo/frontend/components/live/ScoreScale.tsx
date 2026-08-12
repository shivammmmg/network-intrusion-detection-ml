import { percentage } from "../../lib/presentation";

interface ScoreScaleProps {
  probability: number;
  threshold: number;
  attack: boolean;
}

export function ScoreScale({ probability, threshold, attack }: ScoreScaleProps) {
  const scorePos = Math.min(100, Math.max(0, probability * 100));
  const thresholdPos = Math.min(100, Math.max(0, threshold * 100));

  return (
    <div
      className="score-scale"
      role="img"
      aria-label={`Attack score ${percentage(probability)} against a locked threshold of ${percentage(threshold)}`}
    >
      <div className="score-scale-ends">
        <span>Normal</span>
        <span>Attack</span>
      </div>
      <div className="score-scale-body">
        <div className="score-scale-track">
          <div className={`score-scale-fill ${attack ? "is-attack" : "is-normal"}`} style={{ width: `${scorePos}%` }} />
          <div className={`score-scale-marker ${attack ? "is-attack" : "is-normal"}`} style={{ left: `${scorePos}%` }} aria-hidden="true" />
        </div>
        <div className="score-scale-threshold" style={{ left: `${thresholdPos}%` }} aria-hidden="true" />
        <div className="score-scale-threshold-label" style={{ left: `${thresholdPos}%` }}>
          Threshold {percentage(threshold)}
        </div>
      </div>
    </div>
  );
}
