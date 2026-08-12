import { percentage } from "../lib/presentation";

interface ProbabilityMeterProps {
  probability: number;
  threshold: number;
}

export function ProbabilityMeter({ probability, threshold }: ProbabilityMeterProps) {
  const fillWidth = Math.min(100, Math.max(0, probability * 100));
  const markerPosition = Math.min(100, Math.max(0, threshold * 100));

  return (
    <div
      className="probability-meter"
      role="img"
      aria-label={`Attack probability ${percentage(probability)} against a locked threshold of ${percentage(threshold)}`}
    >
      <div className="probability-meter-track">
        <div className="probability-meter-fill" style={{ width: `${fillWidth}%` }} />
        <div className="probability-meter-threshold" style={{ left: `${markerPosition}%` }} />
      </div>
    </div>
  );
}
