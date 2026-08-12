import type { FeatureRecord } from "../../lib/types";
import { buildTrafficSummary } from "../../lib/trafficSummary";

interface TrafficSummaryProps {
  record: FeatureRecord;
  modified: boolean;
}

export function TrafficSummary({ record, modified }: TrafficSummaryProps) {
  const fields = buildTrafficSummary(record);

  return (
    <div className="traffic-summary-wrap">
      <div className="panel-row">
        <span className="section-label">2 · Selected traffic record</span>
        {modified && <span className="status-tag is-warning">Edited</span>}
      </div>
      <dl className="traffic-summary">
        {fields.map((field) => (
          <div key={field.label}>
            <dt>{field.label}</dt>
            <dd>{field.value}</dd>
            <span className="traffic-summary-raw">{field.rawField}</span>
          </div>
        ))}
      </dl>
    </div>
  );
}
