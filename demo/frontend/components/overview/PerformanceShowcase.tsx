import { FINAL_TEST_METRICS, PERFORMANCE_TAKEAWAY } from "../../lib/projectSummary";
import { MetricBarChart } from "./MetricBarChart";

type MetricKey = "prAuc" | "rocAuc" | "f1" | "recall";
const METRIC_COLUMNS: { key: MetricKey; label: string }[] = [
  { key: "prAuc", label: "PR-AUC" },
  { key: "rocAuc", label: "ROC-AUC" },
  { key: "f1", label: "F1" },
  { key: "recall", label: "Recall" }
];

const bestByMetric: Record<MetricKey, number> = {
  prAuc: Math.max(...FINAL_TEST_METRICS.map((row) => row.prAuc)),
  rocAuc: Math.max(...FINAL_TEST_METRICS.map((row) => row.rocAuc)),
  f1: Math.max(...FINAL_TEST_METRICS.map((row) => row.f1)),
  recall: Math.max(...FINAL_TEST_METRICS.map((row) => row.recall))
};

export function PerformanceShowcase() {
  return (
    <div className="performance-showcase">
      <MetricBarChart />
      <table className="performance-table">
        <thead>
          <tr>
            <th>Model</th>
            {METRIC_COLUMNS.map((column) => <th key={column.key}>{column.label}</th>)}
          </tr>
        </thead>
        <tbody>
          {FINAL_TEST_METRICS.map((row) => (
            <tr key={row.id} className={row.id === "random_forest" ? "is-selected" : undefined}>
              <td>
                {row.name}
                {row.id === "random_forest" && <span className="selected-tag">Selected</span>}
              </td>
              {METRIC_COLUMNS.map((column) => (
                <td key={column.key} className={row[column.key] === bestByMetric[column.key] ? "is-best" : undefined}>
                  {row[column.key].toFixed(4)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="performance-takeaway">{PERFORMANCE_TAKEAWAY}</p>
    </div>
  );
}
