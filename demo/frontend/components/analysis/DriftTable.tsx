import { TOP_DRIFT_FEATURES } from "../../lib/analysisSummary";

export function DriftTable() {
  return (
    <table className="stage-table">
      <thead>
        <tr><th>Feature</th><th>PSI</th><th>KS statistic</th></tr>
      </thead>
      <tbody>
        {TOP_DRIFT_FEATURES.map((row) => (
          <tr key={row.feature}>
            <td>{row.feature}</td>
            <td>{row.psi.toFixed(4)}</td>
            <td>{row.ks.toFixed(4)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
