const OFFLINE_FIELDS = ["Protocol", "Service", "Connection state"];

export function OfflineWorkspaceNotice() {
  return (
    <section className="offline-notice">
      <div className="offline-notice-block">
        <p className="section-label">Input contract</p>
        <h3>Traffic record input</h3>
        <p>Feature controls become available when the model artifact service is restored.</p>
        <div className="field-grid">
          {OFFLINE_FIELDS.map((field) => (
            <label key={field} className="field">
              <span className="field-label">{field}</span>
              <select disabled>
                <option>Unavailable</option>
              </select>
            </label>
          ))}
        </div>
        <button type="button" className="button button-primary" disabled>Run prediction</button>
      </div>

      <div className="offline-notice-block">
        <p className="section-label">Prediction result</p>
        <h3>Live model service unavailable</h3>
        <p>Predictions will appear here once the model service is restored.</p>
      </div>
    </section>
  );
}
