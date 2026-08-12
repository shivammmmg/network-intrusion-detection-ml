type View = "overview" | "live" | "compare" | "analysis";

interface AppHeaderProps {
  view: View;
  onViewChange: (view: View) => void;
  serviceReady: boolean;
}

const NAV: Array<{ id: View; label: string }> = [
  { id: "overview", label: "Overview" },
  { id: "live", label: "Live Detection" },
  { id: "compare", label: "Compare Models" },
  { id: "analysis", label: "Analysis" }
];

export function AppHeader({ view, onViewChange, serviceReady }: AppHeaderProps) {
  return (
    <header className="app-header">
      <div className="app-brand">
        <p className="app-brand-eyebrow">EECS 3404</p>
        <h1>Intrusion Detection Inference Console</h1>
      </div>

      <nav className="app-nav" aria-label="Dashboard sections">
        {NAV.map((item) => (
          <button
            key={item.id}
            type="button"
            className={view === item.id ? "is-active" : ""}
            aria-pressed={view === item.id}
            onClick={() => onViewChange(item.id)}
          >
            {item.label}
          </button>
        ))}
      </nav>

      <div className={`status-line ${serviceReady ? "is-ready" : "is-offline"}`} role="status" aria-live="polite">
        <span className="status-dot" aria-hidden="true" />
        {serviceReady ? "Model service ready" : "Model service unavailable"}
      </div>
    </header>
  );
}
