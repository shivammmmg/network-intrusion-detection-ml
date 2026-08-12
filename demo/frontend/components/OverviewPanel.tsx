import type { HealthResponse } from "../lib/types";
import { ExecutiveSummary } from "./overview/ExecutiveSummary";
import { VisualPipeline } from "./overview/VisualPipeline";
import { PerformanceShowcase } from "./overview/PerformanceShowcase";
import { EvidenceGrid } from "./overview/EvidenceGrid";
import { FlowIcon, BarsIcon, SearchIcon } from "./icons";

interface OverviewPanelProps {
  health: HealthResponse | null;
  serviceReady: boolean;
  onNavigate: (view: "live" | "compare" | "analysis") => void;
}

export function OverviewPanel({ health, serviceReady, onNavigate }: OverviewPanelProps) {
  return (
    <section className="overview">
      {/* Section A — executive summary */}
      <ExecutiveSummary onNavigate={onNavigate} />

      {/* Section B — visual pipeline */}
      <div className="overview-section">
        <p className="overview-kicker"><FlowIcon className="kicker-icon" />Methodology</p>
        <VisualPipeline />
      </div>

      {/* Section C — model performance centerpiece */}
      <div className="overview-section">
        <p className="overview-kicker"><BarsIcon className="kicker-icon" />Frozen test evaluation</p>
        <PerformanceShowcase />
      </div>

      {/* Section D — evidence grid */}
      <div className="overview-section">
        <p className="overview-kicker"><SearchIcon className="kicker-icon" />Beyond model accuracy</p>
        <EvidenceGrid onNavigate={onNavigate} />
      </div>

      <p className="evaluation-snapshot">
        Evaluation snapshot <code>{health?.frozen_source_sha ? health.frozen_source_sha.slice(0, 10) : "loading…"}</code>
      </p>

      {!serviceReady && (
        <p className="hint-text is-warning">Live inference is currently offline; the summary above reflects the finalized project state.</p>
      )}
    </section>
  );
}
