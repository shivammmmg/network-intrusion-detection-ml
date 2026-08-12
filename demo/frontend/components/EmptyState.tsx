interface EmptyStateProps {
  mode: "single" | "compare";
  scenario?: string;
}

export function EmptyState({ mode, scenario }: EmptyStateProps) {
  return (
    <div className="empty-state">
      <p className="section-label">No prediction yet</p>
      <p>Select a curated record, review its active features if needed, then run {mode === "compare" ? "all four models" : "the selected model"}.</p>
      {scenario && (
        <p className="hint-text">
          Selected scenario: <strong>{scenario}</strong>
        </p>
      )}
    </div>
  );
}
