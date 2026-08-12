import type { DemoExample } from "../lib/types";
import { scenarioLabel } from "../lib/presentation";

interface ExampleSelectorProps {
  examples: DemoExample[];
  selectedExample?: DemoExample;
  modified: boolean;
  onSelect: (example: DemoExample) => void;
}

export function ExampleSelector({ examples, selectedExample, modified, onSelect }: ExampleSelectorProps) {
  return (
    <div className="example-selector">
      <div className="panel-row">
        <span className="section-label">Curated held-out examples</span>
        {modified && <span className="status-tag is-warning">Modified</span>}
      </div>

      <ul className="example-list" role="list">
        {examples.map((example) => {
          const active = selectedExample?.sample_index === example.sample_index && !modified;
          return (
            <li key={example.sample_index}>
              <button
                type="button"
                className={`example-row ${active ? "is-active" : ""}`}
                aria-pressed={active}
                onClick={() => onSelect(example)}
              >
                <span className="example-row-label">{scenarioLabel[example.scenario] ?? example.scenario}</span>
                <span className="example-row-meta">
                  Held-out label: {example.evaluation_metadata.known_held_out_label ? "Attack" : "Normal"}
                </span>
              </button>
            </li>
          );
        })}
      </ul>

      {selectedExample && !modified && (
        <p className="hint-text">Held-out label is evaluation metadata only. It is never sent to a model.</p>
      )}
      {selectedExample && modified && (
        <p className="hint-text is-warning">Ground-truth comparison disabled — this record has been edited.</p>
      )}
    </div>
  );
}
