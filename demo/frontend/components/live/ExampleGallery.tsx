import type { DemoExample } from "../../lib/types";
import { scenarioLabel, scenarioDescription } from "../../lib/presentation";

interface ExampleGalleryProps {
  examples: DemoExample[];
  selectedExample?: DemoExample;
  modified: boolean;
  onSelect: (example: DemoExample) => void;
}

export function ExampleGallery({ examples, selectedExample, modified, onSelect }: ExampleGalleryProps) {
  return (
    <div className="example-gallery-wrap">
      <div className="panel-row">
        <span className="section-label">1 · Choose a held-out example</span>
        {modified && <span className="status-tag is-warning">Modified</span>}
      </div>

      <ul className="example-gallery" role="list">
        {examples.map((example) => {
          const active = selectedExample?.sample_index === example.sample_index && !modified;
          const label = scenarioLabel[example.scenario] ?? example.scenario;
          const description = scenarioDescription[example.scenario];
          return (
            <li key={example.sample_index}>
              <button
                type="button"
                className={`example-card ${active ? "is-active" : ""}`}
                aria-pressed={active}
                onClick={() => onSelect(example)}
              >
                <span className="example-card-title">{label}</span>
                {description && <span className="example-card-desc">{description}</span>}
                <span className="example-card-meta">
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
