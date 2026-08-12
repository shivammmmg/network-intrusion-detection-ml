import type { DemoExample } from "../../lib/types";
import { scenarioLabel, scenarioDescription } from "../../lib/presentation";

interface ExampleGalleryProps {
  examples: DemoExample[];
  selectedExample?: DemoExample;
  modified: boolean;
  onSelect: (example: DemoExample) => void;
  onSelectCustom: () => void;
}

export function ExampleGallery({ examples, selectedExample, modified, onSelect, onSelectCustom }: ExampleGalleryProps) {
  const customActive = !selectedExample;
  const attackExampleIndex = examples.findIndex((example) => example.scenario === "high_confidence_attack");
  const customInsertIndex = attackExampleIndex === -1 ? examples.length : attackExampleIndex + 1;
  const galleryItems: Array<DemoExample | null> = [
    ...examples.slice(0, customInsertIndex),
    null,
    ...examples.slice(customInsertIndex)
  ];

  return (
    <div className="example-gallery-wrap">
      <div className="panel-row">
        <span className="section-label">1 · Choose a held-out example</span>
        {modified && <span className="status-tag is-warning">Modified</span>}
      </div>

      <ul className="example-gallery" role="list">
        {galleryItems.map((example) => {
          if (!example) {
            return (
              <li key="custom-traffic-record">
                <button
                  type="button"
                  className={`example-card ${customActive ? "is-active" : ""}`}
                  aria-pressed={customActive}
                  onClick={onSelectCustom}
                >
                  <span className="example-card-title">Custom traffic record</span>
                  <span className="example-card-desc">Enter all 39 feature values to evaluate your own traffic record.</span>
                  <span className="example-card-meta">Manual input · no held-out label</span>
                </button>
              </li>
            );
          }

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
      {!selectedExample && (
        <p className="hint-text">Custom records have no held-out label; enter all feature values before running live inference.</p>
      )}
    </div>
  );
}
