import type { FeatureRecord, SchemaResponse } from "../lib/types";
import { Disclosure } from "./Disclosure";

const QUICK_FIELDS = ["proto", "service", "state"];

function formValue(value: FeatureRecord[string] | undefined): string {
  return typeof value === "number" && Number.isNaN(value) ? "" : String(value ?? "");
}

interface FeatureEditorProps {
  schema: SchemaResponse;
  record: FeatureRecord;
  onChange: (name: string, value: string, type: string) => void;
}

export function FeatureEditor({ schema, record, onChange }: FeatureEditorProps) {
  return (
    <div className="feature-editor">
      <fieldset className="quick-fields">
        <legend className="section-label">Primary fields</legend>
        <div className="field-grid">
          {QUICK_FIELDS.map((name) => (
            <label key={name} className="field">
              <span className="field-label">{name}</span>
              <select value={formValue(record[name])} onChange={(event) => onChange(name, event.target.value, "string")}>
                {schema.categorical_categories[name]?.map((option) => (
                  <option key={option} value={option}>{option}</option>
                ))}
              </select>
            </label>
          ))}
        </div>
      </fieldset>

      <Disclosure summary={`Edit all ${schema.fields.length} features`}>
        <div className="field-grid field-grid-dense">
          {schema.fields.map((field) => (
            <label key={field.name} className="field">
              <span className="field-label">{field.name}</span>
              {field.kind === "categorical" ? (
                <select value={formValue(record[field.name])} onChange={(event) => onChange(field.name, event.target.value, field.type)}>
                  {schema.categorical_categories[field.name]?.map((option) => (
                    <option key={option} value={option}>{option}</option>
                  ))}
                </select>
              ) : (
                <input
                  aria-label={field.name}
                  type="number"
                  min={field.non_negative ? 0 : undefined}
                  step={field.type === "integer" ? 1 : "any"}
                  value={formValue(record[field.name])}
                  onChange={(event) => onChange(field.name, event.target.value, field.type)}
                />
              )}
            </label>
          ))}
        </div>
      </Disclosure>
    </div>
  );
}
