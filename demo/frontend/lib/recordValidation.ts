import type { FeatureRecord, SchemaResponse } from "./types";

/**
 * Validate the browser-side record before it is JSON-serialized. The API repeats
 * these checks authoritatively, but keeping invalid browser state out of a
 * request prevents values such as `NaN` becoming JSON `null`.
 */
export function recordValidationError(record: FeatureRecord, schema: SchemaResponse | null): string | null {
  if (!schema) return "The input schema is still loading.";

  for (const field of schema.fields) {
    const value = record[field.name];
    if (value === undefined || value === "") return `${field.name} is required.`;

    if (field.kind === "categorical") {
      if (typeof value !== "string" || !value.trim()) return `${field.name} must be a category.`;
      continue;
    }

    if (typeof value !== "number" || !Number.isFinite(value)) return `${field.name} must be a finite number.`;
    if (field.type === "integer" && !Number.isInteger(value)) return `${field.name} must be an integer.`;
    if (field.non_negative && value < 0) return `${field.name} must be zero or greater.`;
  }

  return null;
}
