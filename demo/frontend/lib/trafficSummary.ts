import type { FeatureRecord } from "./types";

export interface TrafficSummaryField {
  label: string;
  value: string;
  rawField: string;
}

function readRaw(record: FeatureRecord, key: string): FeatureRecord[string] | undefined {
  return record[key];
}

function readString(record: FeatureRecord, key: string): string {
  const value = readRaw(record, key);
  if (value === undefined || value === null || value === "") return "—";
  return String(value).toUpperCase();
}

function readNumber(record: FeatureRecord, key: string): number | undefined {
  const value = readRaw(record, key);
  const numeric = typeof value === "number" ? value : Number(value);
  return Number.isFinite(numeric) ? numeric : undefined;
}

function formatDuration(value: number | undefined): string {
  if (value === undefined) return "—";
  return `${value.toFixed(2)} s`;
}

function formatBytes(value: number | undefined): string {
  if (value === undefined) return "—";
  if (value >= 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${Math.round(value)} B`;
}

function formatCount(value: number | undefined): string {
  return value === undefined ? "—" : String(Math.round(value));
}

/**
 * Human-readable summary of the record's most understandable raw fields —
 * a display transform only. Every value is read directly from the selected
 * record, nothing is invented, estimated, or recomputed.
 */
export function buildTrafficSummary(record: FeatureRecord): TrafficSummaryField[] {
  const spkts = readNumber(record, "spkts");
  const dpkts = readNumber(record, "dpkts");
  const sbytes = readNumber(record, "sbytes");
  const dbytes = readNumber(record, "dbytes");

  return [
    { label: "Protocol", value: readString(record, "proto"), rawField: "proto" },
    { label: "Service", value: readString(record, "service"), rawField: "service" },
    { label: "State", value: readString(record, "state"), rawField: "state" },
    { label: "Duration", value: formatDuration(readNumber(record, "dur")), rawField: "dur" },
    { label: "Packets", value: `${formatCount(spkts)} → ${formatCount(dpkts)}`, rawField: "spkts / dpkts" },
    { label: "Bytes", value: `${formatBytes(sbytes)} → ${formatBytes(dbytes)}`, rawField: "sbytes / dbytes" }
  ];
}
