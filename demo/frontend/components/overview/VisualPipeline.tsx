"use client";

import { motion, useReducedMotion } from "framer-motion";
import { DATA_AUDIT, SPLITS, PREPROCESSING, MODEL_TUNING, LOCKED_THRESHOLDS } from "../../lib/projectSummary";

function fmt(n: number): string {
  return n.toLocaleString();
}

type StageBody =
  | { kind: "hero"; value: string; label: string; chips?: { value: string; label: string }[] }
  | { kind: "breakdown"; value: string; label: string; items: { value: string; label: string }[] }
  | { kind: "split"; segments: { value: number; label: string; tone: "a" | "b" | "c" }[] }
  | { kind: "transform"; from: { value: string; label: string }; to: { value: string; label: string }[]; tag: string }
  | { kind: "table"; columnLabel: string; rows: { name: string; value: string }[] };

interface Stage {
  title: string;
  body: StageBody;
  footnote?: string;
}

const STAGES: Stage[] = [
  {
    title: "Raw data",
    body: {
      kind: "hero",
      value: fmt(DATA_AUDIT.originalTotalRows),
      label: "original records",
      chips: [
        { value: fmt(DATA_AUDIT.originalTrainRows), label: "train" },
        { value: fmt(DATA_AUDIT.originalTestRows), label: "test" }
      ]
    }
  },
  {
    title: "Clean + audit",
    body: {
      kind: "breakdown",
      value: fmt(DATA_AUDIT.duplicateTrainRowsRemoved),
      label: "duplicate rows removed",
      items: [
        { value: String(DATA_AUDIT.ambiguousPredictorVectorsRemoved), label: "conflicting predictor vectors" },
        { value: fmt(DATA_AUDIT.ambiguousPredictorRowsRemoved), label: "conflicting rows removed" },
        { value: fmt(DATA_AUDIT.trainRowsOverlappingTestRemoved), label: "train rows overlapping test predictors" }
      ]
    }
  },
  {
    title: "Final splits",
    body: {
      kind: "split",
      segments: [
        { value: SPLITS.train, label: "train", tone: "a" },
        { value: SPLITS.validation, label: "validation", tone: "b" },
        { value: SPLITS.test, label: "frozen test", tone: "c" }
      ]
    },
    footnote: "Preprocessing fit on train only."
  },
  {
    title: "Preprocess",
    body: {
      kind: "transform",
      from: { value: String(PREPROCESSING.primaryInputFeatures), label: "primary inputs" },
      to: [
        { value: String(PREPROCESSING.linearTransformedFeatures), label: "linear / NN" },
        { value: String(PREPROCESSING.treeTransformedFeatures), label: "tree" }
      ],
      tag: "TTL excluded from primary pipeline"
    }
  },
  {
    title: "Model development",
    body: {
      kind: "table",
      columnLabel: "Round-2 fits",
      rows: MODEL_TUNING.map((model) => ({ name: model.name, value: String(model.round2Configurations) }))
    },
    footnote: "Selected by validation PR-AUC after a Round-1 sweep."
  },
  {
    title: "Validation lock",
    body: {
      kind: "table",
      columnLabel: "Locked threshold",
      rows: LOCKED_THRESHOLDS.map((threshold) => ({ name: threshold.name, value: threshold.value.toFixed(4) }))
    },
    footnote: "Selected on validation data only; frozen before touching test."
  },
  {
    title: "Frozen test",
    body: {
      kind: "hero",
      value: fmt(SPLITS.test),
      label: "untouched test rows"
    },
    footnote: "No threshold tuning on test."
  }
];

function StageBodyView({ body }: { body: StageBody }) {
  switch (body.kind) {
    case "hero":
      return (
        <>
          <div className="stage-metrics">
            <div>
              <strong>{body.value}</strong>
              <span>{body.label}</span>
            </div>
          </div>
          {body.chips && (
            <div className="methodology-chip-row">
              {body.chips.map((chip) => (
                <span key={chip.label} className="methodology-chip">
                  <strong>{chip.value}</strong> {chip.label}
                </span>
              ))}
            </div>
          )}
        </>
      );

    case "breakdown":
      return (
        <>
          <div className="stage-metrics">
            <div>
              <strong>{body.value}</strong>
              <span>{body.label}</span>
            </div>
          </div>
          <div className="methodology-breakdown">
            {body.items.map((item) => (
              <div key={item.label} className="methodology-breakdown-row">
                <strong>{item.value}</strong>
                <span>{item.label}</span>
              </div>
            ))}
          </div>
        </>
      );

    case "split": {
      const total = body.segments.reduce((sum, segment) => sum + segment.value, 0);
      return (
        <>
          <div className="methodology-split-bar">
            {body.segments.map((segment) => (
              <span
                key={segment.label}
                className={`methodology-split-segment tone-${segment.tone}`}
                style={{ width: `${(segment.value / total) * 100}%` }}
              />
            ))}
          </div>
          <div className="methodology-split-legend">
            {body.segments.map((segment) => (
              <span key={segment.label} className="methodology-split-legend-item">
                <span className={`methodology-split-swatch tone-${segment.tone}`} aria-hidden="true" />
                <strong>{fmt(segment.value)}</strong> {segment.label}
              </span>
            ))}
          </div>
        </>
      );
    }

    case "transform":
      return (
        <>
          <div className="methodology-transform">
            <div className="methodology-transform-box">
              <strong>{body.from.value}</strong>
              <span>{body.from.label}</span>
            </div>
            <span className="methodology-transform-arrow" aria-hidden="true">→</span>
            {body.to.map((item) => (
              <div key={item.label} className="methodology-transform-box">
                <strong>{item.value}</strong>
                <span>{item.label}</span>
              </div>
            ))}
          </div>
          <span className="methodology-transform-tag">{body.tag}</span>
        </>
      );

    case "table":
      return (
        <div className="methodology-table">
          <div className="methodology-table-head">
            <span>Model</span>
            <span>{body.columnLabel}</span>
          </div>
          {body.rows.map((row) => (
            <div key={row.name} className="methodology-table-row">
              <span>{row.name}</span>
              <strong>{row.value}</strong>
            </div>
          ))}
        </div>
      );
  }
}

interface StackStepProps {
  stage: Stage;
  index: number;
  total: number;
}

function StackStep({ stage, index, total }: StackStepProps) {
  const reduceMotion = useReducedMotion();
  const tilt = index % 2 === 0 ? -1.2 : 1.2;

  return (
    <div className="methodology-step">
      <div className="methodology-step-sticky" style={{ top: `${88 + index * 8}px` }}>
        <motion.div
          className="methodology-card"
          initial={reduceMotion ? false : { opacity: 0, y: 36, scale: 0.96 }}
          whileInView={{ opacity: 1, y: 0, scale: 1 }}
          viewport={{ once: true, amount: 0.5 }}
          transition={{ duration: 0.45, ease: "easeOut" }}
          style={{ rotate: tilt }}
        >
          <div className="methodology-card-head">
            <span className="methodology-index" aria-hidden="true">{String(index + 1).padStart(2, "0")}</span>
            <div>
              <span className="section-label">Stage {index + 1} of {total}</span>
              <h4>{stage.title}</h4>
            </div>
          </div>
          <StageBodyView body={stage.body} />
          {stage.footnote && <p className="methodology-card-line">{stage.footnote}</p>}
        </motion.div>
      </div>
    </div>
  );
}

export function VisualPipeline() {
  return (
    <div className="methodology-stack">
      {STAGES.map((stage, index) => (
        <StackStep key={stage.title} stage={stage} index={index} total={STAGES.length} />
      ))}
    </div>
  );
}
