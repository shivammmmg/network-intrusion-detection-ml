"use client";

import { motion, useReducedMotion } from "framer-motion";
import { EVIDENCE_HIGHLIGHTS } from "../../lib/projectSummary";
import { EyeIcon, WaveIcon, GaugeIcon, AlertIcon, DiceIcon, CheckIcon } from "../icons";

interface EvidenceGridProps {
  onNavigate: (view: "analysis") => void;
}

const CATEGORY_META: Record<string, { Icon: typeof EyeIcon; tone: "accent" | "warning" | "normal" }> = {
  Explainability: { Icon: EyeIcon, tone: "accent" },
  Drift: { Icon: WaveIcon, tone: "warning" },
  Calibration: { Icon: GaugeIcon, tone: "normal" },
  "TTL ablation": { Icon: AlertIcon, tone: "warning" },
  Uncertainty: { Icon: DiceIcon, tone: "accent" },
  Verification: { Icon: CheckIcon, tone: "normal" }
};

export function EvidenceGrid({ onNavigate }: EvidenceGridProps) {
  const reduceMotion = useReducedMotion();

  return (
    <div className="evidence-grid-wrap">
      <motion.div
        className="evidence-grid"
        initial={reduceMotion ? false : "hidden"}
        whileInView="visible"
        viewport={{ once: true, amount: 0.2 }}
        variants={{ hidden: {}, visible: { transition: { staggerChildren: 0.05 } } }}
      >
        {EVIDENCE_HIGHLIGHTS.map((item) => {
          const meta = CATEGORY_META[item.category] ?? { Icon: EyeIcon, tone: "accent" as const };
          const Icon = meta.Icon;
          return (
            <motion.div
              key={item.category}
              className={`evidence-tile tone-${meta.tone}`}
              variants={{ hidden: reduceMotion ? {} : { opacity: 0, y: 8 }, visible: { opacity: 1, y: 0 } }}
              transition={{ duration: 0.18 }}
            >
              <Icon className="evidence-icon" />
              <span className="section-label">{item.category}</span>
              <strong>{item.headline}</strong>
              {item.fraction && (
                <span className="evidence-fraction-track" aria-hidden="true">
                  <span
                    className="evidence-fraction-fill"
                    style={{ width: `${(item.fraction.numerator / item.fraction.denominator) * 100}%` }}
                  />
                </span>
              )}
              {item.lines.map((line) => <span key={line} className="evidence-line">{line}</span>)}
            </motion.div>
          );
        })}
      </motion.div>
      <button type="button" className="button button-ghost analysis-link" onClick={() => onNavigate("analysis")}>
        Explore full analysis →
      </button>
    </div>
  );
}
