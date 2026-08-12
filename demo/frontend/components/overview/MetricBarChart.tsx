"use client";

import { motion, useReducedMotion } from "framer-motion";
import { FINAL_TEST_METRICS } from "../../lib/projectSummary";

// Bars are scaled from a floor just below the lowest score (not from 0) so the
// real spread between models is visible; PR-AUC's theoretical ceiling is 1.0.
const VALUES = FINAL_TEST_METRICS.map((row) => row.prAuc);
const FLOOR = Math.max(0, Math.floor((Math.min(...VALUES) - 0.02) * 100) / 100);
const CEIL = 1;
const BEST = Math.max(...VALUES);

export function MetricBarChart() {
  const reduceMotion = useReducedMotion();

  return (
    <div className="metric-bar-chart" role="img" aria-label="PR-AUC comparison across the four locked models, ranked highest to lowest">
      <div className="metric-bar-chart-head">
        <span className="section-label">Ranking metric — PR-AUC</span>
        <span className="metric-bar-chart-scale">{FLOOR.toFixed(2)} – {CEIL.toFixed(2)}</span>
      </div>
      {[...FINAL_TEST_METRICS]
        .sort((a, b) => b.prAuc - a.prAuc)
        .map((row) => {
          const widthPct = ((row.prAuc - FLOOR) / (CEIL - FLOOR)) * 100;
          const isBest = row.prAuc === BEST;
          return (
            <div key={row.id} className="metric-bar-row">
              <span className="metric-bar-label">{row.name}</span>
              <span className="metric-bar-track">
                <motion.span
                  className={isBest ? "metric-bar-fill is-best" : "metric-bar-fill"}
                  initial={reduceMotion ? false : { width: 0 }}
                  whileInView={{ width: `${widthPct}%` }}
                  viewport={{ once: true, amount: 0.4 }}
                  transition={{ duration: 0.5, ease: "easeOut" }}
                  style={reduceMotion ? { width: `${widthPct}%` } : undefined}
                />
              </span>
              <span className="metric-bar-value">{row.prAuc.toFixed(4)}</span>
            </div>
          );
        })}
    </div>
  );
}
