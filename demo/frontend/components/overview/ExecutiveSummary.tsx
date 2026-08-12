"use client";

import { useEffect, useState } from "react";
import { animate, useReducedMotion } from "framer-motion";
import { EXECUTIVE_STATS } from "../../lib/projectSummary";

interface ExecutiveSummaryProps {
  onNavigate: (view: "live" | "compare") => void;
}

/** Counts up to an integer stat value on mount; non-numeric values render as-is. */
function CountUpValue({ value }: { value: string }) {
  const numeric = Number(value.replace(/,/g, ""));
  const isNumeric = /^[\d,]+$/.test(value) && !Number.isNaN(numeric);
  const reduceMotion = useReducedMotion();
  const shouldAnimate = isNumeric && !reduceMotion;
  const [display, setDisplay] = useState("0");

  useEffect(() => {
    if (!shouldAnimate) return;
    const controls = animate(0, numeric, {
      duration: 0.9,
      ease: "easeOut",
      onUpdate: (current) => setDisplay(Math.round(current).toLocaleString())
    });
    return () => controls.stop();
  }, [shouldAnimate, numeric]);

  return <>{shouldAnimate ? display : value}</>;
}

export function ExecutiveSummary({ onNavigate }: ExecutiveSummaryProps) {
  return (
    <div className="executive-summary">
      <div className="overview-intro">
        <p className="overview-kicker">
          <span className="kicker-dot" aria-hidden="true" />
          EECS 3404 · finalized system
        </p>
        <h2>Compare four finalized classifiers on UNSW-NB15 traffic records.</h2>
        <p>
          Select a curated example or enter a traffic record to inspect model predictions against
          validation-selected decision thresholds. All four models are frozen; this interface performs
          inference only.
        </p>
        <div className="overview-actions">
          <button type="button" className="button button-primary" onClick={() => onNavigate("live")}>Live detection</button>
          <button type="button" className="button button-ghost" onClick={() => onNavigate("compare")}>Compare models</button>
        </div>
      </div>

      <dl className="headline-stats">
        {EXECUTIVE_STATS.map((stat) => (
          <div key={stat.label}>
            <dt>{stat.label}</dt>
            <dd>
              <CountUpValue value={stat.value} />
              <span className="stat-tick" aria-hidden="true" />
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
