"use client";

import { useRef, useState } from "react";
import type { FigureEntry } from "../lib/analysisSummary";
import { ANALYSIS_SECTIONS } from "../lib/analysisSummary";
import { FigureGrid } from "./analysis/FigureGrid";
import { DriftTable } from "./analysis/DriftTable";
import { FigureLightbox } from "./analysis/FigureLightbox";

export function AnalysisPanel() {
  const [selectedFigure, setSelectedFigure] = useState<FigureEntry | null>(null);
  const triggerRef = useRef<HTMLElement | null>(null);

  function openFigure(figure: FigureEntry, trigger: HTMLButtonElement) {
    triggerRef.current = trigger;
    setSelectedFigure(figure);
  }

  return (
    <div className="analysis-panel">
      {ANALYSIS_SECTIONS.map((section) => (
        <section key={section.id} className="analysis-section" aria-labelledby={`analysis-${section.id}`}>
          <h3 id={`analysis-${section.id}`}>{section.title}</h3>
          <p>{section.summary}</p>
          {section.id === "drift" && <DriftTable />}
          {section.figures && <FigureGrid figures={section.figures} onOpen={openFigure} />}
        </section>
      ))}

      {selectedFigure && (
        <FigureLightbox figure={selectedFigure} restoreFocusRef={triggerRef} onClose={() => setSelectedFigure(null)} />
      )}
    </div>
  );
}
