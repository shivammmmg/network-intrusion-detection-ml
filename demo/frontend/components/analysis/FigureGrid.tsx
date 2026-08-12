import type { FigureEntry } from "../../lib/analysisSummary";

interface FigureGridProps {
  figures: FigureEntry[];
  onOpen: (figure: FigureEntry, trigger: HTMLButtonElement) => void;
}

export function FigureGrid({ figures, onOpen }: FigureGridProps) {
  return (
    <div className="figure-grid">
      {figures.map((figure) => (
        <figure key={figure.src}>
          <button type="button" className="figure-trigger" onClick={(event) => onOpen(figure, event.currentTarget)}>
            {/* eslint-disable-next-line @next/next/no-img-element -- static, locally-served diagnostics assets; next/image optimization is unnecessary here */}
            <img src={figure.src} alt={figure.alt} width={figure.width} height={figure.height} loading="lazy" decoding="async" />
          </button>
          <figcaption>
            <span>{figure.caption}</span>
            <code>{figure.sourcePath}</code>
          </figcaption>
        </figure>
      ))}
    </div>
  );
}
