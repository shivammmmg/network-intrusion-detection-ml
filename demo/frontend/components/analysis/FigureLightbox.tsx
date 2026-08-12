import type { RefObject } from "react";
import type { FigureEntry } from "../../lib/analysisSummary";
import { useFocusTrap } from "../../hooks/useFocusTrap";

interface FigureLightboxProps {
  figure: FigureEntry;
  restoreFocusRef: RefObject<HTMLElement | null>;
  onClose: () => void;
}

export function FigureLightbox({ figure, restoreFocusRef, onClose }: FigureLightboxProps) {
  const { dialogRef, closeRef } = useFocusTrap(true, onClose, restoreFocusRef);

  return (
    <div className="dialog-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
      <section className="lightbox" role="dialog" aria-modal="true" aria-label={figure.alt} ref={dialogRef}>
        <button type="button" className="dialog-close lightbox-close" onClick={onClose} ref={closeRef} aria-label="Close figure">
          <span aria-hidden="true">×</span>
        </button>
        {/* eslint-disable-next-line @next/next/no-img-element -- static, locally-served diagnostics assets */}
        <img src={figure.src} alt={figure.alt} width={figure.width} height={figure.height} />
        <figcaption className="lightbox-caption">
          <span>{figure.caption}</span>
          <code>{figure.sourcePath}</code>
        </figcaption>
      </section>
    </div>
  );
}
