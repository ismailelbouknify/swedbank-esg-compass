import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";

export default function ExportMenu({ assessmentId }: { assessmentId: number }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent | KeyboardEvent) => {
      if (e instanceof KeyboardEvent ? e.key === "Escape" : !ref.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", close);
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("keydown", close);
    };
  }, [open]);

  return (
    <div className="menu" ref={ref}>
      <button type="button" className="btn" aria-haspopup="menu" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
        Export <span aria-hidden="true">▾</span>
      </button>
      {open && (
        <div className="menu-list" role="menu">
          <a role="menuitem" href={api.exportUrl(assessmentId, "xlsx")} onClick={() => setOpen(false)}>Excel</a>
          <a role="menuitem" href={api.exportUrl(assessmentId, "csv")} onClick={() => setOpen(false)}>CSV</a>
          <a role="menuitem" href={api.exportUrl(assessmentId, "json")} onClick={() => setOpen(false)}>JSON</a>
          <Link role="menuitem" to={`/assessments/${assessmentId}/report`} onClick={() => setOpen(false)}>PDF (print report)</Link>
        </div>
      )}
    </div>
  );
}
