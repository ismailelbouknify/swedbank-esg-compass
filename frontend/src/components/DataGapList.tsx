import type { Recommendation } from "../api";
import { StateIcon } from "./ReviewStateBadge";

/** Questions without usable evidence. Sources are listed once per gap only inside the review panel. */
export default function DataGapList({ gaps, onOpen }: { gaps: Recommendation[]; onOpen: (id: number) => void }) {
  if (gaps.length === 0) return <p className="muted">No data gaps.</p>;
  return (
    <ul className="gap-list">
      {gaps.map((r) => (
        <li key={r.id}>
          <div>
            <div className="gap-title">{r.factor === "reporting" ? "Reporting" : r.factor_label}</div>
            {r.factor !== "reporting" && <div className="gap-dim">{r.dimension_label}</div>}
            <div className="gap-reason">{r.reason}</div>
          </div>
          <div className="gap-action">
            {r.review_state !== "no_evidence" && <StateIcon state={r.review_state} />}
            <button type="button" className="link-btn" onClick={() => onOpen(r.id)}>
              Review <span className="sr-only">{r.factor_label} {r.dimension_label}</span> →
            </button>
          </div>
        </li>
      ))}
    </ul>
  );
}
