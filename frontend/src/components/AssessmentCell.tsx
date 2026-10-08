import type { Recommendation } from "../api";
import { STATE, StateIcon } from "./ReviewStateBadge";

const NO_EVIDENCE_TIP = "No evidence found in reviewed sources";

/** One score: the level (or — when nothing was found) plus a single review-state symbol. */
export default function AssessmentCell({ r, onOpen }: { r?: Recommendation; onOpen: (id: number) => void }) {
  if (!r) return <span className="cell-empty">—</span>;
  const changed = r.final_level !== null && r.final_level !== r.recommended_level;
  const showDash = r.review_state === "no_evidence" || (r.evidence_status === "NO_EVIDENCE_FOUND" && !changed);
  const level = r.final_level ?? r.recommended_level;
  const label = `${r.factor_label}, ${r.dimension_label}: ${showDash ? NO_EVIDENCE_TIP : `level ${level} of ${r.max_level}`}. ${STATE[r.review_state].label}.`;
  return (
    <button type="button" className={`cell ${r.review_state}`} onClick={() => onOpen(r.id)} aria-label={label}
            title={showDash ? NO_EVIDENCE_TIP : STATE[r.review_state].label}>
      {showDash ? (
        <span className="cell-score none">—</span>
      ) : (
        <span className="cell-score">
          {level} <span className="of">/ {r.max_level}</span>
        </span>
      )}
      {r.review_state !== "no_evidence" && <StateIcon state={r.review_state} />}
    </button>
  );
}
