import type { Recommendation } from "../api";
import { STATE, StateIcon } from "./ReviewStateBadge";

const NO_EVIDENCE_TIP = "No evidence found in reviewed sources";
const INSUFFICIENT_TIP = "Insufficient source coverage";

/** One score: the level (or — when nothing was found) plus a single review-state symbol. */
export default function AssessmentCell({ r, onOpen }: { r?: Recommendation; onOpen: (id: number) => void }) {
  if (!r) return <span className="cell-empty">—</span>;
  const changed = r.final_level !== null && r.final_level !== r.recommended_level;
  const insufficient = r.evidence_status === "INSUFFICIENT_SOURCE_COVERAGE";
  const showDash = r.review_state === "no_evidence" || ((r.evidence_status === "NO_EVIDENCE_FOUND" || insufficient) && !changed);
  const gapTip = insufficient ? INSUFFICIENT_TIP : NO_EVIDENCE_TIP;
  const level = r.final_level ?? r.recommended_level;
  const label = `${r.factor_label}, ${r.dimension_label}: ${showDash ? gapTip : `level ${level} of ${r.max_level}`}. ${STATE[r.review_state].label}.`;
  return (
    <button type="button" className={`cell ${r.review_state}`} onClick={() => onOpen(r.id)} aria-label={label}
            title={showDash ? gapTip : STATE[r.review_state].label}>
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
