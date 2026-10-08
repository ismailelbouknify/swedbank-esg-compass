import type { ReviewState } from "../api";

// Symbol + text, so the state never depends on colour alone.
export const STATE: Record<ReviewState, { icon: string; label: string }> = {
  review: { icon: "!", label: "Review" },
  approved: { icon: "✓", label: "Approved" },
  changed: { icon: "✓", label: "Changed by analyst" },
  no_evidence: { icon: "—", label: "No evidence" },
};

export function StateIcon({ state }: { state: ReviewState }) {
  return (
    <span className={`state-icon ${state}`} aria-hidden="true">
      {STATE[state].icon}
    </span>
  );
}

export default function ReviewStateBadge({ state }: { state: ReviewState }) {
  return (
    <span className={`state-badge ${state}`}>
      <StateIcon state={state} /> {STATE[state].label}
    </span>
  );
}
