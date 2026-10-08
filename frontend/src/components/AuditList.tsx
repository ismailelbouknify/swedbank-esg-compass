import type { AuditEvent, Recommendation } from "../api";
import { formatDateTime } from "../format";

// Only business events are shown; low-level processing events stay in the backend audit trail.
function describe(ev: AuditEvent, recs: Map<number, Recommendation>): string | null {
  const d = ev.details as Record<string, unknown>;
  const rec = ev.entity_id !== null ? recs.get(ev.entity_id) : undefined;
  const q = rec ? (rec.factor === "reporting" ? "Reporting" : `${rec.factor_label} · ${rec.dimension_label}`) : "an answer";
  switch (ev.event_type) {
    case "assessment_created":
      return d.rerun_of || d.supersedes_id ? "New analysis requested (new version created)" : "Assessment created";
    case "document_uploaded":
      return `Report uploaded${d.filename ? `: ${d.filename}` : ""}`;
    case "analysis_started":
      return "Analysis started";
    case "assessment_completed":
      return "Analysis completed";
    case "assessment_failed":
      return "Analysis could not be completed";
    case "recommendation_approved":
      return `Approved ${q} (level ${d.selected_level})`;
    case "recommendation_overridden":
      return `Changed ${q}: ${d.previous_level} → ${d.selected_level}${d.comment ? ` — “${d.comment}”` : ""}`;
    case "new_analysis_requested":
      return "New analysis started for this assessment";
    case "assessment_replaced":
      return "Replaced by a newer analysis";
    case "export_generated":
      return `Exported (${String(d.format ?? "").toUpperCase()})`;
    default:
      return null;
  }
}

export default function AuditList({ events, recs }: { events: AuditEvent[]; recs: Recommendation[] }) {
  const byId = new Map(recs.map((r) => [r.id, r]));
  const rows = events.map((ev) => ({ ev, text: describe(ev, byId) })).filter((x) => x.text).reverse();
  if (rows.length === 0) return <p className="muted">No activity yet.</p>;
  return (
    <ul className="audit-list">
      {rows.map(({ ev, text }) => (
        <li key={ev.id}>
          <span className="audit-time">{formatDateTime(ev.created_at)}</span>
          <span className="audit-text">{text}</span>
          <span className="audit-actor">{ev.actor === "system" ? "" : ev.actor}</span>
        </li>
      ))}
    </ul>
  );
}
