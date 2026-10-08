import type { Questionnaire, Recommendation } from "../api";
import AssessmentCell from "./AssessmentCell";
import { StateIcon } from "./ReviewStateBadge";

const DIMS = ["planning", "execution", "performance"] as const;

export function ReportingRow({ r, onOpen }: { r?: Recommendation; onOpen: (id: number) => void }) {
  return (
    <section className="panel" aria-labelledby="reporting-h">
      <div className="panel-row">
        <h2 id="reporting-h" className="panel-title">Reporting</h2>
        <div className="reporting-cell">
          <AssessmentCell r={r} onOpen={onOpen} />
          {r && (
            <button type="button" className="link-btn" onClick={() => onOpen(r.id)}>
              Review →
            </button>
          )}
        </div>
      </div>
    </section>
  );
}

export default function AssessmentMatrix({ q, byKey, onOpen }: {
  q: Questionnaire | null;
  byKey: Map<string, Recommendation>;
  onOpen: (id: number) => void;
}) {
  const dimLabel = (d: string) => (d === "performance" ? "Performance" : q?.dimensions[d]?.label ?? d);
  return (
    <section className="panel" aria-labelledby="factors-h">
      <h2 id="factors-h" className="panel-title">Material factors</h2>
      <div className="table-scroll">
        <table className="matrix">
          <thead>
            <tr>
              <th scope="col"><span className="sr-only">Factor</span></th>
              {DIMS.map((d) => <th key={d} scope="col">{dimLabel(d)}</th>)}
            </tr>
          </thead>
          <tbody>
            {(q?.factors ?? []).map((f) => (
              <tr key={f.key}>
                <th scope="row">{f.short_label || f.label}</th>
                {DIMS.map((d) => (
                  <td key={d}>
                    <AssessmentCell r={byKey.get(`${f.key}:${d}`)} onOpen={onOpen} />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="legend">
        <span><StateIcon state="approved" /> Approved</span>
        <span><StateIcon state="review" /> Review</span>
        <span><span className="state-icon no_evidence" aria-hidden="true">—</span> No evidence</span>
      </p>
    </section>
  );
}
