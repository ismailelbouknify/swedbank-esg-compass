import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, type AssessmentSummary, type Questionnaire, type RecommendationDetail, type Source } from "../api";
import { LOGO_SRC, PRODUCT_NAME, PRODUCT_SUBTITLE, useDocumentTitle, useLogoAvailable } from "../brand";
import { STATE } from "../components/ReviewStateBadge";
import { formatDate, formatDateTime } from "../format";

const DIMS = ["planning", "execution", "performance"] as const;

function levelText(r: RecommendationDetail | undefined): string {
  if (!r) return "—";
  const changed = r.final_level !== null && r.final_level !== r.recommended_level;
  if (r.evidence_status === "NO_EVIDENCE_FOUND" && !changed) return "—";
  return `${r.final_level ?? r.recommended_level} / ${r.max_level}`;
}

const question = (r: RecommendationDetail) => (r.factor === "reporting" ? "Reporting" : `${r.factor_label} — ${r.dimension_label}`);

/** A static, print-first layout (not the interactive screen) for "Save as PDF". */
export default function AssessmentReport() {
  const { id } = useParams();
  const aid = Number(id);
  const [summary, setSummary] = useState<AssessmentSummary | null>(null);
  const [details, setDetails] = useState<RecommendationDetail[]>([]);
  const [sources, setSources] = useState<Source[]>([]);
  const [q, setQ] = useState<Questionnaire | null>(null);
  const [error, setError] = useState<string | null>(null);
  const logo = useLogoAvailable();
  // also the default file name when the browser saves the report as PDF
  useDocumentTitle(summary ? `${summary.assessment.company.name} ${summary.assessment.year}` : null);

  useEffect(() => {
    (async () => {
      try {
        const [s, recs, so, qq] = await Promise.all([api.summary(aid), api.recommendations(aid), api.sources(aid), api.questionnaire()]);
        const ds = await Promise.all(recs.map((r) => api.recommendation(r.id)));
        setSummary(s); setSources(so); setQ(qq); setDetails(ds);
      } catch (e) {
        setError((e as Error).message);
      }
    })();
  }, [aid]);

  if (error) return <div className="page"><div className="notice error" role="alert">{error}</div></div>;
  if (!summary || !q) return <div className="page"><p className="muted">Preparing report…</p></div>;

  const a = summary.assessment;
  const byKey = new Map(details.map((r) => [`${r.factor}:${r.dimension}`, r]));
  const reporting = byKey.get("reporting:reporting");
  const withEvidence = details.filter((r) => r.evidence.length > 0);
  const gaps = details.filter((r) => r.evidence_status === "NO_EVIDENCE_FOUND");
  const decisions = details.flatMap((r) => r.decisions.map((d) => ({ r, d }))).sort((x, y) => x.d.id - y.d.id);
  const processed = sources.filter((s) => s.status === "PROCESSED");

  return (
    <div className="report">
      <div className="report-toolbar no-print">
        <Link to={`/assessments/${aid}`}>← Back to assessment</Link>
        <button type="button" className="btn primary" onClick={() => window.print()}>Print / Save as PDF</button>
        <span className="muted small">Tip: switch off “Headers and footers” in the print dialog to hide the page address.</span>
      </div>

      <header className="report-head">
        <div className="report-brand">
          {logo && <img className="report-logo" src={LOGO_SRC} alt="Swedbank" />}
          <div>
            <h1 className="report-product">{PRODUCT_NAME}</h1>
            <div className="report-subtitle">{PRODUCT_SUBTITLE}</div>
          </div>
        </div>
        <dl className="report-cover">
          <div><dt>Company</dt><dd>{a.company.name}</dd></div>
          <div><dt>Assessment year</dt><dd>{a.year}</dd></div>
          <div><dt>Last analysed</dt><dd>{formatDate(a.completed_at)}</dd></div>
          <div><dt>Printed</dt><dd>{formatDate(new Date().toISOString())}</dd></div>
        </dl>
      </header>

      <section className="report-section">
        <h2>Assessment summary</h2>
        <table className="report-table kv">
          <tbody>
            <tr><th>Sources reviewed</th><td>{summary.sources_reviewed}</td></tr>
            <tr><th>Answers reviewed by analyst</th><td>{summary.reviewed} / {summary.total}</td></tr>
            <tr><th>Data gaps</th><td>{summary.data_gaps}</td></tr>
            {a.company.organisation_number && <tr><th>Organisation number</th><td>{a.company.organisation_number}</td></tr>}
          </tbody>
        </table>
      </section>

      <section className="report-section">
        <h2>Reporting</h2>
        {reporting && (
          <p><strong>{levelText(reporting)}</strong> · {STATE[reporting.review_state].label} — {reporting.reason}</p>
        )}
      </section>

      <section className="report-section">
        <h2>Material factors</h2>
        <table className="report-table">
          <thead>
            <tr><th>Factor</th>{DIMS.map((d) => <th key={d}>{d === "performance" ? "Performance" : q.dimensions[d].label}</th>)}</tr>
          </thead>
          <tbody>
            {q.factors.map((f) => (
              <tr key={f.key}>
                <th>{f.short_label || f.label}</th>
                {DIMS.map((d) => {
                  const r = byKey.get(`${f.key}:${d}`);
                  return (
                    <td key={d}>
                      {levelText(r)}
                      {r && <span className="report-state">{r.review_state === "no_evidence" ? "No evidence" : `${STATE[r.review_state].icon} ${STATE[r.review_state].label}`}</span>}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="report-section">
        <h2>Evidence</h2>
        {withEvidence.map((r) => (
          <div key={r.id} className="report-item">
            <h3>{question(r)} · {levelText(r)}</h3>
            <p>{r.reason}</p>
            {r.evidence.slice(0, 3).map((e) => (
              <blockquote key={e.id}>
                “{e.evidence_text}”
                <cite>{e.source_title || e.source_url}{e.page_number ? `, page ${e.page_number}` : ""}{!e.verified ? " (quote to be checked)" : ""}</cite>
              </blockquote>
            ))}
          </div>
        ))}
      </section>

      <section className="report-section">
        <h2>Data gaps</h2>
        {gaps.length === 0 ? <p>No data gaps.</p> : (
          <ul>{gaps.map((r) => <li key={r.id}><strong>{question(r)}:</strong> {r.reason}</li>)}</ul>
        )}
      </section>

      <section className="report-section">
        <h2>Analyst decisions</h2>
        {decisions.length === 0 ? <p>No analyst decisions yet.</p> : (
          <table className="report-table">
            <thead><tr><th>Question</th><th>Decision</th><th>Recommended</th><th>Final</th><th>Comment</th><th>By / date</th></tr></thead>
            <tbody>
              {decisions.map(({ r, d }) => (
                <tr key={d.id}>
                  <td>{question(r)}</td>
                  <td>{d.action === "APPROVE" ? "Approved" : "Changed"}</td>
                  <td>{r.recommended_level}</td>
                  <td>{d.selected_level}</td>
                  <td>{d.comment ?? ""}</td>
                  <td>{d.analyst || "Analyst"}, {formatDateTime(d.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <section className="report-section">
        <h2>Sources</h2>
        <ol>
          {processed.map((s) => (
            <li key={s.id}>{s.title || s.original_filename} <span className="muted">— {s.source_type_label}</span>{s.url && <div className="report-url">{s.url}</div>}</li>
          ))}
        </ol>
      </section>
    </div>
  );
}
