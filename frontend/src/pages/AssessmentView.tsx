import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";
import {
  api, type AssessmentSummary, type AuditEvent, type Questionnaire, type Recommendation, type Source,
} from "../api";
import AssessmentMatrix, { ReportingRow } from "../components/AssessmentMatrix";
import AuditList from "../components/AuditList";
import ConfirmDialog from "../components/ConfirmDialog";
import DataGapList from "../components/DataGapList";
import ExportMenu from "../components/ExportMenu";
import ReviewDrawer from "../components/ReviewDrawer";
import SourceList from "../components/SourceList";
import { useDocumentTitle } from "../brand";
import { STATUS_LABEL, formatDate } from "../format";

const DIM_ORDER = ["reporting", "planning", "execution", "performance"];
type Tab = "assessment" | "sources" | "audit";

export default function AssessmentView() {
  const { id } = useParams();
  const aid = Number(id);
  const navigate = useNavigate();
  const [summary, setSummary] = useState<AssessmentSummary | null>(null);
  const [recs, setRecs] = useState<Recommendation[]>([]);
  const [q, setQ] = useState<Questionnaire | null>(null);
  const [sources, setSources] = useState<Source[] | null>(null);
  const [audit, setAudit] = useState<AuditEvent[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showGaps, setShowGaps] = useState(false);
  const [confirmRerun, setConfirmRerun] = useState(false);
  const [rerunBusy, setRerunBusy] = useState(false);
  useDocumentTitle(summary?.assessment.company.name ?? null);

  // open review item and tab live in the URL so they can be linked to
  const [params, setParams] = useSearchParams();
  const selected = params.get("rec") ? Number(params.get("rec")) : null;
  const tab = (params.get("tab") as Tab) || "assessment";
  const setParam = (key: string, value: string | null) =>
    setParams((p) => { if (value === null) p.delete(key); else p.set(key, value); return p; }, { replace: key === "rec" });

  const reload = useCallback(() => {
    Promise.all([api.summary(aid), api.recommendations(aid)])
      .then(([s, r]) => { setSummary(s); setRecs(r); })
      .catch((e: Error) => setError(e.message));
  }, [aid]);

  useEffect(() => {
    reload();
    api.questionnaire().then(setQ).catch(() => undefined);
  }, [reload]);

  useEffect(() => {
    if (tab === "sources" && sources === null) api.sources(aid).then(setSources).catch((e: Error) => setError(e.message));
    if (tab === "audit") api.audit(aid).then(setAudit).catch((e: Error) => setError(e.message));
  }, [tab, aid, sources]);

  const byKey = useMemo(() => new Map(recs.map((r) => [`${r.factor}:${r.dimension}`, r])), [recs]);
  // review order follows the screen: reporting, then factor by factor
  const ordered = useMemo(() => {
    const fIdx = new Map((q?.factors ?? []).map((f, i) => [f.key, i]));
    return [...recs].sort((a, b) =>
      (a.factor === "reporting" ? -1 : fIdx.get(a.factor) ?? 99) - (b.factor === "reporting" ? -1 : fIdx.get(b.factor) ?? 99)
      || DIM_ORDER.indexOf(a.dimension) - DIM_ORDER.indexOf(b.dimension));
  }, [recs, q]);
  const pending = ordered.filter((r) => r.review_status === "PENDING");
  // items with evidence first; data gaps after
  const reviewQueue = [...pending.filter((r) => r.review_state === "review"), ...pending.filter((r) => r.review_state === "no_evidence")];
  const gaps = ordered.filter((r) => r.evidence_status === "NO_EVIDENCE_FOUND");

  // the next pending item after the open one (wrapping around); an item just approved has left the queue
  const nextAfter = (current: number | null): Recommendation | undefined => {
    const i = reviewQueue.findIndex((r) => r.id === current);
    return [...reviewQueue.slice(i + 1), ...reviewQueue.slice(0, Math.max(i, 0))][0];
  };

  async function rerun() {
    setRerunBusy(true);
    try {
      const a = await api.rerun(aid);
      navigate(`/assessments/${a.id}/progress`);
    } catch (e) {
      setError((e as Error).message);
      setRerunBusy(false);
      setConfirmRerun(false);
    }
  }

  if (error && !summary) return <div className="page"><div className="notice error" role="alert">{error}</div></div>;
  if (!summary) return <div className="page"><p className="muted">Loading…</p></div>;
  const a = summary.assessment;

  if (a.status !== "COMPLETED") {
    return (
      <div className="page narrow">
        <h1>{a.company.name}</h1>
        <p className="muted">Assessment year {a.year} · {STATUS_LABEL[a.status]}</p>
        <Link className="btn primary" to={`/assessments/${aid}/progress`}>View analysis</Link>
      </div>
    );
  }

  const next = nextAfter(selected);
  const tabs: { key: Tab; label: string }[] = [
    { key: "assessment", label: "Assessment" },
    { key: "sources", label: "Sources" },
    { key: "audit", label: "Audit" },
  ];

  return (
    <div className="page">
      {a.superseded_by_id && (
        <div className="notice">
          This is an earlier version of this assessment. <Link to={`/assessments/${a.superseded_by_id}`}>Open the current assessment</Link>
        </div>
      )}
      {summary.newer_version_id && (
        <div className="notice">
          A new analysis of this company is running. <Link to={`/assessments/${summary.newer_version_id}/progress`}>View progress</Link>
        </div>
      )}

      <header className="summary">
        <div>
          <h1>{a.company.name}</h1>
          <p className="summary-meta">
            Assessment year {a.year}
            <span className="dot" aria-hidden="true">·</span>
            {summary.sources_reviewed} {summary.sources_reviewed === 1 ? "source" : "sources"} reviewed
            <span className="dot" aria-hidden="true">·</span>
            Last analysed {formatDate(a.completed_at)}
          </p>
        </div>
        <div className="summary-side">
          <div className="review-progress">
            <strong>{summary.reviewed} / {summary.total}</strong> reviewed
            <div className="bar" aria-hidden="true"><div style={{ width: `${summary.total ? (100 * summary.reviewed) / summary.total : 0}%` }} /></div>
          </div>
          <div className="summary-actions">
            {reviewQueue.length > 0 ? (
              <button type="button" className="btn primary" onClick={() => setParam("rec", String(reviewQueue[0].id))}>
                Continue review
              </button>
            ) : (
              <span className="ok-text">✓ All answers reviewed</span>
            )}
            <ExportMenu assessmentId={aid} />
          </div>
        </div>
      </header>

      <nav className="tabs" aria-label="Assessment sections">
        {tabs.map((t) => (
          <button key={t.key} type="button" className={`tab${tab === t.key ? " active" : ""}`} aria-current={tab === t.key ? "page" : undefined}
                  onClick={() => setParam("tab", t.key === "assessment" ? null : t.key)}>
            {t.label}
          </button>
        ))}
      </nav>

      {error && <div className="notice error" role="alert">{error}</div>}

      {tab === "assessment" && (
        <div className="stack">
          {a.notices.length > 0 && (
            <ul className="notice plain-list">{a.notices.map((n, i) => <li key={i}>{n}</li>)}</ul>
          )}
          <ReportingRow r={byKey.get("reporting:reporting")} onOpen={(rid) => setParam("rec", String(rid))} />
          <AssessmentMatrix q={q} byKey={byKey} onOpen={(rid) => setParam("rec", String(rid))} />
          <section className="panel" aria-labelledby="gaps-h">
            <div className="panel-row">
              <h2 id="gaps-h" className="panel-title">Data gaps: {gaps.length}</h2>
              {gaps.length > 0 && (
                <button type="button" className="btn" aria-expanded={showGaps} onClick={() => setShowGaps((v) => !v)}>
                  {showGaps ? "Hide data gaps" : "Review data gaps"}
                </button>
              )}
            </div>
            {showGaps && <DataGapList gaps={gaps} onOpen={(rid) => setParam("rec", String(rid))} />}
          </section>
          <div className="page-foot">
            {!a.superseded_by_id && !summary.newer_version_id && (
              <button type="button" className="link-btn muted-link" onClick={() => setConfirmRerun(true)}>Run new analysis</button>
            )}
          </div>
        </div>
      )}

      {tab === "sources" && (
        <section className="panel">
          <h2 className="panel-title">Sources</h2>
          {sources === null ? <p className="muted">Loading…</p> : <SourceList sources={sources} />}
        </section>
      )}

      {tab === "audit" && (
        <section className="panel">
          <h2 className="panel-title">Audit</h2>
          {audit === null ? <p className="muted">Loading…</p> : <AuditList events={audit} recs={recs} />}
        </section>
      )}

      {selected !== null && (
        <ReviewDrawer
          recId={selected}
          onClose={() => setParam("rec", null)}
          onChanged={reload}
          onNext={next ? () => setParam("rec", String(next.id)) : null}
        />
      )}

      {confirmRerun && (
        <ConfirmDialog title="Run a new analysis?" confirmLabel="Run new analysis" busy={rerunBusy}
                       onConfirm={rerun} onCancel={() => setConfirmRerun(false)}>
          <p>{a.company.name} ({a.year}) will be analysed again from its sources.</p>
          <p>
            This saved assessment stays available until the new analysis completes, and is then replaced.
            Analyst decisions made here are kept in its history.
          </p>
        </ConfirmDialog>
      )}
    </div>
  );
}
