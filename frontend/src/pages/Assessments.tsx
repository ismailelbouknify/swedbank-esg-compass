import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, type AssessmentListItem } from "../api";
import { useDocumentTitle } from "../brand";
import ConfirmDialog from "../components/ConfirmDialog";
import { STATUS_LABEL, formatDate } from "../format";

function openPath(a: AssessmentListItem): string {
  return a.status === "COMPLETED" ? `/assessments/${a.id}` : `/assessments/${a.id}/progress`;
}

export default function Assessments() {
  useDocumentTitle("Assessments");
  const [q, setQ] = useState("");
  const [rows, setRows] = useState<AssessmentListItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [toDelete, setToDelete] = useState<AssessmentListItem | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [deleted, setDeleted] = useState<string | null>(null);

  async function confirmDelete() {
    if (!toDelete) return;
    setDeleting(true);
    setError(null);
    try {
      await api.deleteAssessment(toDelete.id);
      setRows((r) => r?.filter((x) => x.id !== toDelete.id) ?? null);
      setDeleted(`Assessment deleted: ${toDelete.company.name} ${toDelete.year}.`);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setDeleting(false);
      setToDelete(null);
    }
  }

  useEffect(() => {
    let cancelled = false;
    const t = window.setTimeout(() => {
      api.listAssessments(q.trim() || undefined)
        .then((r) => { if (!cancelled) { setRows(r); setError(null); } })
        .catch((e: Error) => !cancelled && setError(e.message));
    }, q ? 250 : 0);
    return () => { cancelled = true; window.clearTimeout(t); };
  }, [q]);

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Assessments</h1>
          <p className="page-subtitle">Saved company assessments</p>
        </div>
        <Link className="btn primary" to="/">New assessment</Link>
      </div>
      <label className="search">
        <span className="sr-only">Search companies</span>
        <input type="search" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search companies…" />
      </label>

      {error && <div className="notice error" role="alert">{error}</div>}
      {deleted && <div className="notice" role="status">{deleted}</div>}
      {rows === null && !error && <p className="muted">Loading…</p>}
      {rows?.length === 0 && (
        q ? (
          <p className="muted">No saved assessments match your search.</p>
        ) : (
          <div className="empty-state">
            <p>No saved assessments yet.</p>
            <p className="muted">Start a new assessment to begin.</p>
            <Link className="btn primary" to="/">Start a new assessment</Link>
          </div>
        )
      )}

      {rows && rows.length > 0 && (
        <div className="table-scroll">
          <table className="list">
            <thead>
              <tr>
                <th scope="col">Company</th>
                <th scope="col">Year</th>
                <th scope="col">Status</th>
                <th scope="col">Last analysed</th>
                <th scope="col">Review status</th>
                <th scope="col"><span className="sr-only">Open</span></th>
              </tr>
            </thead>
            <tbody>
              {rows.map((a) => (
                <tr key={a.id}>
                  <td className="strong">{a.company.name}</td>
                  <td>{a.year}</td>
                  <td>
                    <span className={`status-text ${a.status.toLowerCase()}`}>{STATUS_LABEL[a.status] ?? a.status}</span>
                  </td>
                  <td>{a.status === "COMPLETED" ? formatDate(a.completed_at, "short") : "—"}</td>
                  <td>
                    {a.total > 0 ? (
                      <>
                        {a.reviewed} / {a.total} reviewed
                        {a.reviewed === a.total && <span className="ok-text" aria-hidden="true"> ✓</span>}
                      </>
                    ) : "—"}
                  </td>
                  <td className="right row-actions">
                    <Link to={openPath(a)} className="open-link">
                      Open<span className="sr-only"> {a.company.name} {a.year}</span> →
                    </Link>
                    <button type="button" className="link-btn danger" disabled={a.status === "RUNNING"}
                            title={a.status === "RUNNING" ? "Available when the analysis has finished" : undefined}
                            onClick={() => { setDeleted(null); setToDelete(a); }}>
                      Delete<span className="sr-only"> {a.company.name} {a.year}</span>
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {toDelete && (
        <ConfirmDialog title={`Delete ${toDelete.company.name} ${toDelete.year}?`} confirmLabel="Delete assessment"
                       danger busy={deleting} onConfirm={confirmDelete} onCancel={() => setToDelete(null)}>
          <p>
            This permanently removes the assessment, its evidence, analyst decisions and any earlier versions.
            Uploaded reports that no other assessment uses are removed too.
          </p>
          <p><strong>This cannot be undone.</strong> Export the assessment first if you need a copy.</p>
        </ConfirmDialog>
      )}
    </div>
  );
}
