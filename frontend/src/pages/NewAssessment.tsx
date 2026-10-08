import { useState, type FormEvent, type ReactNode } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, ApiError, type AssessmentListItem, type Lookup } from "../api";
import { useDocumentTitle } from "../brand";
import { formatDate } from "../format";

const MAX_MB = 50;

function ExistingCard({ a, children }: { a: AssessmentListItem; children?: ReactNode }) {
  return (
    <div className="existing">
      <div className="existing-name">{a.company.name}</div>
      <dl className="facts">
        <div><dt>Assessment year</dt><dd>{a.year}</dd></div>
        <div><dt>Last analysed</dt><dd>{formatDate(a.completed_at)}</dd></div>
        <div><dt>Status</dt><dd>Completed</dd></div>
        <div><dt>Review</dt><dd>{a.reviewed} / {a.total} reviewed</dd></div>
        {a.company.organisation_number && <div><dt>Organisation number</dt><dd>{a.company.organisation_number}</dd></div>}
        {a.company.country && <div><dt>Country</dt><dd>{a.company.country}</dd></div>}
      </dl>
      {children}
    </div>
  );
}

export default function NewAssessment() {
  useDocumentTitle(null);
  const navigate = useNavigate();
  const [company, setCompany] = useState("");
  const [year, setYear] = useState(new Date().getFullYear());
  const [website, setWebsite] = useState("");
  const [country, setCountry] = useState("");
  const [industry, setIndustry] = useState("");
  const [orgNo, setOrgNo] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [isPublic, setIsPublic] = useState(true);
  const [webSearch, setWebSearch] = useState(true);
  const [advanced, setAdvanced] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [found, setFound] = useState<Lookup | null>(null);
  const [confirmRerun, setConfirmRerun] = useState(false);

  // any change to the identity invalidates the previous lookup
  const edit = <T,>(set: (v: T) => void) => (v: T) => { set(v); setFound(null); setConfirmRerun(false); };

  function onFile(f: File | null) {
    setError(null);
    if (f && !f.name.toLowerCase().endsWith(".pdf")) return setError("Only PDF files can be uploaded.");
    if (f && f.size > MAX_MB * 1024 * 1024) return setError(`The file exceeds the ${MAX_MB} MB limit.`);
    setFile(f);
  }

  async function startNew(confirmNew: boolean) {
    const a = await api.createAssessment({
      company_name: company.trim(),
      year,
      website: website || undefined,
      country: country || undefined,
      industry: industry || undefined,
      organisation_number: orgNo || undefined,
      enable_web_search: webSearch,
      confirm_new: confirmNew,
    });
    if (file) await api.uploadDocument(a.id, file, isPublic);
    await api.run(a.id);
    navigate(`/assessments/${a.id}/progress`);
  }

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      // saved assessments are reused: nothing is searched or analysed again unless asked for
      const res = await api.lookup(company.trim(), year, orgNo || undefined, country || undefined);
      if (res.found || res.in_progress) {
        setFound(res);
        setBusy(false);
        return;
      }
      await startNew(false);
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        const detail = err.detail as { assessment_id?: number } | null;
        if (detail?.assessment_id) return navigate(`/assessments/${detail.assessment_id}`);
      }
      setError((err as Error).message);
      setBusy(false);
    }
  }

  async function runNew() {
    if (!found) return;
    setBusy(true);
    setError(null);
    try {
      if (found.completed && !file) {
        // same inputs: re-analyse the saved assessment as a new version (keeps its uploaded report)
        const a = await api.rerun(found.completed.id);
        navigate(`/assessments/${a.id}/progress`);
      } else {
        await startNew(true);
      }
    } catch (err) {
      setError((err as Error).message);
      setBusy(false);
    }
  }

  return (
    <div className="page narrow">
      <h1>New Assessment</h1>
      <p className="lead">
        Create a new sustainability assessment using company information and public or uploaded sustainability evidence.
      </p>

      <form className="form-card" onSubmit={submit}>
        <div className="grid-2">
          <label className="field">
            Company name *
            <input required value={company} onChange={(e) => edit(setCompany)(e.target.value)} placeholder="e.g. Nordvik Components AB" autoComplete="organization" />
          </label>
          <label className="field">
            Assessment year *
            <input required type="number" min={2000} max={2100} value={year} onChange={(e) => edit(setYear)(Number(e.target.value))} />
          </label>
        </div>

        <label className="field">
          Sustainability report
          <span className="hint">Upload PDF (optional). Without a report, public sources are searched.</span>
          <input type="file" accept="application/pdf,.pdf" onChange={(e) => onFile(e.target.files?.[0] ?? null)} />
        </label>
        {file && (
          <label className="check">
            <input type="checkbox" checked={isPublic} onChange={(e) => setIsPublic(e.target.checked)} />
            This report is publicly available
          </label>
        )}

        <button type="button" className="disclosure" aria-expanded={advanced} onClick={() => setAdvanced((v) => !v)}>
          Advanced information <span aria-hidden="true">{advanced ? "▴" : "▾"}</span>
        </button>
        {advanced && (
          <div className="advanced">
            <div className="grid-2">
              <label className="field">
                Company website
                <input value={website} onChange={(e) => setWebsite(e.target.value)} placeholder="https://www.example.com" />
              </label>
              <label className="field">
                Organisation number
                <input value={orgNo} onChange={(e) => edit(setOrgNo)(e.target.value)} placeholder="e.g. 556012-5790" />
              </label>
              <label className="field">
                Country
                <input value={country} onChange={(e) => edit(setCountry)(e.target.value)} />
              </label>
              <label className="field">
                Industry
                <input value={industry} onChange={(e) => setIndustry(e.target.value)} />
              </label>
            </div>
            <label className="check">
              <input type="checkbox" checked={webSearch} onChange={(e) => setWebSearch(e.target.checked)} />
              Search public sources when no report is uploaded
            </label>
          </div>
        )}

        {error && <div className="notice error" role="alert">{error}</div>}

        {!found && (
          <div className="form-actions">
            <button className="btn primary lg" type="submit" disabled={busy || !company.trim()}>
              {busy ? "Starting…" : "Start Analysis"}
            </button>
          </div>
        )}
      </form>

      {found && (
        <section className="result-card" aria-live="polite">
          {found.ambiguous ? (
            <>
              <h2>More than one saved company matches</h2>
              <p className="muted">
                Open the right one, or add the organisation number under Advanced information to tell them apart.
              </p>
              {found.candidates.map((c) => (
                <ExistingCard key={c.id} a={c}>
                  <Link className="btn primary" to={`/assessments/${c.id}`}>Open assessment</Link>
                </ExistingCard>
              ))}
            </>
          ) : found.completed ? (
            <>
              <h2>Existing assessment found</h2>
              <ExistingCard a={found.completed}>
                {found.in_progress && (
                  <p className="notice">
                    A new analysis is already running. <Link to={`/assessments/${found.in_progress.id}/progress`}>View progress</Link>
                  </p>
                )}
                {!confirmRerun ? (
                  <div className="form-actions">
                    <Link className="btn primary" to={`/assessments/${found.completed.id}`}>Open assessment</Link>
                    {!found.in_progress && (
                      <button type="button" className="btn" onClick={() => setConfirmRerun(true)}>Run new analysis</button>
                    )}
                  </div>
                ) : (
                  <div className="confirm-inline">
                    <p>
                      <strong>Run a new analysis?</strong> Public sources{file ? " and your new report" : ""} will be read again.
                      The saved assessment stays available and is replaced only when the new analysis completes.
                      Earlier analyst decisions are kept in its history.
                    </p>
                    <div className="form-actions">
                      <button type="button" className="btn primary" disabled={busy} onClick={runNew}>
                        {busy ? "Starting…" : "Yes, run new analysis"}
                      </button>
                      <button type="button" className="btn" disabled={busy} onClick={() => setConfirmRerun(false)}>Cancel</button>
                    </div>
                  </div>
                )}
              </ExistingCard>
            </>
          ) : found.in_progress ? (
            <>
              <h2>Analysis in progress</h2>
              <p>{found.in_progress.company.name} ({found.in_progress.year}) is already being analysed.</p>
              <Link className="btn primary" to={`/assessments/${found.in_progress.id}/progress`}>View progress</Link>
            </>
          ) : null}
        </section>
      )}
    </div>
  );
}
