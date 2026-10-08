import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, type StatusResponse } from "../api";
import { useDocumentTitle } from "../brand";
import { formatDuration } from "../format";

const POLL_MS = 1500;

/**
 * Elapsed time since the backend's started_at. The clock offset between browser and server is
 * taken from each status response, so the timer ticks smoothly every second locally but stays
 * anchored to backend timestamps (which are also what the stored duration is computed from).
 */
function useElapsed(st: StatusResponse | null, offsetMs: number): number {
  const [now, setNow] = useState(Date.now());
  const running = st?.status === "RUNNING";
  useEffect(() => {
    if (!running) return;
    const t = window.setInterval(() => setNow(Date.now()), 250);
    return () => window.clearInterval(t);
  }, [running]);
  if (!st?.started_at) return 0;
  if (!running && st.duration_seconds !== null) return st.duration_seconds;
  return (now + offsetMs - new Date(st.started_at).getTime()) / 1000;
}

export default function AnalysisProgress() {
  const { id } = useParams();
  const aid = Number(id);
  const navigate = useNavigate();
  const [st, setSt] = useState<StatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pollKey, setPollKey] = useState(0);
  const offset = useRef(0);
  const elapsed = useElapsed(st, offset.current);
  useDocumentTitle(st ? (st.status === "COMPLETED" ? `${st.company_name}` : `Analysing ${st.company_name}`) : null);

  useEffect(() => {
    let timer: number | undefined;
    let cancelled = false;
    const poll = async () => {
      try {
        const s = await api.status(aid);
        if (cancelled) return;
        offset.current = new Date(s.server_time).getTime() - Date.now();
        setSt(s);
        if (s.status === "COMPLETED") timer = window.setTimeout(() => navigate(`/assessments/${aid}`), 2200);
        else if (s.status === "RUNNING") timer = window.setTimeout(poll, POLL_MS);
      } catch (e) {
        if (!cancelled) setError((e as Error).message);
      }
    };
    poll();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [aid, navigate, pollKey]);

  async function start() {
    setError(null);
    try {
      setSt(await api.run(aid));
      setPollKey((k) => k + 1);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  if (error) return <div className="page narrow"><div className="notice error" role="alert">{error}</div></div>;
  if (!st) return <div className="page narrow"><p className="muted">Loading…</p></div>;

  const done = st.status === "COMPLETED";
  return (
    <div className="page narrow">
      <div className="progress-card">
        <h1>{done ? "Assessment completed" : `Analysing ${st.company_name}`}</h1>
        <div className="muted">{done ? `${st.company_name} · ` : ""}Assessment year {st.year}</div>

        {st.status !== "CREATED" && (
          <div className="timer" role="timer" aria-live="off" aria-label={`Elapsed time ${formatDuration(elapsed)}`}>
            {formatDuration(elapsed)}
          </div>
        )}

        {st.status === "RUNNING" && (
          <>
            <p className="stage-now" aria-live="polite">{st.stage_label}…</p>
            <div className="progress" role="progressbar" aria-label="Analysis progress"
                 aria-valuenow={Math.round(st.progress)} aria-valuemin={0} aria-valuemax={100}>
              <div style={{ width: `${Math.max(3, st.progress)}%` }} />
            </div>
          </>
        )}

        <ol className="stages">
          {st.stages.map((s) => (
            <li key={s.index} className={s.state}>
              <span className="stage-dot" aria-hidden="true">{s.state === "done" ? "✓" : s.state === "failed" ? "!" : ""}</span>
              <span>{s.label}</span>
              <span className="sr-only">
                {s.state === "done" ? " (done)" : s.state === "active" ? " (in progress)" : s.state === "failed" ? " (stopped)" : ""}
              </span>
            </li>
          ))}
        </ol>

        {done && (
          <div className="done-row">
            <p>Analysis completed in <strong>{formatDuration(st.duration_seconds ?? elapsed)}</strong></p>
            <Link className="btn primary" to={`/assessments/${aid}`}>Open assessment</Link>
          </div>
        )}

        {st.status === "FAILED" && (
          <div className="notice error" role="alert">
            <p>{st.error ?? "The analysis could not be completed."}</p>
            <button type="button" className="btn primary" onClick={start}>Try again</button>
          </div>
        )}

        {st.status === "CREATED" && (
          <button type="button" className="btn primary" onClick={start}>Start Analysis</button>
        )}

        {st.notices.length > 0 && (
          <ul className="notice plain-list">{st.notices.map((n, i) => <li key={i}>{n}</li>)}</ul>
        )}
      </div>
    </div>
  );
}
