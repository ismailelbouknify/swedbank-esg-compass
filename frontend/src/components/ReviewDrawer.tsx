import { useEffect, useRef, useState } from "react";
import { api, evidenceLink, type Evidence, type RecommendationDetail } from "../api";
import { formatDateTime } from "../format";
import ReviewStateBadge from "./ReviewStateBadge";

const MARK = { met: "✓", not_found: "✗", unknown: "?" } as const;
const MARK_LABEL = { met: "Found", not_found: "Not found", unknown: "Unclear" } as const;
const ANALYST_KEY = "esg.analyst";

function readAnalyst(): string {
  try {
    return localStorage.getItem(ANALYST_KEY) ?? "";
  } catch {
    return "";
  }
}

function EvidenceItem({ e }: { e: Evidence }) {
  const link = evidenceLink(e);
  return (
    <figure className="evidence">
      <figcaption className="evidence-source">
        {e.source_title || e.source_url || "Source"}
        {e.page_number ? <span className="muted"> · page {e.page_number}</span> : null}
      </figcaption>
      <blockquote>“{e.evidence_text}”</blockquote>
      {!e.verified && (
        <p className="evidence-warn">
          <span aria-hidden="true">! </span>This quote could not be fully matched to the source. Please check it.
        </p>
      )}
      {link && (
        <a className="btn sm" href={link} target="_blank" rel="noreferrer">
          Open source <span className="sr-only">(opens in a new tab)</span>
        </a>
      )}
    </figure>
  );
}

export default function ReviewDrawer({ recId, onClose, onChanged, onNext }: {
  recId: number;
  onClose: () => void;
  onChanged: () => void;
  onNext: (() => void) | null;
}) {
  const [rec, setRec] = useState<RecommendationDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<"view" | "change">("view");
  const [level, setLevel] = useState(1);
  const [comment, setComment] = useState("");
  const [analyst, setAnalyst] = useState(readAnalyst);
  const [busy, setBusy] = useState(false);
  const [details, setDetails] = useState(false);
  const closeRef = useRef<HTMLButtonElement>(null);

  const load = () =>
    api.recommendation(recId)
      .then((r) => { setRec(r); setLevel(r.final_level ?? r.recommended_level); })
      .catch((e: Error) => setError(e.message));

  useEffect(() => {
    setRec(null);
    setError(null);
    setMode("view");
    setComment("");
    setDetails(false);
    load();
    closeRef.current?.focus();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [recId]);

  useEffect(() => {
    const onKey = (ev: KeyboardEvent) => ev.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  async function submit(action: "APPROVE" | "OVERRIDE") {
    setBusy(true);
    setError(null);
    try {
      try {
        if (analyst.trim()) localStorage.setItem(ANALYST_KEY, analyst.trim());
      } catch {
        /* storage unavailable: the name is simply not remembered */
      }
      await api.review(recId, {
        action,
        selected_level: action === "OVERRIDE" ? level : undefined,
        comment: comment.trim() || undefined,
        analyst: analyst.trim() || undefined,
      });
      setMode("view");
      setComment("");
      await load();
      onChanged();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const noEvidence = rec?.evidence_status === "NO_EVIDENCE_FOUND";
  const [primary, ...others] = rec?.evidence ?? [];
  const changed = rec && rec.final_level !== null && rec.final_level !== rec.recommended_level;
  const title = rec ? (rec.factor === "reporting" ? "Reporting" : rec.factor_label) : "Loading…";

  return (
    <>
      <div className="drawer-backdrop" onClick={onClose} />
      <aside className="drawer" role="dialog" aria-modal="true" aria-labelledby="drawer-title">
        <div className="drawer-head">
          <div>
            <h2 id="drawer-title">{title}</h2>
            {rec && rec.factor !== "reporting" && <div className="drawer-sub">{rec.dimension_label}</div>}
          </div>
          <button ref={closeRef} type="button" className="icon-btn" onClick={onClose} aria-label="Close review panel">✕</button>
        </div>

        {error && <div className="notice error" role="alert">{error}</div>}

        {rec && (
          <div className="drawer-body">
            <div className="drawer-score">
              {noEvidence && !changed ? (
                <div className="score-line"><span className="score-big">—</span> <span className="muted">No evidence found in reviewed sources</span></div>
              ) : (
                <div className="score-line">
                  <span className="muted">{changed ? "Recommended" : "Recommended:"}</span>{" "}
                  <span className={changed ? "score-struck" : "score-big"}>{rec.recommended_level} / {rec.max_level}</span>
                  {changed && <> <span className="muted">→ Analyst:</span> <span className="score-big">{rec.final_level} / {rec.max_level}</span></>}
                </div>
              )}
              <ReviewStateBadge state={rec.review_state} />
            </div>

            <section>
              <h3 className="label">Why?</h3>
              <p>{rec.reason}</p>
              {rec.analyst_comment && (
                <p className="analyst-comment"><strong>Analyst comment:</strong> {rec.analyst_comment}</p>
              )}
            </section>

            {primary && (
              <section>
                <h3 className="label">Evidence</h3>
                <EvidenceItem e={primary} />
              </section>
            )}

            {noEvidence && (
              <section>
                <h3 className="label">Relevant sources reviewed</h3>
                {rec.sources_reviewed.length ? (
                  <ul className="plain-list">
                    {rec.sources_reviewed.map((s) => <li key={s.id}>{s.title || s.original_filename || s.url}</li>)}
                  </ul>
                ) : (
                  <p className="muted">No sources could be read for this assessment.</p>
                )}
                <p className="muted small">Not found does not mean the practice does not exist.</p>
              </section>
            )}

            {mode === "view" ? (
              <div className="drawer-actions">
                <button type="button" className="btn primary" disabled={busy} onClick={() => submit("APPROVE")}>
                  {rec.review_state === "approved" ? "Approved ✓" : "Approve"}
                </button>
                <button type="button" className="btn" disabled={busy} onClick={() => setMode("change")}>Change</button>
                {onNext && <button type="button" className="btn ghost push" onClick={onNext}>Next item →</button>}
              </div>
            ) : (
              <form className="change-form" onSubmit={(e) => { e.preventDefault(); submit("OVERRIDE"); }}>
                <fieldset>
                  <legend className="label">Select level</legend>
                  {Object.entries(rec.level_definitions).map(([n, text]) => (
                    <label key={n} className={`level-option${Number(n) === level ? " selected" : ""}`}>
                      <input type="radio" name="level" value={n} checked={Number(n) === level} onChange={() => setLevel(Number(n))} />
                      <span className="level-num">{n}</span>
                      <span>{text}{Number(n) === rec.recommended_level && <span className="muted"> (recommended)</span>}</span>
                    </label>
                  ))}
                </fieldset>
                <label className="field">
                  Reason for the change *
                  <textarea rows={3} value={comment} required onChange={(e) => setComment(e.target.value)}
                            placeholder="e.g. The target does not meet the criterion because…" />
                </label>
                <label className="field">
                  Your name or initials
                  <input value={analyst} onChange={(e) => setAnalyst(e.target.value)} />
                </label>
                <div className="drawer-actions">
                  <button type="submit" className="btn primary" disabled={busy || !comment.trim()}>Save change</button>
                  <button type="button" className="btn" disabled={busy} onClick={() => setMode("view")}>Cancel</button>
                </div>
              </form>
            )}

            <button type="button" className="disclosure" aria-expanded={details} onClick={() => setDetails((d) => !d)}>
              More details <span aria-hidden="true">{details ? "▴" : "▾"}</span>
            </button>

            {details && (
              <div className="details">
                {Object.keys(rec.conditions).length > 0 && (
                  <section>
                    <h3 className="label">Detected</h3>
                    <ul className="checklist">
                      {Object.entries(rec.conditions).map(([k, c]) => (
                        <li key={k}>
                          <span className={`mark ${c.status}`} aria-hidden="true">{MARK[c.status]}</span>
                          <span className="sr-only">{MARK_LABEL[c.status]}: </span>
                          <span>{c.label}{c.detail && <span className="muted"> — {c.detail}</span>}</span>
                        </li>
                      ))}
                    </ul>
                  </section>
                )}
                {rec.next_level_missing.length > 0 && (
                  <section>
                    <h3 className="label">Missing for next level</h3>
                    <ul className="plain-list bullets">{rec.next_level_missing.map((m, i) => <li key={i}>{m}</li>)}</ul>
                  </section>
                )}
                {rec.notes.length > 0 && (
                  <section>
                    <h3 className="label">Notes</h3>
                    <ul className="plain-list bullets">{rec.notes.map((n, i) => <li key={i}>{n}</li>)}</ul>
                  </section>
                )}
                {others.length > 0 && (
                  <section>
                    <h3 className="label">Other supporting evidence</h3>
                    {others.map((e) => <EvidenceItem key={e.id} e={e} />)}
                  </section>
                )}
                {!noEvidence && (
                  <section>
                    <h3 className="label">Confidence</h3>
                    <p>{rec.confidence_label}</p>
                  </section>
                )}
                <section>
                  <h3 className="label">Questionnaire levels</h3>
                  <ol className="levels">
                    {Object.entries(rec.level_definitions).map(([n, text]) => (
                      <li key={n} className={Number(n) === (rec.final_level ?? rec.recommended_level) ? "current" : ""}>
                        <span className="level-num">{n}</span>
                        <span>{text}</span>
                      </li>
                    ))}
                  </ol>
                </section>
                {rec.decisions.length > 0 && (
                  <section>
                    <h3 className="label">Review history</h3>
                    <ul className="plain-list small">
                      {rec.decisions.map((d) => (
                        <li key={d.id}>
                          {formatDateTime(d.created_at)} · {d.analyst || "Analyst"} ·{" "}
                          {d.action === "APPROVE" ? `approved level ${d.selected_level}` : `changed ${d.previous_level} → ${d.selected_level}`}
                          {d.comment && <> · “{d.comment}”</>}
                        </li>
                      ))}
                    </ul>
                  </section>
                )}
              </div>
            )}
          </div>
        )}
      </aside>
    </>
  );
}
