import { useEffect, useState } from "react";
import { api, type Questionnaire } from "../api";
import { useDocumentTitle } from "../brand";

// Read-only reference. Wording comes from the project's questionnaire configuration
// (backend/app/config/esg_rules.json), which is also what the assessment rules use.
const ORDER = ["reporting", "planning", "execution", "performance"] as const;
const PILLAR: Record<string, string> = { E: "Environmental", S: "Social", G: "Governance" };

export default function QuestionnaireExample() {
  useDocumentTitle("Questionnaire Example");
  const [q, setQ] = useState<Questionnaire | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.questionnaire().then(setQ).catch((e: Error) => setError(e.message));
  }, []);

  return (
    <div className="page narrow">
      <h1>Questionnaire Example</h1>
      <p className="lead">Reference criteria used by Swedbank ESG Compass</p>
      <p className="muted">
        Reporting is answered once per company. Planning, Execution and Performance Improvement are answered for
        each material factor.
      </p>

      {error && <div className="notice error" role="alert">{error}</div>}
      {!q && !error && <p className="muted">Loading…</p>}

      {q && (
        <>
          <section className="q-section">
            <h2>Material factors</h2>
            <ul className="factor-list">
              {q.factors.map((f) => (
                <li key={f.key}>
                  {f.label} <span className="muted">· {PILLAR[f.pillar] ?? f.pillar}</span>
                </li>
              ))}
            </ul>
          </section>

          {ORDER.filter((k) => q.dimensions[k]).map((k) => {
            const d = q.dimensions[k];
            return (
              <section key={k} className="q-section">
                <h2>{d.label}</h2>
                <p className="q-question">
                  {d.question}
                  <span className="muted"> {d.scope === "company" ? "(company level)" : "(per material factor)"}</span>
                </p>
                <ol className="levels">
                  {Object.entries(d.levels).map(([n, text]) => (
                    <li key={n}>
                      <span className="level-num">{n}</span>
                      <span>{text}</span>
                    </li>
                  ))}
                </ol>
              </section>
            );
          })}
        </>
      )}
    </div>
  );
}
