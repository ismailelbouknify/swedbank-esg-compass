import { api, type Source } from "../api";

export default function SourceList({ sources }: { sources: Source[] }) {
  if (sources.length === 0) return <p className="muted">No sources were found for this assessment.</p>;
  const sorted = [...sources].sort((a, b) => Number(b.status === "PROCESSED") - Number(a.status === "PROCESSED") || a.priority - b.priority);
  return (
    <ul className="source-list">
      {sorted.map((s) => {
        const href = s.has_file ? api.sourceFileUrl(s.id) : s.url;
        const title = s.title || s.original_filename || s.url || "Source";
        return (
          <li key={s.id}>
            <div className="source-title">
              {href ? <a href={href} target="_blank" rel="noreferrer">{title}</a> : title}
            </div>
            <div className="source-meta">
              {s.source_type_label}
              {!s.is_public && " · not public"}
            </div>
            <div className={`source-status ${s.status === "PROCESSED" ? "ok" : "muted"}`}>
              {s.status === "PROCESSED" && <span aria-hidden="true">✓ </span>}
              {s.status_label}
            </div>
          </li>
        );
      })}
    </ul>
  );
}
