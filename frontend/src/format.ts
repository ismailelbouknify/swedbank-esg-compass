const pad = (n: number) => String(n).padStart(2, "0");

/** 83 -> "01:23"; 3725 -> "01:02:05" (hours only once an analysis passes one hour). */
export function formatDuration(totalSeconds: number | null | undefined): string {
  const s = Math.max(0, Math.floor(totalSeconds ?? 0));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return h > 0 ? `${pad(h)}:${pad(m)}:${pad(s % 60)}` : `${pad(m)}:${pad(s % 60)}`;
}

/** "27 September 2026" */
export function formatDate(iso: string | null | undefined, style: "long" | "short" = "long"): string {
  if (!iso) return "—";
  return new Date(iso).toLocaleDateString("en-GB", { day: "numeric", month: style === "long" ? "long" : "short", year: "numeric" });
}

export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  return new Date(iso).toLocaleString("en-GB", { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" });
}

export const STATUS_LABEL: Record<string, string> = {
  CREATED: "Not started",
  RUNNING: "Analysing",
  COMPLETED: "Completed",
  FAILED: "Failed",
};

/** Shown instead of a score when no level is recommended (mirrors NO_LEVEL_LABEL in the backend). */
export const NO_LEVEL_LABEL: Record<string, string> = {
  NO_EVIDENCE_FOUND: "No supporting evidence found",
  EVIDENCE_NOT_VERIFIED: "Evidence could not be verified",
  INSUFFICIENT_SOURCE_COVERAGE: "Insufficient source coverage",
  REQUIRES_REVIEW: "Not analysed",
};
const GAP_STATUSES = ["NO_EVIDENCE_FOUND", "EVIDENCE_NOT_VERIFIED", "INSUFFICIENT_SOURCE_COVERAGE"];

export const isDataGap = (r: { evidence_status: string }) => GAP_STATUSES.includes(r.evidence_status);

/** Why no score is shown, or null when there is a score (an AI-recommended level, or one set by the analyst). */
export function noLevelLabel(r: { recommended_level: number; final_level: number | null; evidence_status: string }): string | null {
  if (r.final_level !== null && r.final_level !== r.recommended_level) return null; // the analyst set a level
  if (r.recommended_level >= 1 && !isDataGap(r)) return null;
  return NO_LEVEL_LABEL[r.evidence_status] ?? NO_LEVEL_LABEL.NO_EVIDENCE_FOUND;
}
