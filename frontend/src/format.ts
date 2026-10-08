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
