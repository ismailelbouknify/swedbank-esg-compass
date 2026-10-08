// Typed client for the FastAPI backend (see backend/app/schemas/api.py).
// Only analyst-facing fields are typed here; technical metadata stays in the backend.

export type EvidenceStatus = "FOUND" | "NO_EVIDENCE_FOUND" | "CONFLICTING_EVIDENCE" | "REQUIRES_REVIEW";
export type ReviewStatus = "PENDING" | "APPROVED" | "OVERRIDDEN";
export type ReviewState = "review" | "approved" | "changed" | "no_evidence";
export type AssessmentStatus = "CREATED" | "RUNNING" | "COMPLETED" | "FAILED";

export interface Company {
  id: number;
  name: string;
  website: string | null;
  domain: string | null;
  country: string | null;
  industry: string | null;
  organisation_number: string | null;
}

export interface Assessment {
  id: number;
  year: number;
  status: AssessmentStatus;
  progress: number;
  error: string | null;
  notices: string[];
  enable_web_search: boolean;
  assessment_version: number;
  supersedes_id: number | null;
  superseded_by_id: number | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  duration_seconds: number | null;
  company: Company;
}

export interface AssessmentListItem extends Assessment {
  reviewed: number;
  total: number;
  data_gaps: number;
}

export interface Lookup {
  found: boolean;
  ambiguous: boolean;
  completed: AssessmentListItem | null;
  in_progress: Assessment | null;
  candidates: AssessmentListItem[];
}

export interface AssessmentSummary {
  assessment: Assessment;
  sources_reviewed: number;
  sources_total: number;
  sustainability_report_found: boolean;
  reviewed: number;
  total: number;
  data_gaps: number;
  newer_version_id: number | null;
}

export interface Stage {
  index: number;
  label: string;
  state: "pending" | "active" | "done" | "failed";
}

export interface StatusResponse {
  id: number;
  company_name: string;
  year: number;
  status: AssessmentStatus;
  stage: number;
  stage_label: string | null;
  progress: number;
  error: string | null;
  notices: string[];
  stages: Stage[];
  started_at: string | null;
  completed_at: string | null;
  duration_seconds: number | null;
  server_time: string;
}

export interface Condition {
  label: string;
  status: "met" | "not_found" | "unknown";
  evidence_ids: number[];
  detail: string | null;
}

export interface Recommendation {
  id: number;
  assessment_id: number;
  factor: string;
  factor_label: string;
  dimension: string;
  dimension_label: string;
  recommended_level: number;
  max_level: number;
  level_description: string;
  evidence_status: EvidenceStatus;
  reason: string;
  conditions: Record<string, Condition>;
  next_level_missing: string[];
  notes: string[];
  evidence_ids: number[];
  confidence_label: "High" | "Medium" | "Low";
  review_status: ReviewStatus;
  review_state: ReviewState;
  final_level: number | null;
  analyst_comment: string | null;
  updated_at: string;
}

export interface Evidence {
  id: number;
  factor: string;
  dimension: string;
  claim_text: string;
  source_id: number;
  source_title: string | null;
  source_url: string | null;
  source_type: string | null;
  source_has_file: boolean;
  page_number: number | null;
  evidence_text: string;
  verified: boolean;
}

export interface Source {
  id: number;
  source_type: string;
  source_type_label: string | null;
  priority: number;
  title: string | null;
  url: string | null;
  original_filename: string | null;
  is_public: boolean;
  is_official: boolean;
  status: string;
  status_label: string | null;
  page_count: number | null;
  report_year: number | null;
  has_file: boolean;
}

export interface Decision {
  id: number;
  action: "APPROVE" | "OVERRIDE";
  previous_level: number;
  selected_level: number;
  comment: string | null;
  analyst: string | null;
  created_at: string;
}

export interface RecommendationDetail extends Recommendation {
  evidence: Evidence[];
  decisions: Decision[];
  level_definitions: Record<string, string>;
  sources_reviewed: Source[];
}

export interface AuditEvent {
  id: number;
  event_type: string;
  entity_type: string | null;
  entity_id: number | null;
  actor: string;
  details: Record<string, unknown>;
  created_at: string;
}

export interface Questionnaire {
  factors: { key: string; label: string; short_label: string; pillar: string }[];
  dimensions: Record<string, { label: string; scope: string; max_level: number; question: string; levels: Record<string, string> }>;
  lookback_years: number;
}

export class ApiError extends Error {
  constructor(message: string, public status: number, public detail: unknown) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch {
    throw new ApiError("The service is not reachable. Please try again in a moment.", 0, null);
  }
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    let detail: unknown = null;
    try {
      const body = await res.json();
      detail = body.detail;
      if (typeof body.detail === "string") msg = body.detail;
      else if (Array.isArray(body.detail)) msg = body.detail.map((d: { msg: string }) => d.msg).join("; ");
      else if (body.detail && typeof body.detail.message === "string") msg = body.detail.message;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(msg, res.status, detail);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

const post = (body?: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: body === undefined ? undefined : JSON.stringify(body),
});

export interface CreateAssessment {
  company_name: string;
  year: number;
  website?: string;
  country?: string;
  industry?: string;
  organisation_number?: string;
  enable_web_search: boolean;
  confirm_new?: boolean;
}

export const api = {
  questionnaire: () => request<Questionnaire>("/api/config/questionnaire"),
  lookup: (company: string, year: number, organisationNumber?: string, country?: string) => {
    const p = new URLSearchParams({ company, year: String(year) });
    if (organisationNumber) p.set("organisation_number", organisationNumber);
    if (country) p.set("country", country);
    return request<Lookup>(`/api/assessments/lookup?${p}`);
  },
  listAssessments: (q?: string) =>
    request<AssessmentListItem[]>(`/api/assessments${q ? `?q=${encodeURIComponent(q)}` : ""}`),
  createAssessment: (body: CreateAssessment) => request<Assessment>("/api/assessments", post(body)),
  uploadDocument: (id: number, file: File, isPublic: boolean) => {
    const form = new FormData();
    form.append("file", file);
    form.append("is_public", String(isPublic));
    return request<Source>(`/api/assessments/${id}/documents`, { method: "POST", body: form });
  },
  run: (id: number) => request<StatusResponse>(`/api/assessments/${id}/run`, post()),
  rerun: (id: number) => request<Assessment>(`/api/assessments/${id}/rerun`, post()),
  deleteAssessment: (id: number) => request<void>(`/api/assessments/${id}`, { method: "DELETE" }),
  status: (id: number) => request<StatusResponse>(`/api/assessments/${id}/status`),
  summary: (id: number) => request<AssessmentSummary>(`/api/assessments/${id}`),
  recommendations: (id: number) => request<Recommendation[]>(`/api/assessments/${id}/recommendations`),
  sources: (id: number) => request<Source[]>(`/api/assessments/${id}/sources`),
  audit: (id: number) => request<AuditEvent[]>(`/api/assessments/${id}/audit`),
  recommendation: (id: number) => request<RecommendationDetail>(`/api/recommendations/${id}`),
  review: (id: number, body: { action: "APPROVE" | "OVERRIDE"; selected_level?: number; comment?: string; analyst?: string }) =>
    request<Recommendation>(`/api/recommendations/${id}/review`, post(body)),
  exportUrl: (id: number, format: "json" | "csv" | "xlsx") => `/api/assessments/${id}/export?format=${format}`,
  sourceFileUrl: (sourceId: number, page?: number | null) => `/api/sources/${sourceId}/file${page ? `#page=${page}` : ""}`,
};

/** Where "Open source" should go for a piece of evidence. */
export function evidenceLink(e: Evidence): string | null {
  if (e.source_has_file) return api.sourceFileUrl(e.source_id, e.page_number);
  return e.source_url;
}
