/**
 * Typed API client — all calls route via Next.js /api/* rewrite → FastAPI.
 */

const BASE = "/api";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`GET ${path} → ${res.status}`);
  return res.json() as Promise<T>;
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    body ? JSON.stringify(body) : undefined,
    cache:   "no-store",
  });
  if (!res.ok) throw new Error(`POST ${path} → ${res.status}`);
  return res.json() as Promise<T>;
}

async function del<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`DELETE ${path} → ${res.status}`);
  return res.json() as Promise<T>;
}

// ── Exports 

import type {
  HealthResponse, CallsResponse, ActiveCall,
  TenantSummary, SIPResourcesResponse,
  EvalResults, MetricsSummary,
} from "./types";

export const api = {
  health:       ()         => get<HealthResponse>("/health"),
  calls:        ()         => get<CallsResponse>("/calls"),
  call:         (room: string) => get<ActiveCall>(`/calls/${encodeURIComponent(room)}`),
  hangupCall:   (room: string) => del<{ status: string }>(`/calls/${encodeURIComponent(room)}`),

  tenants:      ()         => get<TenantSummary[]>("/tenants"),
  tenantConfig: (id: string)  => get<Record<string, unknown>>(`/tenants/${id}/config`),
  sipResources: ()         => get<SIPResourcesResponse>("/routing/sip-resources"),
  ddiRouting:   ()         => get<{ routes: Record<string, string>; default_tenant: string }>("/routing/ddi"),

  evalsLatest:  ()         => get<EvalResults>("/evals/latest"),
  evalsRun:     (tenantId = "surgery_greenfield") =>
                             post<{ job_id: string; status: string }>(`/evals/run?tenant_id=${tenantId}`),
  evalsStatus:  (jobId: string) => get<{ status: string; stage: string }>(`/evals/status/${jobId}`),

  metrics:      ()         => get<MetricsSummary>("/metrics/summary"),

  /** SSE endpoint URL (pass directly to EventSource) */
  eventsUrl:    ()         => `${BASE}/events`,
  roomEventsUrl:(room: string) => `${BASE}/events/${encodeURIComponent(room)}`,
};