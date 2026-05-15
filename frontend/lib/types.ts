// ── Call types 

export type CallState =
  | "CONNECTING" | "CONNECTED" | "STREAMING"
  | "ENDED" | "TIMED_OUT" | "ERROR";

export interface ActiveCall {
  room_name:          string;
  call_uuid:          string;
  tenant_id:          string;
  caller_identity:    string;
  caller_number:      string;
  destination_number: string;
  participant_sid:    string;
  session_id:         string | null;
  trace_id:           string | null;
  state:              CallState;
  started_at:         string;
  ended_at:           string | null;
  duration_seconds:   number;
  dtmf_digits:        string[];
  error:              string | null;
}

export interface CallsResponse {
  active_count: number;
  calls:        ActiveCall[];
}

// ── Tenant types 

export interface TenantSummary {
  tenant_id:    string;
  surgery_name: string;
  phone:        string | null;
  opening_hours: Record<string, string> | null;
}

// ── SIP resource types 

export interface SIPTrunk {
  id:      string;
  name:    string;
  numbers: string[];
}

export interface SIPDispatchRule {
  id:   string;
  name: string;
}

export interface SIPResourcesResponse {
  trunks:         SIPTrunk[];
  dispatch_rules: SIPDispatchRule[];
}

// ── Health types 

export interface HealthResponse {
  status:               string;
  version:              string;
  tenants:              string[];
  active_voice_sessions: number;
  active_calls:         number;
  sse_subscribers:      number;
  tts_ready:            boolean;
  stt_ready:            boolean;
}

// ── Eval types 

export interface RAGASMetrics {
  faithfulness:      number;
  answer_relevancy:  number;
  context_precision: number;
  context_recall:    number;
}

export interface RAGASResult {
  source:       string;
  tenant_id:    string;
  metrics:      RAGASMetrics;
  pass:         boolean;
  sample_count: number;
}

export interface DeepEvalTestCase {
  name:   string;
  passed: boolean;
}

export interface DeepEvalResult {
  source:      string;
  total_tests: number;
  passed:      number;
  failed:      number;
  pass_rate:   number;
  test_cases:  DeepEvalTestCase[];
}

export interface EvalResults {
  status:           string;
  job_id?:          string;
  tenant_id?:       string;
  ragas:            RAGASResult | null;
  deepeval:         DeepEvalResult | null;
  run_at:           string | null;
  duration_seconds: number | null;
  message?:         string;
}

// ── Metrics summary 

export interface MetricsSummary {
  calls_in_history:          number;
  calls_ended_in_history:    number;
  active_calls:              number;
  safety_events_in_history:  number;
  escalations_in_history:    number;
  avg_e2e_latency_ms:        number | null;
  avg_call_duration_s:       number | null;
  ragas_faithfulness:        number | null;
  ragas_answer_relevancy:    number | null;
  safety_gate_accuracy:      number;
  sse_subscribers:           number;
  event_history_size:        number;
}

// ── SSE event types 

export type EmmaEventType =
  | "call_started" | "call_ended" | "call_streaming"
  | "transcript"   | "agent_response" | "safety_event"
  | "tts_start"    | "tts_end" | "error" | "heartbeat";

export interface EmmaEvent {
  type:        EmmaEventType;
  timestamp:   string;
  room_name?:  string;
  tenant_id?:  string;
  // transcript
  text?:       string;
  confidence?: number;
  final?:      boolean;
  // agent_response
  agent?:      string;
  latency_ms?: number;
  // safety_event
  escalated?:  boolean;
  trigger?:    string;
  // call_started / call_ended
  caller_number?:      string;
  destination_number?: string;
  duration_seconds?:   number;
  reason?:             string;
  // error
  error?:      string;
  message?:    string;
}