import useSWR from "swr";
import { api } from "../lib/api";
import type { MetricsSummary } from "../lib/types";

function MetricCard({
  label,
  value,
  unit = "",
  highlight = false,
}: {
  label:      string;
  value:      string | number | null;
  unit?:      string;
  highlight?: boolean;
}) {
  return (
    <div className={`bg-white rounded-lg border p-4 shadow-sm ${highlight ? "border-nhs-blue" : "border-gray-200"}`}>
      <p className="text-xs text-gray-500 mb-1 uppercase tracking-wide">{label}</p>
      <p className={`text-2xl font-bold ${highlight ? "text-nhs-blue" : "text-gray-800"}`}>
        {value !== null && value !== undefined ? `${value}${unit}` : "—"}
      </p>
    </div>
  );
}

interface Props {
  /** If provided, also show SSE connection status */
  sseConnected?: boolean;
}

export default function MetricsPanel({ sseConnected }: Props) {
  const { data, error } = useSWR<MetricsSummary>(
    "metrics_summary",
    () => api.metrics(),
    { refreshInterval: 5_000 }
  );

  if (error)  return <p className="text-red-500 text-sm">Failed to load metrics.</p>;
  if (!data)  return <p className="text-gray-400 text-sm animate-pulse">Loading metrics…</p>;

  const faithfulness = data.ragas_faithfulness !== null
    ? (data.ragas_faithfulness * 100).toFixed(0) + "%"
    : null;

  const safetyPct = (data.safety_gate_accuracy * 100).toFixed(0) + "%";

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
      <MetricCard
        label="Active Calls"
        value={data.active_calls}
        highlight={data.active_calls > 0}
      />
      <MetricCard
        label="Calls (session)"
        value={data.calls_in_history}
      />
      <MetricCard
        label="Avg Latency"
        value={data.avg_e2e_latency_ms !== null ? Math.round(data.avg_e2e_latency_ms) : null}
        unit="ms"
      />
      <MetricCard
        label="Avg Duration"
        value={data.avg_call_duration_s !== null ? Math.round(data.avg_call_duration_s) : null}
        unit="s"
      />
      <MetricCard
        label="RAGAS Faithfulness"
        value={faithfulness}
      />
      <MetricCard
        label="Safety Gate"
        value={safetyPct}
        highlight
      />
    </div>
  );
}