import { useState, useMemo } from "react";
import useSWR from "swr";
import type { NextPage } from "next";
import type { CallsResponse, ActiveCall, HealthResponse } from "../lib/types";
import { api } from "../lib/api";
import { useSSE } from "../lib/useSSE";
import Layout from "../components/Layout";
import MetricsPanel from "../components/MetricsPanel";
import CallTranscript from "../components/CallTranscript";
import AgentTrace from "../components/AgentTrace";
import SafetyLog from "../components/SafetyLog";
import TenantSelector from "../components/TenantSelector";
import StatusBadge from "../components/StatusBadge";

function callStateVariant(state: string): "green" | "yellow" | "red" | "grey" {
  if (state === "STREAMING")  return "green";
  if (state === "CONNECTED")  return "yellow";
  if (state === "CONNECTING") return "yellow";
  if (state === "ERROR")      return "red";
  if (state === "TIMED_OUT")  return "red";
  return "grey";
}

const LiveMonitor: NextPage = () => {
  const [selectedRoom, setSelectedRoom] = useState<string | null>(null);
  const [tenantFilter, setTenantFilter] = useState<string | null>(null);

  // REST polling
  const { data: health } = useSWR<HealthResponse>(
    "health",
    () => api.health(),
    { refreshInterval: 5_000 }
  );
  const { data: callsData } = useSWR<CallsResponse>(
    "calls",
    () => api.calls(),
    { refreshInterval: 2_000 }
  );

  // SSE stream
  const { events, connected: sseConnected, error: sseError, clearEvents } =
    useSSE({ url: api.eventsUrl() });

  // Filtered calls
  const calls = useMemo<ActiveCall[]>(() => {
    const all = callsData?.calls ?? [];
    if (!tenantFilter) return all;
    return all.filter((c) => c.tenant_id === tenantFilter);
  }, [callsData, tenantFilter]);

  const handleHangup = async (roomName: string) => {
    if (!confirm(`Hang up call ${roomName}?`)) return;
    try {
      await api.hangupCall(roomName);
    } catch (e) {
      alert(`Hangup failed: ${e}`);
    }
  };

  return (
    <Layout title="Live Call Monitor">
      {/* Status bar */}
      <div className="flex flex-wrap items-center gap-3 mb-5">
        <StatusBadge
          label={health?.status === "ok" ? "Backend OK" : "Backend Unavailable"}
          variant={health?.status === "ok" ? "green" : "red"}
          dot
        />
        <StatusBadge
          label={sseConnected ? "SSE Connected" : "SSE Disconnected"}
          variant={sseConnected ? "green" : "yellow"}
          dot
        />
        {sseError && (
          <span className="text-xs text-yellow-700">{sseError}</span>
        )}
        <div className="ml-auto flex items-center gap-3">
          <TenantSelector value={tenantFilter} onChange={setTenantFilter} />
          <button
            onClick={clearEvents}
            className="text-xs text-gray-500 hover:text-gray-700 underline"
          >
            Clear events
          </button>
        </div>
      </div>

      {/* Metrics panel */}
      <div className="mb-6">
        <MetricsPanel sseConnected={sseConnected} />
      </div>

      {/* Main 3-column layout */}
      <div className="grid grid-cols-12 gap-4">

        {/* Active Calls — left column */}
        <div className="col-span-12 lg:col-span-3">
          <div className="bg-white rounded-lg border border-gray-200 shadow-sm">
            <div className="px-4 py-3 border-b border-gray-200 flex items-center justify-between">
              <h2 className="font-semibold text-gray-700 text-sm">
                Active Calls{" "}
                {callsData?.active_count !== undefined && (
                  <span className="ml-1 text-nhs-blue">({callsData.active_count})</span>
                )}
              </h2>
            </div>

            {calls.length === 0 ? (
              <div className="p-6 text-center text-gray-400 text-sm">
                No active calls
              </div>
            ) : (
              <div className="divide-y divide-gray-100">
                {calls.map((call) => (
                  <button
                    key={call.room_name}
                    onClick={() => setSelectedRoom(
                      selectedRoom === call.room_name ? null : call.room_name
                    )}
                    className={`w-full text-left px-4 py-3 hover:bg-gray-50 transition-colors
                      ${selectedRoom === call.room_name ? "bg-blue-50 border-l-2 border-nhs-blue" : ""}`}
                  >
                    <div className="flex justify-between items-start mb-1">
                      <span className="text-xs font-mono text-gray-500 truncate max-w-[120px]">
                        {call.call_uuid.slice(0, 8)}
                      </span>
                      <StatusBadge
                        label={call.state}
                        variant={callStateVariant(call.state)}
                        dot={call.state === "STREAMING"}
                      />
                    </div>
                    <p className="text-sm text-gray-800 font-medium truncate">
                      {call.caller_number}
                    </p>
                    <div className="flex justify-between mt-1 text-xs text-gray-400">
                      <span>{call.tenant_id.replace("surgery_", "")}</span>
                      <span>{Math.round(call.duration_seconds)}s</span>
                    </div>
                    {call.state !== "ENDED" && (
                      <button
                        onClick={(e) => { e.stopPropagation(); handleHangup(call.room_name); }}
                        className="mt-2 w-full text-xs text-red-600 hover:text-red-800 border border-red-200 rounded py-0.5"
                      >
                        Hang up
                      </button>
                    )}
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Safety log (below calls) */}
          <div className="mt-4 bg-white rounded-lg border border-gray-200 shadow-sm">
            <div className="px-4 py-3 border-b border-gray-200">
              <h2 className="font-semibold text-gray-700 text-sm">🛡️ Safety Events</h2>
            </div>
            <div className="p-3">
              <SafetyLog events={events} />
            </div>
          </div>
        </div>

        {/* Transcript — centre column */}
        <div className="col-span-12 lg:col-span-5">
          <div className="bg-white rounded-lg border border-gray-200 shadow-sm h-full">
            <div className="px-4 py-3 border-b border-gray-200 flex items-center justify-between">
              <h2 className="font-semibold text-gray-700 text-sm">
                🎤 Transcript
                {selectedRoom && (
                  <span className="ml-2 text-xs font-mono text-gray-400">
                    {selectedRoom.slice(-8)}
                  </span>
                )}
              </h2>
              {selectedRoom && (
                <button
                  onClick={() => setSelectedRoom(null)}
                  className="text-xs text-gray-400 hover:text-gray-600"
                >
                  ✕ deselect
                </button>
              )}
            </div>
            <div className="p-4">
              <CallTranscript events={events} roomName={selectedRoom} />
            </div>
          </div>
        </div>

        {/* Agent trace — right column */}
        <div className="col-span-12 lg:col-span-4">
          <div className="bg-white rounded-lg border border-gray-200 shadow-sm">
            <div className="px-4 py-3 border-b border-gray-200">
              <h2 className="font-semibold text-gray-700 text-sm">
                🤖 Agent Decision Trace
              </h2>
            </div>
            <div className="p-4">
              <AgentTrace events={events} roomName={selectedRoom} />
            </div>
          </div>
        </div>

      </div>
    </Layout>
  );
};

export default LiveMonitor;