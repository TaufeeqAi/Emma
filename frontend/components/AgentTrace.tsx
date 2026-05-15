import { useMemo } from "react";
import type { EmmaEvent } from "../lib/types";

interface Props {
  events:   EmmaEvent[];
  roomName: string | null;
}

interface TraceStep {
  id:        string;
  timestamp: string;
  label:     string;
  detail:    string;
  color:     string;
  icon:      string;
}

function agentLabel(event: EmmaEvent): string {
  switch (event.type) {
    case "call_started":    return "Call connected";
    case "call_streaming":  return "Pipeline active";
    case "transcript":      return event.final ? "STT (final)" : "STT (interim)";
    case "safety_event":    return event.escalated ? "🚨 ESCALATED" : "Safety ✓";
    case "agent_response":  return `${event.agent ?? "Agent"} responded`;
    case "tts_start":       return "TTS started";
    case "tts_end":         return "TTS finished";
    case "call_ended":      return "Call ended";
    case "error":           return "Error";
    default:                return event.type;
  }
}

function agentDetail(event: EmmaEvent): string {
  if (event.text && event.text.length > 80)
    return event.text.slice(0, 77) + "…";
  if (event.text)    return event.text;
  if (event.reason)  return `reason: ${event.reason}`;
  if (event.message) return event.message;
  if (event.error)   return event.error;
  return "";
}

function stepColor(event: EmmaEvent): string {
  if (event.type === "safety_event" && event.escalated) return "border-red-500 bg-red-50";
  if (event.type === "call_started")  return "border-green-500 bg-green-50";
  if (event.type === "call_ended")    return "border-gray-400 bg-gray-50";
  if (event.type === "error")         return "border-red-400 bg-red-50";
  if (event.type === "agent_response") return "border-nhs-blue bg-blue-50";
  return "border-gray-300 bg-white";
}

const TYPE_ICONS: Partial<Record<EmmaEvent["type"], string>> = {
  call_started:   "📞",
  call_streaming: "🔊",
  transcript:     "🎤",
  safety_event:   "🛡️",
  agent_response: "🤖",
  tts_start:      "🔈",
  tts_end:        "✅",
  call_ended:     "📴",
  error:          "❌",
};

export default function AgentTrace({ events, roomName }: Props) {
  const steps = useMemo<TraceStep[]>(() => {
    if (!roomName) return [];
    return events
      .filter((e) => e.room_name === roomName && e.type !== "heartbeat")
      .slice(-30)
      .map((e, i) => ({
        id:        `${i}-${e.timestamp}`,
        timestamp: e.timestamp,
        label:     agentLabel(e),
        detail:    agentDetail(e),
        color:     stepColor(e),
        icon:      TYPE_ICONS[e.type] ?? "·",
      }));
  }, [events, roomName]);

  if (!roomName) {
    return (
      <div className="text-gray-400 text-sm text-center py-8">
        Select an active call
      </div>
    );
  }

  if (steps.length === 0) {
    return (
      <div className="text-gray-400 text-sm text-center py-8 animate-pulse">
        Waiting for agent events…
      </div>
    );
  }

  return (
    <div className="relative space-y-2 max-h-96 overflow-y-auto pl-4 pr-1">
      {/* Vertical line */}
      <div className="absolute left-6 top-2 bottom-2 w-0.5 bg-gray-200" />

      {steps.map((step) => (
        <div key={step.id} className="relative flex gap-3">
          {/* Timeline dot */}
          <div className="flex-shrink-0 w-6 h-6 rounded-full bg-white border-2 border-gray-300 flex items-center justify-center text-xs z-10">
            {step.icon}
          </div>

          <div className={`flex-1 rounded border px-3 py-2 text-xs ${step.color}`}>
            <div className="flex justify-between items-start">
              <span className="font-semibold text-gray-700">{step.label}</span>
              <span className="text-gray-400 ml-2 whitespace-nowrap">
                {new Date(step.timestamp).toLocaleTimeString()}
              </span>
            </div>
            {step.detail && (
              <p className="mt-0.5 text-gray-500 truncate">{step.detail}</p>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}