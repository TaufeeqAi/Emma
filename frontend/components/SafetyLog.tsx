import { useMemo } from "react";
import type { EmmaEvent } from "../lib/types";
import StatusBadge from "./StatusBadge";

interface Props {
  events: EmmaEvent[];
}

export default function SafetyLog({ events }: Props) {
  const safetyEvents = useMemo(
    () =>
      events
        .filter((e) => e.type === "safety_event")
        .slice()
        .reverse(),
    [events]
  );

  if (safetyEvents.length === 0) {
    return (
      <div className="text-center py-8 text-gray-400 text-sm">
        No safety events
      </div>
    );
  }

  return (
    <div className="space-y-2 max-h-72 overflow-y-auto">
      {safetyEvents.map((e, i) => (
        <div
          key={`safety-${i}-${e.timestamp}`}
          className={`rounded border px-3 py-2 text-sm
            ${e.escalated
              ? "border-red-400 bg-red-50"
              : "border-green-300 bg-green-50"}`}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span>{e.escalated ? "🚨" : "✅"}</span>
              <StatusBadge
                label={e.escalated ? "ESCALATED" : "Safe"}
                variant={e.escalated ? "red" : "green"}
              />
              {e.room_name && (
                <span className="text-xs text-gray-500 truncate max-w-[120px]">
                  {e.room_name.split("-").slice(-1)[0]}
                </span>
              )}
            </div>
            <span className="text-xs text-gray-400">
              {new Date(e.timestamp).toLocaleTimeString()}
            </span>
          </div>
          {e.trigger && (
            <p className="mt-1 text-xs text-gray-600">
              Trigger: <span className="font-mono">{e.trigger}</span>
            </p>
          )}
          {e.escalated && (
            <p className="mt-1 text-xs font-semibold text-red-700">
              ↳ Emergency services advised (hardcoded — not LLM-decided)
            </p>
          )}
        </div>
      ))}
    </div>
  );
}