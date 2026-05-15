import { useMemo } from "react";
import type { EmmaEvent } from "../lib/types";
import StatusBadge from "./StatusBadge";

interface Props {
  events:   EmmaEvent[];
  roomName: string | null;
}

interface TranscriptLine {
  id:         string;
  speaker:    "patient" | "emma";
  text:       string;
  confidence: number | undefined;
  timestamp:  string;
  final:      boolean;
  latency_ms: number | undefined;
}

export default function CallTranscript({ events, roomName }: Props) {
  const lines = useMemo<TranscriptLine[]>(() => {
    if (!roomName) return [];

    const relevant = events.filter(
      (e) =>
        e.room_name === roomName &&
        (e.type === "transcript" || e.type === "agent_response")
    );

    return relevant.map((e, i) => ({
      id:         `${i}-${e.timestamp}`,
      speaker:    e.type === "transcript" ? "patient" : "emma",
      text:       e.text ?? "",
      confidence: e.confidence,
      timestamp:  e.timestamp,
      final:      e.final ?? true,
      latency_ms: e.latency_ms,
    }));
  }, [events, roomName]);

  if (!roomName) {
    return (
      <div className="flex items-center justify-center h-40 text-gray-400 text-sm">
        Select an active call to view transcript
      </div>
    );
  }

  if (lines.length === 0) {
    return (
      <div className="flex items-center justify-center h-40 text-gray-400 text-sm animate-pulse">
        Waiting for speech…
      </div>
    );
  }

  return (
    <div className="space-y-2 max-h-96 overflow-y-auto pr-1">
      {lines.map((line) => (
        <div
          key={line.id}
          className={`flex gap-3 ${line.speaker === "emma" ? "flex-row-reverse" : ""}`}
        >
          <div
            className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold
              ${line.speaker === "emma"
                ? "bg-nhs-blue text-white"
                : "bg-gray-200 text-gray-600"}`}
          >
            {line.speaker === "emma" ? "E" : "P"}
          </div>

          <div
            className={`max-w-[78%] rounded-lg px-3 py-2 text-sm shadow-sm
              ${line.speaker === "emma"
                ? "bg-nhs-blue text-white rounded-tr-none"
                : "bg-white border border-gray-200 text-gray-800 rounded-tl-none"}`}
          >
            <p className={`${!line.final ? "opacity-60 italic" : ""}`}>
              {line.text}
            </p>
            <div
              className={`mt-1 flex gap-2 text-xs
                ${line.speaker === "emma" ? "text-nhs-blue-lt" : "text-gray-400"}`}
            >
              <span>{new Date(line.timestamp).toLocaleTimeString()}</span>
              {line.confidence !== undefined && (
                <span>conf: {(line.confidence * 100).toFixed(0)}%</span>
              )}
              {line.latency_ms !== undefined && (
                <span>⚡ {Math.round(line.latency_ms)}ms</span>
              )}
              {!line.final && <span>…</span>}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}