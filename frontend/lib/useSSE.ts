/**
 * React hook for Server-Sent Events with:
 *   - automatic reconnection (exponential back-off)
 *   - event history buffer (last N events)
 *   - typed EmmaEvent parsing
 */

import { useEffect, useRef, useState, useCallback } from "react";
import type { EmmaEvent } from "./types";

const MAX_EVENTS = 200;
const INITIAL_RETRY_MS = 1_000;
const MAX_RETRY_MS = 30_000;

interface UseSSEOptions {
  url: string;
  enabled?: boolean;
}

interface UseSSEReturn {
  events:       EmmaEvent[];
  connected:    boolean;
  error:        string | null;
  clearEvents:  () => void;
  latestByType: (type: EmmaEvent["type"]) => EmmaEvent | undefined;
}

export function useSSE({ url, enabled = true }: UseSSEOptions): UseSSEReturn {
  const [events, setEvents]       = useState<EmmaEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const [error, setError]         = useState<string | null>(null);

  const esRef        = useRef<EventSource | null>(null);
  const retryMs      = useRef(INITIAL_RETRY_MS);
  const retryTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);
  const unmounted    = useRef(false);

  const connect = useCallback(() => {
    if (!enabled || unmounted.current) return;

    const es = new EventSource(url);
    esRef.current = es;

    es.onopen = () => {
      setConnected(true);
      setError(null);
      retryMs.current = INITIAL_RETRY_MS;
    };

    es.onmessage = (e: MessageEvent<string>) => {
      try {
        const event = JSON.parse(e.data) as EmmaEvent;
        if (event.type === "heartbeat") return; // don't store heartbeats
        setEvents((prev) => {
          const next = [...prev, event];
          return next.length > MAX_EVENTS ? next.slice(-MAX_EVENTS) : next;
        });
      } catch {
        // malformed JSON — ignore
      }
    };

    es.onerror = () => {
      es.close();
      setConnected(false);
      if (!unmounted.current) {
        setError(`SSE disconnected — retrying in ${retryMs.current / 1000}s`);
        retryTimeout.current = setTimeout(() => {
          retryMs.current = Math.min(retryMs.current * 2, MAX_RETRY_MS);
          connect();
        }, retryMs.current);
      }
    };
  }, [url, enabled]);

  useEffect(() => {
    unmounted.current = false;
    connect();

    return () => {
      unmounted.current = true;
      esRef.current?.close();
      if (retryTimeout.current) clearTimeout(retryTimeout.current);
    };
  }, [connect]);

  const clearEvents = useCallback(() => setEvents([]), []);

  const latestByType = useCallback(
    (type: EmmaEvent["type"]) =>
      [...events].reverse().find((e) => e.type === type),
    [events]
  );

  return { events, connected, error, clearEvents, latestByType };
}