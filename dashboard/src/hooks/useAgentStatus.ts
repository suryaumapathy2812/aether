"use client";

import { useEffect, useRef, useState } from "react";
import { getSessionToken } from "@/lib/api";

/**
 * Agent status — what the orb reflects.
 *
 * - "connected"    — agent is reachable and healthy
 * - "thinking"     — agent is processing (set externally by chat page)
 * - "listening"    — agent is listening for input (set externally by chat page)
 * - "disconnected" — agent unreachable or no session
 * - "unknown"      — initial state before first check
 */
export type AgentStatus =
  | "connected"
  | "thinking"
  | "listening"
  | "disconnected"
  | "unknown";

const ORCHESTRATOR_URL =
  process.env.NEXT_PUBLIC_ORCHESTRATOR_URL || "http://localhost:9000";

const POLL_INTERVAL = 10_000; // 10 seconds

/**
 * Poll the orchestrator /health endpoint to determine agent reachability.
 *
 * Used on non-chat pages. Chat page manages its own status via WS events
 * and passes it directly to StatusOrb.
 */
export function useAgentStatus(): AgentStatus {
  const [status, setStatus] = useState<AgentStatus>("unknown");
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function check() {
      const token = getSessionToken();
      if (!token) {
        if (!cancelled) setStatus("disconnected");
        return;
      }

      try {
        const res = await fetch(`${ORCHESTRATOR_URL}/health`, {
          signal: AbortSignal.timeout(5000),
        });
        if (!cancelled) {
          setStatus(res.ok ? "connected" : "disconnected");
        }
      } catch {
        if (!cancelled) setStatus("disconnected");
      }
    }

    check();
    intervalRef.current = setInterval(check, POLL_INTERVAL);

    return () => {
      cancelled = true;
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  return status;
}
