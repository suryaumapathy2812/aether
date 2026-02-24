"use client";

import { useMemo } from "react";
import { useRealtime } from "@/components/RealtimeProvider";

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

export function useAgentStatus(): AgentStatus {
  const { connState } = useRealtime();

  return useMemo<AgentStatus>(() => {
    if (connState === "thinking") return "thinking";
    if (connState === "listening") return "listening";
    if (connState === "connected") return "connected";
    if (connState === "connecting") return "unknown";
    return "disconnected";
  }, [connState]);
}
