"use client";

import { useEffect, useState } from "react";
import { fetchWithAuth } from "@/lib/api";

export type AgentStatus =
  | "connected"
  | "thinking"
  | "listening"
  | "disconnected"
  | "unknown";

export function useAgentStatus(): AgentStatus {
  const [status, setStatus] = useState<AgentStatus>("unknown");

  useEffect(() => {
    let cancelled = false;
    async function check(): Promise<void> {
      try {
        const res = await fetchWithAuth("/health", { method: "GET" });
        if (cancelled) return;
        setStatus(res.ok ? "connected" : "disconnected");
      } catch {
        if (!cancelled) {
          setStatus("disconnected");
        }
      }
    }

    void check();
    const timer = window.setInterval(() => {
      void check();
    }, 10000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  return status;
}
