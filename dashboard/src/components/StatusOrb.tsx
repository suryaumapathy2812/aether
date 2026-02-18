"use client";

import type { AgentStatus } from "@/hooks/useAgentStatus";

/**
 * StatusOrb — a small glowing dot that reflects the agent's live state.
 *
 * Pure visual indicator, no interactivity. Sits in the page header.
 *
 * States:
 *   connected    — soft white glow, slow breathe
 *   listening    — brighter white, gentle pulse
 *   thinking     — faster pulse, shimmer
 *   disconnected — dim grey, no animation
 *   unknown      — very faint, no animation
 */
export default function StatusOrb({
  status,
  size = 8,
}: {
  status: AgentStatus;
  size?: number;
}) {
  const config = orbConfig[status];

  return (
    <span
      className="inline-block rounded-full shrink-0"
      aria-label={`Agent status: ${status}`}
      style={{
        width: size,
        height: size,
        backgroundColor: config.color,
        boxShadow: config.glow,
        animation: config.animation,
        opacity: config.opacity,
        transition: "background-color 0.6s ease, box-shadow 0.6s ease, opacity 0.6s ease",
      }}
    />
  );
}

const orbConfig: Record<
  AgentStatus,
  { color: string; glow: string; animation: string; opacity: number }
> = {
  connected: {
    color: "rgba(255, 255, 255, 0.85)",
    glow: "0 0 6px rgba(255, 255, 255, 0.15), 0 0 12px rgba(255, 255, 255, 0.05)",
    animation: "status-orb-breathe 4s ease-in-out infinite",
    opacity: 1,
  },
  listening: {
    color: "rgba(255, 255, 255, 0.95)",
    glow: "0 0 8px rgba(255, 255, 255, 0.25), 0 0 16px rgba(255, 255, 255, 0.08)",
    animation: "status-orb-pulse 2s ease-in-out infinite",
    opacity: 1,
  },
  thinking: {
    color: "rgba(255, 255, 255, 0.9)",
    glow: "0 0 10px rgba(255, 255, 255, 0.3), 0 0 20px rgba(255, 255, 255, 0.1)",
    animation: "status-orb-think 1.2s ease-in-out infinite",
    opacity: 1,
  },
  disconnected: {
    color: "rgba(255, 255, 255, 0.2)",
    glow: "none",
    animation: "none",
    opacity: 0.5,
  },
  unknown: {
    color: "rgba(255, 255, 255, 0.1)",
    glow: "none",
    animation: "none",
    opacity: 0.3,
  },
};
