import { useEffect, useState } from "react";

import {
  ensureDirectAgentConnection,
  getDirectAgentConnection,
  type DirectAgentConnection,
} from "#/lib/api";

type DirectAgentState = {
  connection: DirectAgentConnection | null;
  loading: boolean;
  error: string | null;
};

export function useDirectAgent(): DirectAgentState {
  const [state, setState] = useState<DirectAgentState>({
    connection: getDirectAgentConnection(),
    loading: true,
    error: null,
  });

  useEffect(() => {
    let cancelled = false;
    let refreshTimer: ReturnType<typeof setTimeout> | null = null;

    async function load(force = false) {
      const connection = await ensureDirectAgentConnection(force);
      if (cancelled) return;
      setState({
        connection,
        loading: false,
        error: connection ? null : "Direct agent unavailable",
      });
      if (!connection) return;
      const refreshInMs = Math.max(connection.expiresAt * 1000 - Date.now() - 5 * 60 * 1000, 30_000);
      refreshTimer = setTimeout(() => {
        void load(true);
      }, refreshInMs);
    }

    void load();

    return () => {
      cancelled = true;
      if (refreshTimer) clearTimeout(refreshTimer);
    };
  }, []);

  return state;
}
