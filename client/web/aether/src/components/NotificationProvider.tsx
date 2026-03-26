import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import { useSession } from "#/lib/auth-client";
import { toast } from "sonner";
import { directAgentWs, ensureDirectAgentConnection, orchestratorWs } from "#/lib/api";

// ── Types ──

export interface AgentNotification {
  id: string;
  type: string; // "notification" | "task_update"
  title: string;
  body: string;
  timestamp: number;
  read: boolean;
  payload?: Record<string, unknown>;
}

interface NotificationContextValue {
  notifications: AgentNotification[];
  unreadCount: number;
  connected: boolean;
  pendingApprovals: never[];
  loadingApprovals: boolean;
  markRead: (id: string) => void;
  markAllRead: () => void;
  clearAll: () => void;
}

const NotificationContext = createContext<NotificationContextValue | null>(null);

const MAX_NOTIFICATIONS = 100;
const PING_INTERVAL = 25_000;

const NOTIFICATIONS_STORAGE_PREFIX = "aether.notifications";

type NotificationToastOptions = {
  duration?: number;
  action?: {
    label: string;
    onClick: () => void;
  };
};

function randomId(): string {
  return `notif-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function openApprovalScreen(taskId?: string): void {
  const params = new URLSearchParams();
  if (taskId) params.set("task", taskId);
  const href = params.size > 0 ? `/agent?${params.toString()}` : "/agent";
  window.location.href = href;
}

// ── Provider ──

export default function NotificationProvider({ children }: { children: React.ReactNode }) {
  const { data: session } = useSession();
  const userId = session?.user?.id || "";
  const token = session?.session?.token || "";

  const [notifications, setNotifications] = useState<AgentNotification[]>([]);
  const [connected, setConnected] = useState(false);
  const pendingApprovals: never[] = [];
  const loadingApprovals = false;

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const mountedRef = useRef(false);
  const pingTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!userId) {
      setNotifications([]);
      return;
    }
    try {
      const raw = window.localStorage.getItem(`${NOTIFICATIONS_STORAGE_PREFIX}.${userId}`);
      if (!raw) {
        setNotifications([]);
        return;
      }
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) {
        setNotifications([]);
        return;
      }
      const restored = parsed
        .filter((item) => item && typeof item === "object")
        .map((item) => ({
          id: String(item.id || randomId()),
          type: String(item.type || "notification"),
          title: String(item.title || "Notification"),
          body: String(item.body || ""),
          timestamp: Number(item.timestamp || Date.now()),
          read: Boolean(item.read),
          payload:
            item.payload && typeof item.payload === "object"
              ? (item.payload as Record<string, unknown>)
              : undefined,
        }))
        .slice(0, MAX_NOTIFICATIONS);
      setNotifications(restored);
    } catch {
      setNotifications([]);
    }
  }, [userId]);

  useEffect(() => {
    if (!userId) return;
    try {
      window.localStorage.setItem(
        `${NOTIFICATIONS_STORAGE_PREFIX}.${userId}`,
        JSON.stringify(notifications),
      );
    } catch {
      // Ignore storage errors; runtime notifications still work.
    }
  }, [notifications, userId]);

  // ── Helpers (stable — no deps on callbacks) ──

  const addNotification = useCallback(
    (notif: AgentNotification, options?: NotificationToastOptions) => {
      setNotifications((prev) => {
        const next = [notif, ...prev];
        return next.length > MAX_NOTIFICATIONS ? next.slice(0, MAX_NOTIFICATIONS) : next;
      });
      toast(notif.title, {
        description: notif.body,
        duration: options?.duration ?? 5000,
        action: options?.action,
      });
    },
    [],
  );

  const addNotificationRef = useRef(addNotification);
  addNotificationRef.current = addNotification;

  // ── Message handler (standalone, no hook deps) ──

  function handleWsMessage(data: string) {
    let msg: { type: string; payload?: Record<string, unknown> };
    try {
      msg = JSON.parse(data);
    } catch {
      return;
    }

    const add = addNotificationRef.current;

    switch (msg.type) {
      case "task_update": {
        const p = msg.payload || {};
        const event = String(p.event || "");
        const title = String(p.title || "Task update");
        const status = String(p.status || "");
        if (event === "waiting_input" || event === "completed" || event === "failed") {
          const taskId = String(p.task_id || "").trim();
          const isWaitingInput = event === "waiting_input";
          add(
            {
              id: randomId(),
              type: "task_update",
              title,
              body: isWaitingInput ? "Waiting for your input" : `Task ${status}`,
              timestamp: Date.now(),
              read: false,
              payload: p,
            },
            isWaitingInput
              ? {
                  duration: 7000,
                  action: {
                    label: "Review",
                    onClick: () => openApprovalScreen(taskId || undefined),
                  },
                }
              : undefined,
          );
        }
        break;
      }
      case "notification": {
        const p = msg.payload || {};
        add({
          id: randomId(),
          type: "notification",
          title: String(p.title || "Notification"),
          body: String(p.text || p.body || ""),
          timestamp: Date.now(),
          read: false,
          payload: p,
        });
        break;
      }
      case "pong":
        break;
      default:
        break;
    }
  }

  // ── WebSocket connect / reconnect ──

  useEffect(() => {
    if (!userId) return;
    mountedRef.current = true;

    function cleanup() {
      if (pingTimerRef.current) {
        clearInterval(pingTimerRef.current);
        pingTimerRef.current = null;
      }
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.close();
        wsRef.current = null;
      }
      setConnected(false);
    }

    function scheduleReconnect() {
      if (!mountedRef.current) return;
      if (reconnectTimerRef.current) return;
      const attempt = reconnectAttemptsRef.current;
      const delay = Math.min(1000 * 2 ** attempt, 30_000);
      reconnectAttemptsRef.current = attempt + 1;
      reconnectTimerRef.current = setTimeout(() => {
        reconnectTimerRef.current = null;
        void openSocket();
      }, delay);
    }

    async function openSocket() {
      if (!mountedRef.current) return;
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.close();
        wsRef.current = null;
      }

      const direct = await ensureDirectAgentConnection();
      const socket = direct
        ? directAgentWs("/agent/v1/ws/notifications", direct)
        : orchestratorWs("/agent/v1/ws/notifications", token || undefined);
      wsRef.current = socket;

      socket.onopen = () => {
        setConnected(true);
        reconnectAttemptsRef.current = 0;
        if (pingTimerRef.current) clearInterval(pingTimerRef.current);
        pingTimerRef.current = setInterval(() => {
          if (socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({ type: "ping" }));
          }
        }, PING_INTERVAL);
      };

      socket.onclose = () => {
        setConnected(false);
        wsRef.current = null;
        if (pingTimerRef.current) {
          clearInterval(pingTimerRef.current);
          pingTimerRef.current = null;
        }
        scheduleReconnect();
      };

      socket.onerror = () => {};

      socket.onmessage = (event) => {
        handleWsMessage(event.data);
      };
    }

    void openSocket();

    return () => {
      mountedRef.current = false;
      cleanup();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userId, token]);

  // ── Actions ──

  const markRead = useCallback((id: string) => {
    setNotifications((prev) => prev.map((n) => (n.id === id ? { ...n, read: true } : n)));
  }, []);

  const markAllRead = useCallback(() => {
    setNotifications((prev) => prev.map((n) => ({ ...n, read: true })));
  }, []);

  const clearAll = useCallback(() => {
    setNotifications([]);
  }, []);

  const unreadCount = notifications.filter((n) => !n.read).length;

  return (
    <NotificationContext.Provider
      value={{
        notifications,
        unreadCount,
        connected,
        pendingApprovals,
        loadingApprovals,
        markRead,
        markAllRead,
        clearAll,
      }}
    >
      {children}
    </NotificationContext.Provider>
  );
}

export function useNotifications(): NotificationContextValue {
  const ctx = useContext(NotificationContext);
  if (!ctx) {
    throw new Error("useNotifications must be used inside NotificationProvider");
  }
  return ctx;
}
