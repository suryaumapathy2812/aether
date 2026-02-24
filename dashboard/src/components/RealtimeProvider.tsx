"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useSession } from "@/lib/auth-client";

type ChatRole = "user" | "assistant";

export interface ChatMessageItem {
  id: string;
  role: ChatRole;
  text: string;
}

export type RealtimeConnectionState =
  | "connecting"
  | "connected"
  | "listening"
  | "thinking"
  | "disconnected";

interface RealtimeContextValue {
  connState: RealtimeConnectionState;
  errorText: string | null;
  messages: ChatMessageItem[];
  sendText: (text: string) => boolean;
  clearMessages: () => void;
}

const RealtimeContext = createContext<RealtimeContextValue | null>(null);

const ICE_SERVERS: RTCIceServer[] = [{ urls: "stun:stun.l.google.com:19302" }];
const ORCHESTRATOR_URL = process.env.NEXT_PUBLIC_ORCHESTRATOR_URL || "";

function randomId(prefix: string): string {
  return `${prefix}-${Math.random().toString(36).slice(2, 10)}`;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

export default function RealtimeProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const { data: session, isPending } = useSession();
  const token = session?.session?.token || null;

  const [connState, setConnState] = useState<RealtimeConnectionState>("disconnected");
  const [errorText, setErrorText] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessageItem[]>([]);

  const pcRef = useRef<RTCPeerConnection | null>(null);
  const dcRef = useRef<RTCDataChannel | null>(null);
  const pcIdRef = useRef<string | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const pingTimerRef = useRef<number | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const connectPeerFnRef = useRef<(() => Promise<void>) | null>(null);
  const connectSeqRef = useRef(0);
  const connectInFlightRef = useRef(false);
  const mountedRef = useRef(false);
  const replacingConnectionRef = useRef(false);
  const currentAssistantIdRef = useRef<string | null>(null);

  const authHeaders = useCallback((): Record<string, string> => {
    return token ? { Authorization: `Bearer ${token}` } : {};
  }, [token]);

  const appendMessage = useCallback((role: ChatRole, text: string): void => {
    setMessages((prev) => [...prev, { id: randomId(role), role, text }]);
  }, []);

  const upsertAssistantChunk = useCallback((chunk: string): void => {
    const existingId = currentAssistantIdRef.current;
    if (!existingId) {
      const id = randomId("assistant");
      currentAssistantIdRef.current = id;
      setMessages((prev) => [...prev, { id, role: "assistant", text: chunk }]);
      return;
    }
    setMessages((prev) =>
      prev.map((m) => (m.id === existingId ? { ...m, text: `${m.text}${chunk}` } : m))
    );
  }, []);

  const finalizeAssistantTurn = useCallback((): void => {
    currentAssistantIdRef.current = null;
    setConnState("connected");
  }, []);

  const postOffer = useCallback(
    async (
      localDescription: RTCSessionDescriptionInit
    ): Promise<{ sdp: string; type: string; pc_id: string }> => {
      const response = await fetch(`${ORCHESTRATOR_URL}/api/webrtc/offer`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...authHeaders(),
        },
        body: JSON.stringify({
          sdp: localDescription.sdp,
          type: localDescription.type,
        }),
      });

      if (!response.ok) {
        throw new Error(`Offer failed (${response.status})`);
      }
      return response.json();
    },
    [authHeaders]
  );

  const postIce = useCallback(
    async (pcId: string, candidate: RTCIceCandidate): Promise<void> => {
      await fetch(`${ORCHESTRATOR_URL}/api/webrtc/ice`, {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
          ...authHeaders(),
        },
        body: JSON.stringify({
          pc_id: pcId,
          candidates: [
            {
              candidate: candidate.candidate,
              sdpMid: candidate.sdpMid,
              sdpMLineIndex: candidate.sdpMLineIndex,
            },
          ],
        }),
      });
    },
    [authHeaders]
  );

  const waitForAgentReady = useCallback(async (): Promise<void> => {
    for (let attempt = 0; attempt < 8; attempt += 1) {
      const response = await fetch(`${ORCHESTRATOR_URL}/api/agent/ready`, {
        method: "GET",
        headers: authHeaders(),
      });

      if (response.ok) {
        const body = (await response.json()) as { ready?: boolean };
        if (body.ready) return;
      }

      await sleep(Math.min(400 * 2 ** attempt, 2500));
    }
    throw new Error("agent unavailable");
  }, [authHeaders]);

  const disconnectPeer = useCallback(async (sendStop: boolean): Promise<void> => {
    if (reconnectTimerRef.current !== null) {
      window.clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    if (pingTimerRef.current !== null) {
      window.clearInterval(pingTimerRef.current);
      pingTimerRef.current = null;
    }

    const dc = dcRef.current;
    const pc = pcRef.current;
    dcRef.current = null;
    pcRef.current = null;
    if (!sendStop) {
      pcIdRef.current = null;
    }

    if (sendStop && dc && dc.readyState === "open") {
      try {
        dc.send(JSON.stringify({ type: "stream_stop", data: {} }));
      } catch {
        // ignore
      }
    }

    try {
      dc?.close();
    } catch {
      // ignore
    }
    try {
      pc?.close();
    } catch {
      // ignore
    }
  }, []);

  const scheduleReconnect = useCallback((): void => {
    if (!mountedRef.current || replacingConnectionRef.current) {
      return;
    }
    if (reconnectTimerRef.current !== null) return;

    const attempt = reconnectAttemptsRef.current + 1;
    reconnectAttemptsRef.current = attempt;
    const delayMs = Math.min(1000 * 2 ** (attempt - 1), 10000);

    reconnectTimerRef.current = window.setTimeout(() => {
      reconnectTimerRef.current = null;
      const reconnectFn = connectPeerFnRef.current;
      if (!reconnectFn) return;
      void reconnectFn().catch((error: unknown) => {
        setErrorText(error instanceof Error ? error.message : "failed to reconnect");
        scheduleReconnect();
      });
    }, delayMs);
  }, []);

  const connectPeer = useCallback(async (): Promise<void> => {
    if (!token || !mountedRef.current) return;
    if (connectInFlightRef.current) return;
    connectInFlightRef.current = true;

    const connectSeq = ++connectSeqRef.current;
    const isStaleConnection = (): boolean => {
      return !mountedRef.current || connectSeqRef.current !== connectSeq;
    };

    try {
      replacingConnectionRef.current = true;
      await disconnectPeer(false);
      replacingConnectionRef.current = false;

      if (isStaleConnection()) return;

      setConnState("connecting");
      setErrorText(null);

      await waitForAgentReady();
      if (isStaleConnection()) return;

      const pc = new RTCPeerConnection({ iceServers: ICE_SERVERS });
      pcRef.current = pc;

      if (isStaleConnection()) {
        try {
          pc.close();
        } catch {
          // ignore
        }
        return;
      }

      pc.addTransceiver("audio", { direction: "recvonly" });

      const channel = pc.createDataChannel("aether-events");
      dcRef.current = channel;

      channel.onopen = () => {
        if (dcRef.current !== channel || pcRef.current !== pc) return;
        reconnectAttemptsRef.current = 0;
        setConnState("connected");
        channel.send(JSON.stringify({ type: "stream_start", data: {} }));
        if (pingTimerRef.current !== null) {
          window.clearInterval(pingTimerRef.current);
        }
        pingTimerRef.current = window.setInterval(() => {
          if (channel.readyState === "open") {
            channel.send(JSON.stringify({ type: "ping", data: {} }));
          }
        }, 15000);
      };

      channel.onclose = () => {
        if (dcRef.current !== channel || pcRef.current !== pc) return;
        if (pingTimerRef.current !== null) {
          window.clearInterval(pingTimerRef.current);
          pingTimerRef.current = null;
        }
        setConnState("disconnected");
        scheduleReconnect();
      };

      channel.onerror = () => {
        if (dcRef.current !== channel || pcRef.current !== pc) return;
        if (pingTimerRef.current !== null) {
          window.clearInterval(pingTimerRef.current);
          pingTimerRef.current = null;
        }
        setConnState("disconnected");
        setErrorText("realtime connection error");
        scheduleReconnect();
      };

      channel.onmessage = (event) => {
        if (dcRef.current !== channel || pcRef.current !== pc) return;
        try {
          const payload = JSON.parse(String(event.data));
          const type = String(payload.type || "");
          const data = payload.data;

          if (type === "text_chunk") {
            setConnState("thinking");
            upsertAssistantChunk(String(data || ""));
            return;
          }
          if (type === "stream_end") {
            finalizeAssistantTurn();
            return;
          }
          if (type === "status") {
            const state = String(data || "");
            if (state === "thinking" || state === "speaking") {
              setConnState("thinking");
            } else if (state === "listening") {
              setConnState("listening");
            } else if (state === "recovered") {
              setConnState("connected");
            } else if (state === "reconnecting") {
              setConnState("disconnected");
            }
            return;
          }
          if (type === "error") {
            setErrorText(String(data || "realtime error"));
            setConnState("disconnected");
            scheduleReconnect();
            return;
          }
          if (type === "pong") {
            setConnState((prev) => (prev === "thinking" ? prev : "connected"));
          }
        } catch {
          // ignore malformed side-channel payloads
        }
      };

      pc.onconnectionstatechange = () => {
        if (pcRef.current !== pc) return;
        const state = pc.connectionState;
        if (state === "connected") {
          setConnState((prev) => (prev === "thinking" ? prev : "connected"));
        } else if (
          state === "failed" ||
          state === "disconnected" ||
          state === "closed"
        ) {
          setConnState("disconnected");
          scheduleReconnect();
        }
      };

      pc.onicecandidate = (event) => {
        if (pcRef.current !== pc) return;
        if (!event.candidate) return;
        const pcId = pcIdRef.current;
        if (!pcId) return;
        void postIce(pcId, event.candidate);
      };

      pc.ontrack = (event) => {
        if (pcRef.current !== pc) return;
        const [stream] = event.streams;
        if (!stream) return;
        const audio = new Audio();
        audio.autoplay = true;
        audio.srcObject = stream;
        void audio.play().catch(() => {});
      };

      const offer = await pc.createOffer();
      if (isStaleConnection() || pcRef.current !== pc) return;

      await pc.setLocalDescription(offer);
      if (isStaleConnection() || pcRef.current !== pc) return;

      const answer = await postOffer(offer);
      if (isStaleConnection() || pcRef.current !== pc) return;

      pcIdRef.current = answer.pc_id;

      try {
        await pc.setRemoteDescription({
          type: answer.type as RTCSdpType,
          sdp: answer.sdp,
        });
      } catch (error) {
        if (
          error instanceof DOMException &&
          error.name === "InvalidStateError" &&
          pcRef.current !== pc
        ) {
          return;
        }
        throw error;
      }
    } finally {
      connectInFlightRef.current = false;
    }
  }, [disconnectPeer, finalizeAssistantTurn, postIce, postOffer, scheduleReconnect, token, upsertAssistantChunk, waitForAgentReady]);

  useEffect(() => {
    connectPeerFnRef.current = connectPeer;
  }, [connectPeer]);

  useEffect(() => {
    if (isPending) return;

    if (!token) {
      mountedRef.current = false;
      void disconnectPeer(true);
      setConnState("disconnected");
      return;
    }

    mountedRef.current = true;
    void connectPeer();

    return () => {
      mountedRef.current = false;
      void disconnectPeer(true);
    };
  }, [connectPeer, disconnectPeer, isPending, token]);

  const sendText = useCallback(
    (text: string): boolean => {
      const dc = dcRef.current;
      if (!text.trim()) return false;
      if (!dc || dc.readyState !== "open") {
        setErrorText("realtime connection is not ready");
        setConnState("disconnected");
        scheduleReconnect();
        return false;
      }

      appendMessage("user", text);
      setErrorText(null);
      setConnState("thinking");
      currentAssistantIdRef.current = null;

      try {
        dc.send(JSON.stringify({ type: "text", data: text }));
        return true;
      } catch {
        setErrorText("failed to send message");
        setConnState("disconnected");
        scheduleReconnect();
        return false;
      }
    },
    [appendMessage, scheduleReconnect]
  );

  const clearMessages = useCallback(() => {
    setMessages([]);
    currentAssistantIdRef.current = null;
  }, []);

  const value = useMemo<RealtimeContextValue>(
    () => ({
      connState,
      errorText,
      messages,
      sendText,
      clearMessages,
    }),
    [clearMessages, connState, errorText, messages, sendText]
  );

  return <RealtimeContext.Provider value={value}>{children}</RealtimeContext.Provider>;
}

export function useRealtime(): RealtimeContextValue {
  const context = useContext(RealtimeContext);
  if (!context) {
    throw new Error("useRealtime must be used inside RealtimeProvider");
  }
  return context;
}
