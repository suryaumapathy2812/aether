"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { ChevronLeft } from "lucide-react";
import { useSession } from "@/lib/auth-client";
import { getSessionToken } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import StatusOrb from "@/components/StatusOrb";
import ChatMessage from "@/components/ChatMessage";
import type { AgentStatus } from "@/hooks/useAgentStatus";

type ChatRole = "user" | "assistant";

interface ChatMessageItem {
  id: string;
  role: ChatRole;
  text: string;
}

type ConnectionState = "connecting" | "connected" | "thinking" | "disconnected";

const ICE_SERVERS: RTCIceServer[] = [{ urls: "stun:stun.l.google.com:19302" }];
const ORCHESTRATOR_URL = process.env.NEXT_PUBLIC_ORCHESTRATOR_URL || "";

function orbStatus(state: ConnectionState): AgentStatus {
  if (state === "thinking") return "thinking";
  if (state === "connected") return "connected";
  return "disconnected";
}

function randomId(prefix: string): string {
  return `${prefix}-${Math.random().toString(36).slice(2, 10)}`;
}

function authHeaders(): Record<string, string> {
  const token = getSessionToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export default function ChatPage() {
  const router = useRouter();
  const { data: session, isPending } = useSession();

  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<ChatMessageItem[]>([]);
  const [connState, setConnState] = useState<ConnectionState>("connecting");
  const [errorText, setErrorText] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const pcRef = useRef<RTCPeerConnection | null>(null);
  const dcRef = useRef<RTCDataChannel | null>(null);
  const pcIdRef = useRef<string | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const pingTimerRef = useRef<number | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const connectSeqRef = useRef(0);
  const connectInFlightRef = useRef(false);
  const mountedRef = useRef(false);
  const currentAssistantIdRef = useRef<string | null>(null);
  const stoppingRef = useRef(false);
  const replacingConnectionRef = useRef(false);

  const isLoading = connState === "thinking";

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (!isPending && !session) router.push("/");
  }, [isPending, session, router]);

  useEffect(() => {
    if (isPending || !session) return;
    mountedRef.current = true;
    void connectPeer();

    return () => {
      mountedRef.current = false;
      void disconnectPeer(true);
    };
  }, [isPending, session]);

  function appendMessage(role: ChatRole, text: string): void {
    setMessages((prev) => [...prev, { id: randomId(role), role, text }]);
  }

  function upsertAssistantChunk(chunk: string): void {
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
  }

  function finalizeAssistantTurn(): void {
    currentAssistantIdRef.current = null;
    setConnState("connected");
  }

  async function postOffer(
    localDescription: RTCSessionDescriptionInit
  ): Promise<{ sdp: string; type: string; pc_id: string }> {
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
  }

  async function postIce(pcId: string, candidate: RTCIceCandidate): Promise<void> {
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
  }

  async function connectPeer(): Promise<void> {
    if (connectInFlightRef.current) return;
    connectInFlightRef.current = true;

    const connectSeq = ++connectSeqRef.current;

    function isStaleConnection(): boolean {
      return !mountedRef.current || connectSeqRef.current !== connectSeq;
    }

    function isClosedPeer(target: RTCPeerConnection): boolean {
      return (
        target.signalingState === "closed" || target.connectionState === "closed"
      );
    }

    try {
      replacingConnectionRef.current = true;
      await disconnectPeer(false);
      replacingConnectionRef.current = false;

      if (isStaleConnection()) return;

      pcIdRef.current = null;
      setConnState("connecting");
      setErrorText(null);

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

    // Request downstream audio from agent even for text-first chat so the
    // SDP includes an audio m-line compatible with server-side WebRTC setup.
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
          } else if (state === "listening" || state === "recovered") {
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

    // Receive remote audio track if present; autoplay in hidden element.
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
      if (isStaleConnection() || pcRef.current !== pc) {
        return;
      }

      if (isClosedPeer(pc)) {
        return;
      }

      await pc.setLocalDescription(offer);
      if (isStaleConnection() || pcRef.current !== pc || isClosedPeer(pc)) {
        return;
      }

      const answer = await postOffer(offer);
      if (isStaleConnection() || pcRef.current !== pc || isClosedPeer(pc)) {
        return;
      }

      pcIdRef.current = answer.pc_id;

      try {
        if (isClosedPeer(pc)) {
          return;
        }
        await pc.setRemoteDescription({
          type: answer.type as RTCSdpType,
          sdp: answer.sdp,
        });
      } catch (error) {
        if (
          error instanceof DOMException &&
          error.name === "InvalidStateError" &&
          (pcRef.current !== pc || isClosedPeer(pc))
        ) {
          return;
        }
        throw error;
      }
    } finally {
      connectInFlightRef.current = false;
    }
  }

  function scheduleReconnect(): void {
    if (!mountedRef.current || stoppingRef.current || replacingConnectionRef.current) {
      return;
    }
    if (reconnectTimerRef.current !== null) return;

    const attempt = reconnectAttemptsRef.current + 1;
    reconnectAttemptsRef.current = attempt;
    const delayMs = Math.min(1000 * 2 ** (attempt - 1), 10000);

    reconnectTimerRef.current = window.setTimeout(() => {
      reconnectTimerRef.current = null;
      void connectPeer().catch((e: unknown) => {
        setErrorText(e instanceof Error ? e.message : "failed to reconnect");
        scheduleReconnect();
      });
    }, delayMs);
  }

  async function disconnectPeer(sendStop: boolean): Promise<void> {
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
        stoppingRef.current = true;
        dc.send(JSON.stringify({ type: "stream_stop", data: {} }));
      } catch {
        // ignore
      } finally {
        stoppingRef.current = false;
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
  }

  function handleSubmit(e: React.FormEvent): void {
    e.preventDefault();
    const text = input.trim();
    if (!text || connState === "thinking") return;

    const dc = dcRef.current;
    if (!dc || dc.readyState !== "open") {
      setErrorText("realtime connection is not ready");
      setConnState("disconnected");
      scheduleReconnect();
      return;
    }

    appendMessage("user", text);
    setInput("");
    setErrorText(null);
    setConnState("thinking");
    currentAssistantIdRef.current = null;

    try {
      dc.send(JSON.stringify({ type: "text", data: text }));
    } catch {
      setErrorText("failed to send message");
      setConnState("disconnected");
      scheduleReconnect();
    }
  }

  if (isPending) return null;

  return (
    <div className="h-full flex flex-col w-full px-6 sm:px-8">
      <header className="flex items-center justify-between pt-7 sm:pt-8 pb-4 shrink-0">
        <Button
          variant="aether-ghost"
          size="icon"
          onClick={() => router.push("/home")}
          className="w-8 h-8 min-w-[44px] min-h-[44px] -ml-2"
          aria-label="Go back"
        >
          <ChevronLeft className="size-[18px]" strokeWidth={1.5} />
        </Button>
        <span className="text-[11px] tracking-[0.18em] uppercase text-secondary-foreground font-normal">
          Chat
        </span>
        <div className="w-8 flex items-center justify-center">
          <StatusOrb status={orbStatus(connState)} size={10} />
        </div>
      </header>

      <Separator className="shrink-0 w-auto opacity-80" />

      <div className="flex-1 overflow-y-auto min-h-0 pt-6">
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <p className="text-muted-foreground text-xs">type a message to begin</p>
          </div>
        ) : (
          <div className="space-y-6 pb-3">
            {messages.map((m) => (
              <ChatMessage key={m.id} role={m.role} text={m.text} />
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {errorText && (
        <div className="py-2">
          <p className="text-[11px] text-red-400">{errorText}</p>
        </div>
      )}

      <form
        onSubmit={handleSubmit}
        className="flex items-center border border-border rounded-full bg-white/6 pb-safe mt-4 mb-2"
      >
        <Input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={isLoading ? "thinking..." : "type a message..."}
          disabled={isLoading}
          className="flex-1 bg-transparent border-0 rounded-none shadow-none px-4 py-3 text-base md:text-[13px] font-medium focus-visible:ring-0 disabled:opacity-50 h-auto leading-none"
          style={{ fontSize: "16px" }}
        />
        <Button
          type="submit"
          variant="aether-ghost"
          disabled={!input.trim() || isLoading}
          className="text-[10px] tracking-[0.1em] uppercase shrink-0 px-4 self-center"
        >
          send
        </Button>
      </form>
    </div>
  );
}
