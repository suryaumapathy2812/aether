import { DefaultChatTransport } from "ai";
import { getDirectAgentConnection, getSessionToken } from "./api";

/**
 * AetherChatTransport connects useChat() to our /agent/v1/conversations/turn endpoint.
 *
 * Our backend emits AI SDK-compatible typed chunks (UIMessageChunk format):
 *   data: {"type":"start"}\n\n
 *   data: {"type":"text-delta","delta":"Got it..."}\n\n
 *   data: {"type":"tool-input-available","toolName":"search","toolCallId":"c1","input":{...}}\n\n
 *   data: {"type":"tool-output-available","toolCallId":"c1","output":"..."}\n\n
 *   data: {"type":"finish","finishReason":"stop"}\n\n
 *   data: [DONE]\n\n
 *
 * We extend DefaultChatTransport which already parses AI SDK SSE streams.
 * The only customization: inject our auth token and map the request body
 * to our backend's expected format.
 */
export function createAetherTransport(opts: { userId: string; sessionId?: string }) {
  const direct = getDirectAgentConnection();
  return new DefaultChatTransport({
    api: direct ? `${direct.baseUrl}/agent/v1/conversations/turn` : "/agent/v1/conversations/turn",
    credentials: "include",
    headers: () => {
      const token = getSessionToken();
      const h: Record<string, string> = {};
      if (direct?.directToken) {
        h["Authorization"] = `Bearer ${direct.directToken}`;
      } else if (token) {
        h["Authorization"] = `Bearer ${token}`;
      }
      return h;
    },
    prepareSendMessagesRequest: async ({ messages, body }) => {
      // Map AI SDK UIMessages → our backend's expected format.
      const mapped = messages.map((m) => {
        const textParts = m.parts
          .filter((p): p is Extract<typeof p, { type: "text" }> => p.type === "text")
          .map((p) => p.text)
          .join("");
        return { role: m.role, content: textParts || "" };
      });

      return {
        body: {
          messages: mapped,
          user: opts.userId,
          session: opts.sessionId || "",
          ...((body as Record<string, unknown>) || {}),
        },
      };
    },
  });
}
