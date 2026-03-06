"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import PageShell from "@/components/PageShell";
import MinimalInput from "@/components/MinimalInput";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { useSession } from "@/lib/auth-client";
import { IconBrandTelegram } from "@tabler/icons-react";
import {
  listChannels,
  connectTelegram,
  disconnectChannel,
  enableChannel,
  disableChannel,
  type ChannelInfo,
} from "@/lib/api";

export default function TelegramChannelPage() {
  const router = useRouter();
  const { data: session, isPending } = useSession();
  const [channels, setChannels] = useState<ChannelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [connecting, setConnecting] = useState(false);

  // Form
  const [botToken, setBotToken] = useState("");
  const [chatId, setChatId] = useState("");

  useEffect(() => {
    if (isPending) return;
    if (!session) {
      router.push("/");
      return;
    }
    loadChannels();
  }, [session, isPending, router]);

  async function loadChannels() {
    try {
      setLoading(true);
      const all = await listChannels(session?.user?.id);
      setChannels(all.filter((c) => c.channel_type === "telegram"));
    } catch {
      // fail silently — empty list shown
    } finally {
      setLoading(false);
    }
  }

  async function handleConnect(e: React.FormEvent) {
    e.preventDefault();
    if (!botToken.trim()) {
      toast.error("Bot token is required");
      return;
    }
    if (!chatId.trim()) {
      toast.error("Chat ID is required");
      return;
    }

    try {
      setConnecting(true);
      const result = await connectTelegram({
        user_id: session?.user?.id,
        bot_token: botToken.trim(),
        chat_id: chatId.trim(),
      });
      toast.success(
        `Connected @${result.bot_info?.username || "bot"}`
      );
      setBotToken("");
      setChatId("");
      await loadChannels();
    } catch (e: unknown) {
      toast.error(
        e instanceof Error ? e.message : "Failed to connect"
      );
    } finally {
      setConnecting(false);
    }
  }

  async function handleDisconnect(channel: ChannelInfo) {
    try {
      await disconnectChannel(channel.id);
      toast.success("Disconnected");
      await loadChannels();
    } catch (e: unknown) {
      toast.error(
        e instanceof Error ? e.message : "Failed to disconnect"
      );
    }
  }

  async function handleToggle(channel: ChannelInfo) {
    try {
      if (channel.enabled) {
        await disableChannel(channel.id);
        toast.success("Turned off");
      } else {
        await enableChannel(channel.id);
        toast.success("Turned on");
      }
      await loadChannels();
    } catch (e: unknown) {
      toast.error(
        e instanceof Error ? e.message : "Failed to update"
      );
    }
  }

  if (isPending || !session) return null;

  return (
    <PageShell title="Telegram" back="/channels">
      <div className="space-y-6">
        {/* Description */}
        <div>
          <p className="text-sm text-secondary-foreground leading-relaxed font-normal max-w-[78ch]">
            Connect a Telegram bot so you can message Aether directly from
            Telegram. You&apos;ll need a bot token from{" "}
            <a
              href="https://t.me/BotFather"
              target="_blank"
              rel="noopener noreferrer"
              className="underline underline-offset-2"
            >
              @BotFather
            </a>{" "}
            and your chat ID.
          </p>
        </div>

        {/* Connect form */}
        <form onSubmit={handleConnect} className="space-y-1">
          <MinimalInput
            label="Bot Token"
            type="password"
            value={botToken}
            onChange={setBotToken}
            placeholder="paste your bot token"
          />
          <MinimalInput
            label="Chat ID"
            value={chatId}
            onChange={setChatId}
            placeholder="your telegram chat id"
          />
          <p className="text-[10px] text-muted-foreground -mt-3 mb-4">
            Send any message to your bot, then visit{" "}
            <a
              href="https://t.me/userinfobot"
              target="_blank"
              rel="noopener noreferrer"
              className="underline underline-offset-2"
            >
              @userinfobot
            </a>{" "}
            to find your chat ID.
          </p>
          <Button
            type="submit"
            variant="aether"
            size="aether"
            disabled={connecting || !botToken.trim() || !chatId.trim()}
            className="w-full"
          >
            {connecting ? "connecting..." : "connect"}
          </Button>
        </form>

        {/* Connected bots */}
        {!loading && channels.length > 0 && (
          <>
            <Separator />
            <div className="space-y-2">
              <h2 className="text-xs tracking-widest text-muted-foreground uppercase font-normal">
                Connected bots
              </h2>
              {channels.map((channel) => (
                <TelegramRow
                  key={channel.id}
                  channel={channel}
                  onToggle={() => handleToggle(channel)}
                  onDisconnect={() => handleDisconnect(channel)}
                />
              ))}
            </div>
          </>
        )}
      </div>
    </PageShell>
  );
}

function TelegramRow({
  channel,
  onToggle,
  onDisconnect,
}: {
  channel: ChannelInfo;
  onToggle: () => void;
  onDisconnect: () => void;
}) {
  const [busy, setBusy] = useState(false);

  async function handleDisconnect() {
    setBusy(true);
    await onDisconnect();
    setBusy(false);
  }

  return (
    <div className="flex items-center justify-between gap-3 rounded-2xl bg-white/5 border border-border/60 px-4 py-3">
      <div className="flex items-center gap-3 min-w-0">
        <IconBrandTelegram className="size-5 shrink-0 text-muted-foreground" strokeWidth={1.5} />
        <div className="min-w-0">
          <p className="text-[14px] text-foreground font-medium truncate">
            {channel.display_name}
          </p>
          <p className="text-[11px] text-muted-foreground">
            {channel.channel_id}
          </p>
        </div>
      </div>

      <div className="flex items-center gap-2 shrink-0">
        <button
          onClick={onToggle}
          className={`text-[11px] tracking-wider px-2.5 py-1 rounded-full transition-colors ${
            channel.enabled
              ? "bg-green-500/10 text-green-400 hover:bg-green-500/20"
              : "bg-muted/50 text-muted-foreground hover:bg-muted"
          }`}
        >
          {channel.enabled ? "on" : "off"}
        </button>
        <button
          onClick={handleDisconnect}
          disabled={busy}
          className="text-[11px] tracking-wider px-2.5 py-1 rounded-full bg-red-500/10 text-red-400 hover:bg-red-500/20 transition-colors"
        >
          {busy ? "..." : "remove"}
        </button>
      </div>
    </div>
  );
}
