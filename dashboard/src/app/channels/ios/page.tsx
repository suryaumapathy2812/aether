"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { IconDeviceMobile } from "@tabler/icons-react";
import ContentShell from "@/components/ContentShell";
import MinimalInput from "@/components/MinimalInput";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { useSession } from "@/lib/auth-client";
import {
  claimPairingCode,
  listChannels,
  disconnectChannel,
  enableChannel,
  disableChannel,
  type ChannelInfo,
} from "@/lib/api";

export default function IOSChannelPage() {
  const router = useRouter();
  const { data: session, isPending } = useSession();
  const [channels, setChannels] = useState<ChannelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [claiming, setClaiming] = useState(false);
  const [pairingCode, setPairingCode] = useState("");

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
      setChannels(all.filter((c) => c.channel_type === "ios"));
    } catch {
      // Fail silently; page still usable
    } finally {
      setLoading(false);
    }
  }

  async function handleClaim(e: React.FormEvent) {
    e.preventDefault();
    const code = pairingCode.trim().toUpperCase();
    if (!code) {
      toast.error("Pairing code is required");
      return;
    }
    try {
      setClaiming(true);
      await claimPairingCode(code);
      toast.success("Device claimed. Your iOS app will pair in a few seconds.");
      setPairingCode("");
      await loadChannels();
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : "Failed to claim pairing code");
    } finally {
      setClaiming(false);
    }
  }

  async function handleDisconnect(channel: ChannelInfo) {
    try {
      await disconnectChannel(channel.id);
      toast.success("Device removed");
      await loadChannels();
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : "Failed to remove device");
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
      toast.error(e instanceof Error ? e.message : "Failed to update channel");
    }
  }

  if (isPending || !session) return null;

  return (
    <ContentShell title="iOS" back="/channels">
      <div className="space-y-6">
        <div>
          <p className="text-sm text-secondary-foreground leading-relaxed font-normal max-w-[78ch]">
            On your iPhone, open Aether and tap start pairing. Enter the code shown in the app here to link the
            device as an iOS channel.
          </p>
        </div>

        <form onSubmit={handleClaim} className="space-y-1">
          <MinimalInput
            label="Pairing Code"
            value={pairingCode}
            onChange={setPairingCode}
            placeholder="AETHER-XXXX-YYYY"
          />
          <p className="text-[10px] text-muted-foreground -mt-3 mb-4">
            Codes expire in 10 minutes. Keep the iOS app open until it shows paired.
          </p>
          <Button
            type="submit"
            variant="aether"
            size="aether"
            disabled={claiming || !pairingCode.trim()}
            className="w-full"
          >
            {claiming ? "claiming..." : "claim code"}
          </Button>
        </form>

        {!loading && channels.length > 0 && (
          <>
            <Separator />
            <div className="space-y-2">
              <h2 className="text-xs tracking-widest text-muted-foreground uppercase font-normal">
                Connected iOS devices
              </h2>
              {channels.map((channel) => (
                <IOSRow
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
    </ContentShell>
  );
}

function IOSRow({
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
        <IconDeviceMobile className="size-5 shrink-0 text-muted-foreground" strokeWidth={1.5} />
        <div className="min-w-0">
          <p className="text-[14px] text-foreground font-medium truncate">{channel.display_name}</p>
          <p className="text-[11px] text-muted-foreground">{channel.channel_id}</p>
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
