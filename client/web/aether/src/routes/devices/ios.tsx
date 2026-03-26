import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { useNavigate } from "@tanstack/react-router";
import { toast } from "sonner";
import ContentShell from "#/components/ContentShell";
import MinimalInput from "#/components/MinimalInput";
import { Button } from "#/components/ui/button";
import { Separator } from "#/components/ui/separator";
import { useSession } from "#/lib/auth-client";
import { IconDeviceMobile } from "@tabler/icons-react";
import {
  claimPairingCode,
  listChannels,
  disconnectChannel,
  enableChannel,
  disableChannel,
  type ChannelInfo,
} from "#/lib/api";

export const Route = createFileRoute("/devices/ios")({
  component: IOSDevicePage,
});

function IOSDevicePage() {
  const navigate = useNavigate();
  const { data: session, isPending } = useSession();
  const [channels, setChannels] = useState<ChannelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [claiming, setClaiming] = useState(false);
  const [pairingCode, setPairingCode] = useState("");

  useEffect(() => {
    if (isPending) return;
    if (!session) {
      navigate({ to: "/" });
      return;
    }
    loadChannels();
  }, [session, isPending, navigate]);

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
      toast.error(
        e instanceof Error ? e.message : "Failed to claim pairing code"
      );
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
      toast.error(
        e instanceof Error ? e.message : "Failed to update channel"
      );
    }
  }

  if (isPending || !session) return null;

  return (
    <ContentShell title="iOS App" back="/devices">
      <div className="space-y-6">
        <div className="flex items-start gap-3">
          <IconDeviceMobile
            className="size-5 shrink-0 text-muted-foreground mt-0.5"
            strokeWidth={1.5}
          />
          <p className="text-sm text-muted-foreground leading-relaxed max-w-[72ch]">
            On your iPhone, open Aether and tap start pairing. Enter the code
            shown in the app here to link the device.
          </p>
        </div>

        <form onSubmit={handleClaim} className="space-y-1">
          <MinimalInput
            label="Pairing Code"
            value={pairingCode}
            onChange={(v) => setPairingCode(v.toUpperCase())}
            placeholder="XXXX-XXXX"
          />
          <p className="text-xs text-muted-foreground -mt-3 mb-4">
            Codes expire in 10 minutes. Keep the iOS app open until it shows
            paired.
          </p>
          <Button
            type="submit"
            variant="aether"
            size="aether"
            disabled={claiming || !pairingCode.trim()}
            className="w-full"
          >
            {claiming ? "claiming..." : "pair device"}
          </Button>
        </form>

        {!loading && channels.length > 0 && (
          <>
            <Separator />
            <div className="space-y-2">
              <h2 className="text-xs tracking-[0.15em] text-muted-foreground uppercase font-normal">
                Connected devices
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
    <div className="flex items-center justify-between gap-3 rounded-2xl bg-accent/30 border border-border px-4 py-3">
      <div className="flex items-center gap-3 min-w-0">
        <IconDeviceMobile
          className="size-4 shrink-0 text-muted-foreground"
          strokeWidth={1.5}
        />
        <div className="min-w-0">
          <p className="text-sm text-foreground font-medium truncate">
            {channel.display_name}
          </p>
          <p className="text-sm text-muted-foreground/60">
            {channel.channel_id}
          </p>
        </div>
      </div>

      <div className="flex items-center gap-2 shrink-0">
        <button
          onClick={onToggle}
          className={`text-sm tracking-wider px-2.5 py-1 rounded-full transition-colors ${
            channel.enabled
              ? "bg-emerald-500/10 text-emerald-400 hover:bg-emerald-500/20"
              : "bg-accent/40 text-muted-foreground hover:bg-accent/80"
          }`}
        >
          {channel.enabled ? "on" : "off"}
        </button>
        <button
          onClick={handleDisconnect}
          disabled={busy}
          className="text-sm tracking-wider px-2.5 py-1 rounded-full bg-red-500/10 text-red-400 hover:bg-red-500/20 transition-colors"
        >
          {busy ? "..." : "remove"}
        </button>
      </div>
    </div>
  );
}
