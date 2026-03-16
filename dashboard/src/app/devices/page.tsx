"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import ContentShell from "@/components/ContentShell";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import {
  listDevices,
  listChannels,
  registerTelegramDevice,
  deleteDevice,
  claimPairingCode,
  disconnectChannel,
  enableChannel,
  disableChannel,
  type Device,
  type ChannelInfo,
} from "@/lib/api";
import { useSession } from "@/lib/auth-client";
import {
  IconBrandTelegram,
  IconDeviceMobile,
  IconChevronRight,
  IconTrash,
} from "@tabler/icons-react";

type Tab = "connected" | "browse";

export default function DevicesPage() {
  const router = useRouter();
  const { data: session, isPending } = useSession();
  const [tab, setTab] = useState<Tab>("connected");
  const [devices, setDevices] = useState<Device[]>([]);
  const [channels, setChannels] = useState<ChannelInfo[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (isPending) return;
    if (!session) { router.push("/"); return; }
    loadAll();
  }, [session, isPending, router]);

  async function loadAll() {
    setLoading(true);
    try {
      const [devs, chans] = await Promise.all([
        listDevices().catch(() => [] as Device[]),
        listChannels(session?.user?.id).catch(() => [] as ChannelInfo[]),
      ]);
      setDevices(devs);
      setChannels(chans);
    } finally {
      setLoading(false);
    }
  }

  if (isPending || !session) return null;

  const connectedDevices = devices;
  const iosChannels = channels.filter((c) => c.channel_type === "ios");
  const telegramChannels = channels.filter((c) => c.channel_type === "telegram");
  const hasConnected = connectedDevices.length > 0 || iosChannels.length > 0 || telegramChannels.length > 0;

  return (
    <ContentShell title="Devices">
      {/* Tabs */}
      <div className="flex gap-4 mb-6">
        <button
          onClick={() => setTab("connected")}
          className={`text-[13px] pb-1.5 transition-colors ${tab === "connected" ? "text-foreground border-b border-foreground" : "text-muted-foreground hover:text-foreground/70"}`}
        >
          Connected
        </button>
        <button
          onClick={() => setTab("browse")}
          className={`text-[13px] pb-1.5 transition-colors ${tab === "browse" ? "text-foreground border-b border-foreground" : "text-muted-foreground hover:text-foreground/70"}`}
        >
          Browse
        </button>
      </div>

      {loading ? (
        <p className="text-muted-foreground/60 text-xs">loading...</p>
      ) : tab === "connected" ? (
        <ConnectedTab
          devices={connectedDevices}
          iosChannels={iosChannels}
          telegramChannels={telegramChannels}
          hasConnected={hasConnected}
          onDelete={async (id) => {
            await deleteDevice(id).catch(() => {});
            toast.success("Device removed");
            loadAll();
          }}
          onDisconnectChannel={async (id) => {
            await disconnectChannel(id).catch(() => {});
            toast.success("Disconnected");
            loadAll();
          }}
          onToggleChannel={async (id, enabled) => {
            if (enabled) await enableChannel(id).catch(() => {});
            else await disableChannel(id).catch(() => {});
            loadAll();
          }}
        />
      ) : (
        <BrowseTab onSetup={loadAll} userId={session.user.id} />
      )}
    </ContentShell>
  );
}

// ── Connected tab ──

function ConnectedTab({
  devices,
  iosChannels,
  telegramChannels,
  hasConnected,
  onDelete,
  onDisconnectChannel,
  onToggleChannel,
}: {
  devices: Device[];
  iosChannels: ChannelInfo[];
  telegramChannels: ChannelInfo[];
  hasConnected: boolean;
  onDelete: (id: string) => void;
  onDisconnectChannel: (id: string) => void;
  onToggleChannel: (id: string, enabled: boolean) => void;
}) {
  if (!hasConnected) {
    return (
      <p className="text-muted-foreground text-xs">
        No devices connected yet. Go to Browse to add one.
      </p>
    );
  }

  return (
    <div className="space-y-1">
      {devices.map((d) => (
        <div key={d.id} className="flex items-center justify-between py-3 group">
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2.5">
              <IconBrandTelegram className="size-4 text-muted-foreground shrink-0" />
              <span className="text-[13px] text-foreground font-medium">{d.name}</span>
            </div>
            <p className="text-[11px] text-muted-foreground mt-0.5 ml-6.5">
              {d.device_type}{d.paired_at && ` · ${new Date(d.paired_at).toLocaleDateString()}`}
            </p>
          </div>
          <button
            onClick={() => onDelete(d.id)}
            className="shrink-0 p-1.5 rounded text-muted-foreground/40 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-all"
          >
            <IconTrash className="size-3.5" />
          </button>
        </div>
      ))}

      {iosChannels.map((c) => (
        <div key={c.id} className="flex items-center justify-between py-3 group">
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2.5">
              <IconDeviceMobile className="size-4 text-muted-foreground shrink-0" />
              <span className="text-[13px] text-foreground font-medium">{c.display_name || "iOS"}</span>
              <span className={`w-1.5 h-1.5 rounded-full ${c.enabled ? "bg-emerald-400" : "bg-muted-foreground/40"}`} />
            </div>
          </div>
          <div className="flex items-center gap-3 shrink-0">
            <Switch
              checked={c.enabled}
              onCheckedChange={(v) => onToggleChannel(c.id, v)}
              size="sm"
            />
            <button
              onClick={() => onDisconnectChannel(c.id)}
              className="p-1.5 rounded text-muted-foreground/40 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-all"
            >
              <IconTrash className="size-3.5" />
            </button>
          </div>
        </div>
      ))}

      {telegramChannels.map((c) => (
        <div key={c.id} className="flex items-center justify-between py-3 group">
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2.5">
              <IconBrandTelegram className="size-4 text-muted-foreground shrink-0" />
              <span className="text-[13px] text-foreground font-medium">{c.display_name || "Telegram"}</span>
              <span className={`w-1.5 h-1.5 rounded-full ${c.enabled ? "bg-emerald-400" : "bg-muted-foreground/40"}`} />
            </div>
          </div>
          <div className="flex items-center gap-3 shrink-0">
            <Switch
              checked={c.enabled}
              onCheckedChange={(v) => onToggleChannel(c.id, v)}
              size="sm"
            />
            <button
              onClick={() => onDisconnectChannel(c.id)}
              className="p-1.5 rounded text-muted-foreground/40 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-all"
            >
              <IconTrash className="size-3.5" />
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Browse tab ──

function BrowseTab({ onSetup, userId }: { onSetup: () => void; userId: string }) {
  return (
    <div className="space-y-8">
      <IOSSetup onDone={onSetup} />
      <TelegramSetup onDone={onSetup} />
    </div>
  );
}

function IOSSetup({ onDone }: { onDone: () => void }) {
  const [code, setCode] = useState("");
  const [claiming, setClaiming] = useState(false);

  async function handleClaim(e: React.FormEvent) {
    e.preventDefault();
    const trimmed = code.trim().toUpperCase();
    if (!trimmed) { toast.error("Pairing code is required"); return; }
    setClaiming(true);
    try {
      await claimPairingCode(trimmed);
      toast.success("Device paired successfully");
      setCode("");
      onDone();
    } catch (err: unknown) {
      toast.error(err instanceof Error ? err.message : "Failed to pair");
    } finally {
      setClaiming(false);
    }
  }

  return (
    <div>
      <div className="flex items-center gap-2.5 mb-1">
        <IconDeviceMobile className="size-4 text-muted-foreground" />
        <p className="text-[13px] text-foreground font-medium">iOS App</p>
      </div>
      <p className="text-[11px] text-muted-foreground mb-4">
        Pair your iPhone by entering the code shown in the Aether iOS app.
      </p>
      <form onSubmit={handleClaim} className="flex gap-2">
        <input
          value={code}
          onChange={(e) => setCode(e.target.value.toUpperCase())}
          placeholder="XXXX-XXXX"
          maxLength={9}
          className="flex-1 h-8 px-3 text-[13px] bg-background border border-input rounded-md focus:outline-none focus:ring-1 focus:ring-ring text-foreground placeholder:text-muted-foreground/40 font-mono tracking-wider"
        />
        <Button variant="ghost" size="sm" type="submit" disabled={claiming} className="h-8 px-3 text-[12px]">
          {claiming ? "..." : "Pair"}
        </Button>
      </form>
    </div>
  );
}

function TelegramSetup({ onDone }: { onDone: () => void }) {
  const [botToken, setBotToken] = useState("");
  const [chatIds, setChatIds] = useState("");
  const [name, setName] = useState("");
  const [registering, setRegistering] = useState(false);

  async function handleRegister(e: React.FormEvent) {
    e.preventDefault();
    if (!botToken.trim()) { toast.error("Bot token is required"); return; }
    setRegistering(true);
    try {
      await registerTelegramDevice(botToken.trim(), chatIds.trim() || undefined, name.trim() || undefined);
      toast.success("Telegram connected");
      setBotToken("");
      setChatIds("");
      setName("");
      onDone();
    } catch (err: unknown) {
      toast.error(err instanceof Error ? err.message : "Failed to connect");
    } finally {
      setRegistering(false);
    }
  }

  return (
    <div>
      <div className="flex items-center gap-2.5 mb-1">
        <IconBrandTelegram className="size-4 text-muted-foreground" />
        <p className="text-[13px] text-foreground font-medium">Telegram Bot</p>
      </div>
      <p className="text-[11px] text-muted-foreground mb-4">
        Connect a Telegram bot to chat with Aether from Telegram. Create one with @BotFather first.
      </p>
      <form onSubmit={handleRegister} className="space-y-3">
        <div>
          <label className="text-[12px] text-muted-foreground mb-1.5 block">
            Bot Token <span className="text-red-400/60">*</span>
          </label>
          <input
            type="password"
            value={botToken}
            onChange={(e) => setBotToken(e.target.value)}
            placeholder="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
            className="w-full h-8 px-3 text-[13px] bg-background border border-input rounded-md focus:outline-none focus:ring-1 focus:ring-ring text-foreground placeholder:text-muted-foreground/40"
          />
        </div>
        <div>
          <label className="text-[12px] text-muted-foreground mb-1.5 block">Allowed Chat IDs</label>
          <input
            value={chatIds}
            onChange={(e) => setChatIds(e.target.value)}
            placeholder="123456789, 987654321"
            className="w-full h-8 px-3 text-[13px] bg-background border border-input rounded-md focus:outline-none focus:ring-1 focus:ring-ring text-foreground placeholder:text-muted-foreground/40"
          />
          <p className="text-[10px] text-muted-foreground/60 mt-1">Comma-separated. Leave empty to allow any chat.</p>
        </div>
        <div>
          <label className="text-[12px] text-muted-foreground mb-1.5 block">Device Name</label>
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="My Telegram Bot"
            className="w-full h-8 px-3 text-[13px] bg-background border border-input rounded-md focus:outline-none focus:ring-1 focus:ring-ring text-foreground placeholder:text-muted-foreground/40"
          />
        </div>
        <Button variant="ghost" size="sm" type="submit" disabled={registering} className="h-8 px-3 text-[12px]">
          {registering ? "..." : "Connect"}
        </Button>
      </form>
    </div>
  );
}
