"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { toast } from "sonner";
import ContentShell from "@/components/ContentShell";
import { Switch } from "@/components/ui/switch";
import {
  listDevices,
  listChannels,
  deleteDevice,
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

const DEVICE_TYPES = [
  {
    type: "ios",
    name: "iOS App",
    description: "Pair your iPhone with Aether using a one-time code",
    icon: IconDeviceMobile,
    href: "/devices/ios",
  },
  {
    type: "telegram",
    name: "Telegram Bot",
    description: "Connect a Telegram bot to chat with Aether",
    icon: IconBrandTelegram,
    href: "/devices/telegram",
  },
] as const;

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
        <BrowseTab />
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

function BrowseTab() {
  return (
    <div className="space-y-1">
      {DEVICE_TYPES.map((device) => {
        const Icon = device.icon;
        return (
          <Link
            key={device.type}
            href={device.href}
            className="group flex items-center gap-3 px-3 py-3 -mx-3 rounded-xl hover:bg-white/[0.03] transition-colors"
          >
            <Icon className="size-4 text-muted-foreground shrink-0" strokeWidth={1.5} />
            <div className="flex-1 min-w-0">
              <span className="text-[13px] font-medium text-foreground">
                {device.name}
              </span>
              <p className="text-[11px] text-muted-foreground/60 mt-0.5 line-clamp-1">
                {device.description}
              </p>
            </div>
            <IconChevronRight className="size-3.5 text-muted-foreground/20 group-hover:text-muted-foreground/40 transition-colors shrink-0" />
          </Link>
        );
      })}
    </div>
  );
}
