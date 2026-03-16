"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import ContentShell from "@/components/ContentShell";
import { useSession } from "@/lib/auth-client";
import { listChannels, type ChannelInfo } from "@/lib/api";
import {
  IconBrandTelegram,
  IconBrandWhatsapp,
  IconBrandSlack,
  IconDeviceMobile,
} from "@tabler/icons-react";

/** Available channel types — add new ones here as they're implemented. */
const CHANNEL_TYPES = [
  {
    type: "telegram",
    name: "Telegram",
    description: "Message Aether through a Telegram bot",
    icon: IconBrandTelegram,
    href: "/channels/telegram",
    available: true,
  },
  {
    type: "ios",
    name: "iOS",
    description: "Pair your iPhone with Aether using a one-time code",
    icon: IconDeviceMobile,
    href: "/channels/ios",
    available: true,
  },
  {
    type: "whatsapp",
    name: "WhatsApp",
    description: "Connect WhatsApp for messaging",
    icon: IconBrandWhatsapp,
    href: "/channels/whatsapp",
    available: false,
  },
  {
    type: "slack",
    name: "Slack",
    description: "Bring Aether into your Slack workspace",
    icon: IconBrandSlack,
    href: "/channels/slack",
    available: false,
  },
] as const;

export default function ChannelsPage() {
  const router = useRouter();
  const { data: session, isPending } = useSession();
  const [channels, setChannels] = useState<ChannelInfo[]>([]);
  const [loading, setLoading] = useState(true);

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
      const data = await listChannels(session?.user?.id);
      setChannels(data);
    } catch {
      // Silently fail — hub page still shows channel types
    } finally {
      setLoading(false);
    }
  }

  if (isPending || !session) return null;

  /** Count connected channels per type */
  function connectedCount(type: string): number {
    return channels.filter(
      (c) => c.channel_type === type && c.enabled
    ).length;
  }

  return (
    <ContentShell title="Channels" back="/home">
      <div className="space-y-2">
        <p className="text-[12px] text-muted-foreground mb-1">
          Connect messaging platforms so you can talk to Aether from anywhere.
        </p>

        {CHANNEL_TYPES.map((ch) => {
          const count = connectedCount(ch.type);

          const Icon = ch.icon;

          return (
            <div
              key={ch.type}
              className="py-4 px-3 rounded-2xl border border-border/60 bg-white/5 flex items-start justify-between gap-4"
            >
              {ch.available ? (
                <Link
                  href={ch.href}
                  className="block group min-w-0 flex-1"
                >
                  <div className="flex items-start gap-3">
                    <Icon className="size-5 shrink-0 text-muted-foreground group-hover:text-secondary-foreground transition-colors duration-300" strokeWidth={1.5} />
                    <div className="min-w-0">
                      <h3 className="text-[14px] text-foreground group-hover:text-secondary-foreground transition-colors duration-300 font-medium mb-1">
                        {ch.name}
                      </h3>
                      <p className="text-[12px] text-muted-foreground leading-relaxed font-normal line-clamp-2">
                        {ch.description}
                      </p>
                    </div>
                  </div>
                </Link>
              ) : (
                <div className="flex items-start gap-3 min-w-0 flex-1 opacity-50">
                  <Icon className="size-5 shrink-0 text-muted-foreground" strokeWidth={1.5} />
                  <div className="min-w-0">
                    <h3 className="text-[14px] text-foreground font-medium mb-1">
                      {ch.name}
                    </h3>
                    <p className="text-[12px] text-muted-foreground leading-relaxed font-normal line-clamp-2">
                      {ch.description}
                    </p>
                  </div>
                </div>
              )}

              <div className="flex items-start justify-end shrink-0 pt-0.5">
                {!ch.available ? (
                  <span className="text-[11px] tracking-wider px-2.5 py-1 rounded-full bg-muted/50 text-muted-foreground">
                    coming soon
                  </span>
                ) : loading ? (
                  <span className="text-[11px] tracking-wider px-2.5 py-1 rounded-full bg-muted/50 text-muted-foreground">
                    ...
                  </span>
                ) : count > 0 ? (
                  <Link href={ch.href}>
                    <span className="text-[11px] tracking-wider px-2.5 py-1 rounded-full bg-green-500/10 text-green-400">
                      {count} connected
                    </span>
                  </Link>
                ) : (
                  <Link
                    href={ch.href}
                    className="text-[11px] tracking-wider px-2.5 py-1 rounded-full bg-secondary/10 text-secondary-foreground hover:bg-secondary/20 transition-colors duration-300"
                  >
                    set up
                  </Link>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </ContentShell>
  );
}
