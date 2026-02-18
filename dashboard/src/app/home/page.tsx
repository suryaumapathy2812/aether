"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import MenuList from "@/components/MenuList";
import { useSession } from "@/lib/auth-client";
import StatusOrb from "@/components/StatusOrb";
import { Separator } from "@/components/ui/separator";
import { useAgentStatus } from "@/hooks/useAgentStatus";

/**
 * Home — navigation hub. User greeting + menu items.
 */
export default function HomePage() {
  const router = useRouter();
  const { data: session, isPending } = useSession();

  useEffect(() => {
    if (!isPending && !session) {
      router.push("/");
    }
  }, [session, isPending, router]);

  const agentStatus = useAgentStatus();

  if (isPending || !session) return null;

  const name = session.user.name || session.user.email;

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-6">
      {/* User greeting */}
      <div className="mb-10 text-center">
        <p className="text-sm tracking-[0.1em] text-secondary-foreground font-light">
          {name}
        </p>
        <Separator className="w-16 mx-auto mt-3" />
      </div>

      {/* Menu */}
      <MenuList
        items={[
          { label: "Chat", href: "/chat" },
          { label: "Agent", href: "/agent" },
          { label: "Devices", href: "/devices" },
          { label: "Services", href: "/services" },
          { label: "Plugins", href: "/plugins" },
          { label: "Memory", href: "/memory" },
          { label: "Account", href: "/account" },
        ]}
      />

      {/* Brand + status */}
      <div className="mt-14 flex items-center gap-2">
        <span className="text-[10px] tracking-[0.3em] text-muted-foreground italic font-light">
          aether
        </span>
        <StatusOrb status={agentStatus} size={6} />
      </div>
    </div>
  );
}
