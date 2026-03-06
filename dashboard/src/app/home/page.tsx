"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import MenuList from "@/components/MenuList";
import { useSession } from "@/lib/auth-client";
import { Separator } from "@/components/ui/separator";
import NotificationBell from "@/components/NotificationBell";

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

  if (isPending || !session) return null;

  const name = session.user.name || session.user.email;

  return (
    <div className="h-full relative flex flex-col items-center justify-center px-6 sm:px-8">
      <div className="absolute top-7 left-6 sm:top-8 sm:left-8 z-10">
        <span className="logo-wordmark text-[10px] text-muted-foreground font-medium">
          aether
        </span>
      </div>

      <div className="absolute top-7 right-6 sm:top-8 sm:right-8 z-10">
        <NotificationBell />
      </div>

      {/* User greeting */}
      <div className="mb-12 text-center">
        <p className="text-xs tracking-[0.18em] uppercase text-muted-foreground font-medium mb-2">
          Welcome back
        </p>
        <p className="text-lg tracking-[0.02em] text-secondary-foreground font-medium">
          {name}
        </p>
        <Separator className="w-20 mx-auto mt-3 opacity-80" />
      </div>

      {/* Menu */}
      <MenuList
        items={[
          { label: "Chat", href: "/chat" },
          { label: "Agent", href: "/agent" },
          // Notifications entry intentionally hidden; access via bell icon.
          { label: "Memory", href: "/memory" },
          { label: "Channels", href: "/channels" },
          { label: "Plugins", href: "/plugins" },
          { label: "Account", href: "/account" },
        ]}
      />
    </div>
  );
}
