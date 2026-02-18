"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import MenuList from "@/components/MenuList";
import { useSession } from "@/lib/auth-client";

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
    <div className="min-h-screen flex flex-col items-center justify-center px-6">
      {/* User greeting */}
      <div className="mb-10 text-center">
        <p className="text-sm tracking-[0.1em] text-[var(--color-text-secondary)] font-light">
          {name}
        </p>
        <div className="w-16 h-px bg-[var(--color-border)] mx-auto mt-3" />
      </div>

      {/* Menu */}
      <MenuList
        items={[
          { label: "Chat", href: "/chat" },
          { label: "Devices", href: "/devices" },
          { label: "Services", href: "/services" },
          { label: "Memory", href: "/memory" },
          { label: "Account", href: "/account" },
        ]}
      />

      {/* Brand */}
      <div className="mt-14">
        <span className="text-[10px] tracking-[0.3em] text-[var(--color-text-muted)] italic font-light">
          aether
        </span>
      </div>
    </div>
  );
}
