"use client";

import { Suspense } from "react";
import { useRouter, usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { useSession } from "@/lib/auth-client";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import {
  IconMessage,
  IconMessage2Filled,
  IconBrain,
  IconDeviceMobile,
  IconDeviceMobileFilled,
  IconPlugConnected,
  IconSparkles,
} from "@tabler/icons-react";

const NAV_ITEMS = [
  { href: "/devices", label: "Devices" },
  { href: "/memory", label: "Memory" },
  { href: "/plugins", label: "Plugins" },
  { href: "/skills", label: "Skills" },
] as const;

const HIDE_TOOLBAR = ["/"];

function NavIcon({ href, active }: { href: string; active: boolean }) {
  switch (href) {
    case "/devices":
      return active ? <IconDeviceMobileFilled className="size-4" /> : <IconDeviceMobile className="size-4" strokeWidth={1.5} />;
    case "/memory":
      return <IconBrain className="size-4" strokeWidth={active ? 2 : 1.5} />;
    case "/plugins":
      return <IconPlugConnected className="size-4" strokeWidth={active ? 2 : 1.5} />;
    case "/skills":
      return <IconSparkles className="size-4" strokeWidth={active ? 2 : 1.5} />;
    default:
      return null;
  }
}

export default function FloatingToolbar() {
  return (
    <Suspense>
      <FloatingToolbarInner />
    </Suspense>
  );
}

function FloatingToolbarInner() {
  const router = useRouter();
  const pathname = usePathname();
  const { data: session } = useSession();

  if (HIDE_TOOLBAR.includes(pathname)) return null;

  return (
    <div className="fixed top-0 left-0 right-0 z-40 flex items-center justify-between px-4 py-3 pointer-events-none">
      {/* Left — logo (new chat), sessions */}
      <div className="flex items-center gap-0.5 pointer-events-auto">
        <button
          onClick={() => router.push("/chat")}
          className="logo-wordmark text-[13px] font-bold text-foreground/40 hover:text-foreground/70 transition-colors px-1.5 cursor-pointer"
        >
          aether
        </button>

        <Tooltip>
          <TooltipTrigger asChild>
            <button
              onClick={() => router.push("/sessions")}
              className={cn(
                "w-8 h-8 flex items-center justify-center rounded-lg transition-colors",
                pathname.startsWith("/sessions")
                  ? "text-foreground bg-white/[0.06]"
                  : "text-muted-foreground hover:text-foreground hover:bg-white/[0.06]"
              )}
              aria-label="Sessions"
            >
              {pathname.startsWith("/sessions") ? (
                <IconMessage2Filled className="size-4" />
              ) : (
                <IconMessage className="size-4" strokeWidth={1.5} />
              )}
            </button>
          </TooltipTrigger>
          <TooltipContent side="bottom">Sessions</TooltipContent>
        </Tooltip>
      </div>

      {/* Right — nav icons + account */}
      <div className="flex items-center gap-1 pointer-events-auto">
        {NAV_ITEMS.map((item) => {
          const isActive = pathname.startsWith(item.href);
          return (
            <Tooltip key={item.href}>
              <TooltipTrigger asChild>
                <button
                  onClick={() => router.push(item.href)}
                  className={cn(
                    "w-8 h-8 flex items-center justify-center rounded-lg transition-colors",
                    isActive
                      ? "text-foreground bg-white/[0.06]"
                      : "text-muted-foreground hover:text-foreground hover:bg-white/[0.06]"
                  )}
                  aria-label={item.label}
                >
                  <NavIcon href={item.href} active={isActive} />
                </button>
              </TooltipTrigger>
              <TooltipContent side="bottom">{item.label}</TooltipContent>
            </Tooltip>
          );
        })}

        <Tooltip>
          <TooltipTrigger asChild>
            <button
              onClick={() => router.push("/account")}
              className={cn(
                "w-8 h-8 flex items-center justify-center rounded-lg transition-colors ml-1",
                pathname.startsWith("/account")
                  ? "bg-white/[0.06]"
                  : "hover:bg-white/[0.06]"
              )}
              aria-label="Account"
            >
              <Avatar size="sm">
                {session?.user?.image && (
                  <AvatarImage
                    src={session.user.image}
                    alt={session?.user?.name || ""}
                  />
                )}
                <AvatarFallback className="text-[11px]">
                  {(session?.user?.name || session?.user?.email || "U")
                    .charAt(0)
                    .toUpperCase()}
                </AvatarFallback>
              </Avatar>
            </button>
          </TooltipTrigger>
          <TooltipContent side="bottom">Account</TooltipContent>
        </Tooltip>
      </div>
    </div>
  );
}
