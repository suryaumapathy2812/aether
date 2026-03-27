import { Suspense, useEffect, useState } from "react";
import { useNavigate, useLocation } from "@tanstack/react-router";
import { useTheme } from "#/components/ThemeProvider";
import { cn } from "#/lib/utils";
import { useSession } from "#/lib/auth-client";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "#/components/ui/tooltip";
import { Avatar, AvatarFallback, AvatarImage } from "#/components/ui/avatar";
import {
  IconMessage,
  IconMessage2Filled,
  IconBrain,
  IconDeviceMobile,
  IconDeviceMobileFilled,
  IconPlugConnected,
  IconSparkles,
  IconSun,
  IconMoon,
} from "@tabler/icons-react";

const NAV_ITEMS = [
  { href: "/devices", label: "Devices" },
  { href: "/memory", label: "Memory" },
  { href: "/integrations", label: "Integrations" },
  { href: "/skills", label: "Skills" },
] as const;

const HIDE_TOOLBAR = ["/"];
const HEADER_SHORTCUTS = {
  chat: ["G", "C"],
  sessions: ["Cmd", "B"],
  devices: ["G", "D"],
  memory: ["G", "M"],
  integrations: ["G", "P"],
  account: ["G", "S"],
} as const;

function NavIcon({ href, active }: { href: string; active: boolean }) {
  switch (href) {
    case "/devices":
      return active ? <IconDeviceMobileFilled className="size-5" /> : <IconDeviceMobile className="size-5" strokeWidth={1.5} />;
    case "/memory":
      return <IconBrain className="size-5" strokeWidth={active ? 2 : 1.5} />;
    case "/integrations":
      return <IconPlugConnected className="size-5" strokeWidth={active ? 2 : 1.5} />;
    case "/skills":
      return <IconSparkles className="size-5" strokeWidth={active ? 2 : 1.5} />;
    default:
      return null;
  }
}

function ShortcutBadge({
  keys,
  visible,
}: {
  keys: readonly string[];
  visible: boolean;
}) {
  return (
    <div
      className={cn(
        "pointer-events-none absolute -bottom-2 left-1/2 z-20 flex -translate-x-1/2 translate-y-full items-center gap-1 rounded-full border border-border/70 bg-background/95 px-1.5 py-1 shadow-lg backdrop-blur-sm transition-all duration-150",
        visible ? "opacity-100" : "opacity-0 translate-y-[calc(100%_-_4px)]",
      )}
      aria-hidden={!visible}
    >
      {keys.map((key) => (
        <kbd
          key={key}
          className="inline-flex h-4 min-w-4 items-center justify-center rounded border border-border/70 bg-muted/70 px-1 text-[9px] font-medium text-foreground/80 shadow-sm"
        >
          {key}
        </kbd>
      ))}
    </div>
  );
}

export default function FloatingToolbar() {
  return (
    <Suspense>
      <FloatingToolbarInner />
    </Suspense>
  );
}

function FloatingToolbarInner() {
  const navigate = useNavigate();
  const { pathname } = useLocation();
  const { data: session, isPending } = useSession();
  const { resolvedTheme, setTheme } = useTheme();
  const [showCommandHints, setShowCommandHints] = useState(false);

  useEffect(() => {
    const isMac = /Mac|iPhone|iPad|iPod/i.test(navigator.platform);
    if (!isMac) return;

    const syncHints = (event: KeyboardEvent) => {
      if (event.metaKey || event.getModifierState("Meta")) {
        setShowCommandHints(true);
        return;
      }
      setShowCommandHints(false);
    };

    const hideHints = () => {
      setShowCommandHints(false);
    };

    const handleKeyUp = (event: KeyboardEvent) => {
      if (event.key === "Meta" || !event.getModifierState("Meta")) {
        setShowCommandHints(false);
        return;
      }
      syncHints(event);
    };

    window.addEventListener("keydown", syncHints, true);
    window.addEventListener("keyup", handleKeyUp, true);
    document.addEventListener("keydown", syncHints, true);
    document.addEventListener("keyup", handleKeyUp, true);
    window.addEventListener("blur", hideHints);
    document.addEventListener("visibilitychange", hideHints);
    window.addEventListener("pointerup", hideHints, true);

    return () => {
      window.removeEventListener("keydown", syncHints, true);
      window.removeEventListener("keyup", handleKeyUp, true);
      document.removeEventListener("keydown", syncHints, true);
      document.removeEventListener("keyup", handleKeyUp, true);
      window.removeEventListener("blur", hideHints);
      document.removeEventListener("visibilitychange", hideHints);
      window.removeEventListener("pointerup", hideHints, true);
    };
  }, []);

  if (isPending || !session || pathname === "/login" || HIDE_TOOLBAR.includes(pathname)) {
    return null;
  }

  const toggleTheme = () =>
    setTheme(resolvedTheme === "dark" ? "light" : "dark");

  return (
    <>
      {/* Gradient fade from background to transparent */}
      <div className="fixed top-0 left-0 right-0 z-30 h-20 bg-gradient-to-b from-background to-transparent pointer-events-none" />

      <div className="fixed top-0 left-0 right-0 z-40 flex items-center justify-between px-4 py-3 pointer-events-none">
      {/* Left — logo (new chat) */}
      <div className="relative flex items-center pointer-events-auto">
        <button
          onClick={() => navigate({ to: "/chat" })}
          className="flex items-center justify-center rounded-lg transition-colors cursor-pointer"
          aria-label="Home"
        >
          <img
            src={
              resolvedTheme === "dark"
                ? "/icon-animated-white.svg"
                : "/icon-animated-black.svg"
            }
            alt="Aether"
            className="w-8 h-8"
          />
        </button>
        <ShortcutBadge
          keys={HEADER_SHORTCUTS.chat}
          visible={showCommandHints}
        />
      </div>

      {/* Right — nav icons + account */}
      <div className="flex items-center gap-1 pointer-events-auto">
        <div className="relative">
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                onClick={() => navigate({ to: "/sessions" })}
                className={cn(
                  "w-9 h-9 flex items-center justify-center rounded-lg text-foreground transition-colors",
                  pathname.startsWith("/sessions")
                    ? "bg-foreground text-background"
                    : "hover:bg-accent/60"
                )}
                aria-label="Sessions"
              >
                {pathname.startsWith("/sessions") ? (
                  <IconMessage2Filled className="size-5" />
                ) : (
                  <IconMessage className="size-5" strokeWidth={1.5} />
                )}
              </button>
            </TooltipTrigger>
            <TooltipContent side="bottom">Sessions</TooltipContent>
          </Tooltip>
          <ShortcutBadge
            keys={HEADER_SHORTCUTS.sessions}
            visible={showCommandHints}
          />
        </div>

        {NAV_ITEMS.map((item) => {
          const isActive = pathname.startsWith(item.href);
          const shortcutKey =
            item.href === "/devices"
              ? HEADER_SHORTCUTS.devices
              : item.href === "/memory"
                ? HEADER_SHORTCUTS.memory
                : item.href === "/integrations"
                  ? HEADER_SHORTCUTS.integrations
                  : null;
          return (
            <div key={item.href} className="relative">
              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    onClick={() => navigate({ to: item.href })}
                    className={cn(
                      "w-9 h-9 flex items-center justify-center rounded-lg text-foreground transition-colors",
                      isActive
                        ? "bg-foreground text-background"
                        : "hover:bg-accent/60"
                    )}
                    aria-label={item.label}
                  >
                    <NavIcon href={item.href} active={isActive} />
                  </button>
                </TooltipTrigger>
                <TooltipContent side="bottom">{item.label}</TooltipContent>
              </Tooltip>
              {shortcutKey && (
                <ShortcutBadge
                  keys={shortcutKey}
                  visible={showCommandHints}
                />
              )}
            </div>
          );
        })}

        {/* Theme toggle */}
        <Tooltip>
          <TooltipTrigger asChild>
            <button
              onClick={toggleTheme}
              className="w-9 h-9 flex items-center justify-center rounded-lg text-foreground transition-colors hover:bg-accent/60"
              aria-label="Toggle theme"
            >
              {resolvedTheme === "dark" ? (
                <IconSun className="size-5" strokeWidth={1.5} />
              ) : (
                <IconMoon className="size-5" strokeWidth={1.5} />
              )}
            </button>
          </TooltipTrigger>
          <TooltipContent side="bottom">Toggle theme</TooltipContent>
        </Tooltip>

        <div className="relative">
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                onClick={() => navigate({ to: "/account" })}
                className={cn(
                  "w-9 h-9 flex items-center justify-center rounded-lg transition-colors",
                  pathname.startsWith("/account")
                    ? "bg-foreground text-background"
                    : "text-foreground hover:bg-accent/60"
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
                  <AvatarFallback className="text-sm">
                    {(session?.user?.name || session?.user?.email || "U")
                      .charAt(0)
                      .toUpperCase()}
                  </AvatarFallback>
                </Avatar>
              </button>
            </TooltipTrigger>
            <TooltipContent side="bottom">Account</TooltipContent>
          </Tooltip>
          <ShortcutBadge
            keys={HEADER_SHORTCUTS.account}
            visible={showCommandHints}
          />
        </div>
      </div>
    </div>
    </>
  );
}
